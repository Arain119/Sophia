from __future__ import annotations

from collections.abc import Mapping

import torch

from ml.runtime.model.attention import SophiaMLA
from ml.training.loss_stats import supervised_token_count
from ml.training.model_contracts import require_compute_loss
from ml.training.pretrain.engine.sm120_concurrent_ffn import install_sm120_runtime
from ml.training.pretrain.engine.step_execution import (
    SM120_GRAPH_BACKEND,
    StepExecutionPlan,
)
from ml.training.pretrain.engine.step_runner_impl import (
    AccumulatedUpdate,
    StepRunner,
)
from ml.training.pretrain.resources import (
    PretrainBatchValue,
    PretrainDataIter,
)
from ml.training.pretrain.release_config import RELEASE_QK_CLIP_THRESHOLD


_FORMAL_ACCUMULATION_STEPS = 320
_GRAPH_MICRO_STEPS = 32
_GRAPH_BLOCKS_PER_UPDATE = _FORMAL_ACCUMULATION_STEPS // _GRAPH_MICRO_STEPS
_BATCH_SIZE = 1
_SEQUENCE_LENGTH = 4096
_GRAPH_CAPTURE_WARMUP_REPLAYS = 2
_ALLOWED_BATCH_KEYS = frozenset({"input_ids", "labels", "labels_is_input_ids"})


class SM120GraphRunner(StepRunner):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        base_dtype: torch.dtype,
        token_weighted: bool,
        execution_plan: StepExecutionPlan,
        optimizer: torch.optim.Optimizer | None,
    ) -> None:
        if execution_plan.backend != SM120_GRAPH_BACKEND:
            raise ValueError(f"SM120GraphRunner received {execution_plan.backend!r}")
        if base_dtype != torch.bfloat16 or not token_weighted:
            raise RuntimeError("SM120 graph requires BF16 token-weighted training")
        if bool(getattr(model, "gradient_checkpointing", False)):
            raise RuntimeError("SM120 graph requires gradient checkpointing disabled")
        require_compute_loss(model, context=type(self).__name__)
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError("SM120 graph requires CUDA model weights")
        torch.cuda.set_device(device)
        self.runtime_evidence = install_sm120_runtime(model)
        self._model = model
        self._device = device
        self._amp_ctx = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        )

        import torch._dynamo as torch_dynamo
        import torch._inductor as torch_inductor

        torch_dynamo.config.recompile_limit = max(
            int(torch_dynamo.config.recompile_limit), 64
        )
        compile_options = dict(
            torch_inductor.list_mode_options("max-autotune-no-cudagraphs")
        )
        compile_options["triton.cudagraphs"] = False
        self._compiled_model = torch.compile(model, options=compile_options)
        self._trainable_params = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        self._main_grads = tuple(
            torch.zeros_like(parameter, dtype=torch.float32, device=device)
            for parameter in self._trainable_params
        )
        self._static_tokens = torch.empty(
            (_GRAPH_MICRO_STEPS, _BATCH_SIZE, _SEQUENCE_LENGTH),
            device=device,
            dtype=torch.int32,
        )
        self._static_batches = tuple(
            {
                "input_ids": self._static_tokens[index],
                "labels": self._static_tokens[index],
            }
            for index in range(_GRAPH_MICRO_STEPS)
        )
        self._graph: torch.cuda.CUDAGraph | None = None
        self._has_captured_graph = False
        self._graph_ready = False
        self._optimizer = optimizer
        self._optimizer_state_offloaded = False
        mla_layers = tuple(
            (int(module.layer_idx), module.attn)
            for module in model.modules()
            if isinstance(getattr(module, "attn", None), SophiaMLA)
        )
        self._mla_layer_indices = tuple(index for index, _module in mla_layers)
        self._mla_modules = tuple(module for _index, module in mla_layers)
        if len(self._mla_modules) != 7:
            raise RuntimeError("SM120 MLA telemetry requires seven MLA layers")
        if any(module.num_heads != 16 for module in self._mla_modules):
            raise RuntimeError("SM120 MLA telemetry requires sixteen heads")
        self._attention_logit_max = torch.full(
            (len(self._mla_modules), 16),
            float("-inf"),
            device=self._device,
            dtype=torch.float32,
        )
        for module, layer_buffer in zip(
            self._mla_modules,
            self._attention_logit_max,
            strict=True,
        ):
            module.enable_attention_logit_telemetry(layer_buffer)
        self._static_results: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()
        self.execution_plan = execution_plan
        self.runtime_evidence.update(
            {
                "backend": SM120_GRAPH_BACKEND,
                "graph_micro_steps": _GRAPH_MICRO_STEPS,
                "graph_blocks_per_update": _GRAPH_BLOCKS_PER_UPDATE,
                "mla_logit_telemetry": True,
                "qk_clip_enabled": True,
            }
        )

    @property
    def zero_grad_set_to_none(self) -> bool:
        return False

    def begin_update(self, *, accum_steps: int) -> None:
        if int(accum_steps) != _FORMAL_ACCUMULATION_STEPS:
            raise RuntimeError(
                f"SM120 graph requires accumulation_steps={_FORMAL_ACCUMULATION_STEPS}"
            )
        self._attention_logit_max.fill_(float("-inf"))

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        if self._optimizer is not None and self._optimizer is not optimizer:
            raise RuntimeError("SM120 graph optimizer changed after runner construction")
        self._optimizer = optimizer
        initialize_state = getattr(optimizer, "initialize_state", None)
        if callable(initialize_state):
            initialize_state()

    def gradient_pairs(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(zip(self._trainable_params, self._main_grads, strict=True))

    def prepare_for_evaluation(self) -> None:
        # Evaluation releases the graph, so keep the large optimizer state off
        # device while the eager validation pass is running and while the next
        # graph is captured.
        self._offload_optimizer_state_for_capture()
        graph = self._graph
        torch.cuda.synchronize(self._device)
        self._graph_ready = False
        self._static_results = ()
        for parameter in self._model.parameters():
            parameter.grad = None
        self._graph = None
        if graph is not None:
            graph.reset()
            del graph
        torch.cuda.empty_cache()

    def prepare_for_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if not self._optimizer_state_offloaded:
            return
        if self._optimizer is not optimizer:
            raise RuntimeError("SM120 graph optimizer changed during resume capture")
        self._move_optimizer_state(torch.device("cuda", self._device.index or 0))
        self._optimizer_state_offloaded = False

    def post_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        apply_clip = getattr(optimizer, "apply_mla_qk_clip", None)
        if not callable(apply_clip):
            raise RuntimeError("QK-Clip requires the Muon hybrid optimizer")
        apply_clip(
            self._mla_modules,
            self._attention_logit_max,
            threshold=RELEASE_QK_CLIP_THRESHOLD,
        )

    def attention_logit_metrics(
        self,
    ) -> tuple[tuple[int, ...], torch.Tensor]:
        return self._mla_layer_indices, self._attention_logit_max

    def run_micro(
        self,
        batch: Mapping[str, PretrainBatchValue],
        *,
        accum_steps: int,
    ) -> torch.Tensor:
        del batch, accum_steps
        raise RuntimeError("SM120 graph executes complete updates")

    def run_update(
        self,
        *,
        data_iter: PretrainDataIter,
        accumulation_steps: int,
        device: torch.device,
        token_weighted_cfg: bool,
    ) -> AccumulatedUpdate:
        self.begin_update(accum_steps=accumulation_steps)
        if torch.device(device) != self._device or not token_weighted_cfg:
            raise RuntimeError("SM120 graph training contract changed")
        update_loss_sum = torch.zeros((), device=self._device, dtype=torch.float32)
        update_supervised_tokens = torch.zeros(
            (), device=self._device, dtype=torch.int64
        )
        for _ in range(_GRAPH_BLOCKS_PER_UPDATE):
            self._fill_slab(data_iter)
            if self._graph is None or not self._graph_ready:
                self._capture_graph()
            else:
                self._graph.replay()
            block_loss_sum = torch.stack(
                [result[0] for result in self._static_results]
            ).sum()
            block_supervised_tokens = torch.stack(
                [result[1] for result in self._static_results]
            ).sum()
            update_loss_sum.add_(block_loss_sum)
            update_supervised_tokens.add_(block_supervised_tokens)
        return AccumulatedUpdate(
            token_weighted_active=True,
            micro_loss_sum=torch.zeros(
                (), device=self._device, dtype=torch.float32
            ),
            update_loss_sum=update_loss_sum,
            update_supervised_tokens=update_supervised_tokens,
        )

    def _fill_slab(self, data_iter: PretrainDataIter) -> None:
        for index in range(_GRAPH_MICRO_STEPS):
            batch = next(data_iter)
            if not frozenset(batch).issubset(_ALLOWED_BATCH_KEYS):
                raise RuntimeError("SM120 graph received unsupported batch fields")
            input_ids = batch.get("input_ids")
            labels = batch.get("labels")
            if not torch.is_tensor(input_ids) or labels is not input_ids:
                raise RuntimeError("SM120 graph requires labels to alias input_ids")
            if (
                input_ids.device != self._device
                or input_ids.dtype != torch.int32
                or tuple(input_ids.shape) != (_BATCH_SIZE, _SEQUENCE_LENGTH)
            ):
                raise RuntimeError("SM120 graph requires CUDA int32 B1/T4096 batches")
            self._static_tokens[index].copy_(input_ids)

    def _run_static_micro(
        self,
        batch: Mapping[str, PretrainBatchValue],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_inputs = dict(batch)
        model_inputs["compute_loss"] = True
        with self._amp_ctx:
            output = self._compiled_model(**model_inputs)
            loss = output.loss
        labels = batch["labels"]
        assert torch.is_tensor(labels)
        supervised = supervised_token_count(labels, ignore_index=-100)
        loss_sum = loss * supervised.to(dtype=loss.dtype)
        loss_sum.backward()
        self._accumulate_main_gradients()
        return loss_sum.detach(), supervised.detach()

    def _gradients(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            parameter.grad
            for parameter in self._model.parameters()
            if parameter.grad is not None
        )

    def _accumulate_main_gradients(self) -> None:
        active_pairs = [
            (parameter.grad, main_grad)
            for parameter, main_grad in zip(
                self._trainable_params, self._main_grads, strict=True
            )
            if parameter.grad is not None
        ]
        if len(active_pairs) != len(self._trainable_params):
            raise RuntimeError("SM120 graph did not produce every trainable gradient")
        model_grads = [gradient for gradient, _main_grad in active_pairs]
        main_grads = [main_grad for _gradient, main_grad in active_pairs]
        foreach_add = getattr(torch, "_foreach_add_", None)
        foreach_zero = getattr(torch, "_foreach_zero_", None)
        if not callable(foreach_add) or not callable(foreach_zero):
            raise RuntimeError(
                "SM120 graph requires torch foreach gradient accumulation primitives"
            )
        try:
            foreach_add(main_grads, model_grads)
            foreach_zero(list(model_grads))
        except RuntimeError as exc:
            raise RuntimeError("SM120 FP32 gradient accumulation failed") from exc

    @staticmethod
    def _zero_gradients(gradients: tuple[torch.Tensor, ...]) -> None:
        if gradients:
            torch._foreach_zero_(list(gradients))

    def _zero_all_gradients(self) -> None:
        self._zero_gradients(self._gradients())
        self._zero_gradients(self._main_grads)

    def _capture_graph(self) -> None:
        recapturing = bool(self._has_captured_graph)
        self._offload_optimizer_state_for_capture()
        if recapturing:
            torch.cuda.reset_peak_memory_stats(self._device)
        output = self._model.get_output_embeddings()
        output._sophia_release_cuda_cache_before_linear_ce = True
        self._run_static_micro(self._static_batches[0])
        torch.cuda.synchronize(self._device)
        self._zero_all_gradients()
        warmup_stream = torch.cuda.Stream(device=self._device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(_GRAPH_CAPTURE_WARMUP_REPLAYS):
                self._run_static_micro(self._static_batches[0])
                self._zero_all_gradients()
        torch.cuda.current_stream(self._device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self._device)
        self._zero_all_gradients()
        torch.cuda.empty_cache()
        graph = torch.cuda.CUDAGraph()
        captured_results = []
        with torch.cuda.graph(graph, capture_error_mode="global"):
            for batch in self._static_batches:
                captured_results.append(self._run_static_micro(batch))
        torch.cuda.synchronize(self._device)
        self._zero_all_gradients()
        self._graph = graph
        self._graph_ready = True
        self._has_captured_graph = True
        self._static_results = tuple(captured_results)
        graph.replay()
        self.runtime_evidence.update(
            {
                "graph_count": 1,
                "capture_allocated_bytes": int(
                    torch.cuda.memory_allocated(self._device)
                ),
                "capture_reserved_bytes": int(
                    torch.cuda.memory_reserved(self._device)
                ),
            }
        )
        if recapturing:
            self.runtime_evidence.update(
                {
                    "recapture_peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(self._device)
                    ),
                    "recapture_peak_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(self._device)
                    ),
                }
            )

    def _offload_optimizer_state_for_capture(self) -> None:
        optimizer = self._optimizer
        if optimizer is None or self._optimizer_state_offloaded:
            return
        self._move_optimizer_state(torch.device("cpu"))
        torch.cuda.synchronize(self._device)
        torch.cuda.empty_cache()
        self._optimizer_state_offloaded = True

    def _move_optimizer_state(self, device: torch.device) -> None:
        optimizer = self._optimizer
        if optimizer is None:
            return
        owners = [optimizer]
        for attr_name in ("_muon_opt", "_adamw_opt"):
            inner = getattr(optimizer, attr_name, None)
            if isinstance(inner, torch.optim.Optimizer):
                owners.append(inner)
        for owner in owners:
            for state in owner.state.values():
                for key, value in list(state.items()):
                    if torch.is_tensor(value) and value.device != device:
                        state[key] = value.to(device=device)


__all__ = ["SM120GraphRunner"]
