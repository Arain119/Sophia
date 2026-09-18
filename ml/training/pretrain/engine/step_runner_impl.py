from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from ml.training.pretrain.resources import (
    PretrainBatch,
    PretrainBatchValue,
    PretrainDataIter,
)
from ml.training.loss_stats import supervised_token_count
from ml.training.model_contracts import require_compute_loss
from ml.training.pretrain.engine.step_execution import (
    SM120_GRAPH_BACKEND,
    StepExecutionPlan,
)


def _build_precision_context(
    *,
    base_dtype: torch.dtype,
    device: torch.device,
):
    return (
        torch.autocast(
            device_type=str(device.type),
            dtype=base_dtype,
            enabled=(device.type == "cuda"),
        )
        if device.type in ("cuda", "cpu")
        else nullcontext()
    )


@dataclass(frozen=True)
class MicroStepResult:
    """
    Output of a single micro-step.

    - `loss_detached`: mean loss scalar (detached)
    - `loss_sum_detached`: sum loss scalar over supervised tokens (detached)
    - `supervised_tokens`: number of supervised tokens contributing to the loss (detached, int64 scalar)
    """

    loss_detached: torch.Tensor
    loss_sum_detached: torch.Tensor
    supervised_tokens: torch.Tensor


@dataclass(frozen=True)
class AccumulatedUpdate:
    token_weighted_active: bool
    micro_loss_sum: torch.Tensor
    update_loss_sum: torch.Tensor
    update_supervised_tokens: torch.Tensor


class StepRunner:
    @property
    def zero_grad_set_to_none(self) -> bool:
        raise NotImplementedError

    def begin_update(self, *, accum_steps: int) -> None:
        del accum_steps

    def gradient_pairs(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...] | None:
        return None

    def prepare_for_evaluation(self) -> None:
        return None

    def prepare_for_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def post_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def attention_logit_metrics(
        self,
    ) -> tuple[tuple[int, ...], torch.Tensor] | None:
        return None

    def run_micro(
        self, batch: Mapping[str, PretrainBatchValue], *, accum_steps: int
    ) -> torch.Tensor | MicroStepResult:
        raise NotImplementedError

    def run_update(
        self,
        *,
        data_iter: PretrainDataIter,
        accumulation_steps: int,
        device: torch.device,
        token_weighted_cfg: bool,
    ) -> AccumulatedUpdate:
        self.begin_update(accum_steps=accumulation_steps)
        micro_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        update_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        update_supervised_tokens = torch.zeros((), device=device, dtype=torch.int64)
        token_weighted_active = bool(token_weighted_cfg)
        for _ in range(int(accumulation_steps)):
            micro_out = self.run_micro(
                next(data_iter), accum_steps=int(accumulation_steps)
            )
            if isinstance(micro_out, MicroStepResult):
                token_weighted_active = True
                update_loss_sum.add_(
                    micro_out.loss_sum_detached.to(device=device, dtype=torch.float32)
                )
                update_supervised_tokens.add_(
                    micro_out.supervised_tokens.to(device=device, dtype=torch.int64)
                )
            elif token_weighted_active:
                raise RuntimeError(
                    "token-weighted training requires token-weighted runner output"
                )
            else:
                micro_loss_sum.add_(micro_out.to(device=device, dtype=torch.float32))
        return AccumulatedUpdate(
            token_weighted_active=token_weighted_active,
            micro_loss_sum=micro_loss_sum,
            update_loss_sum=update_loss_sum,
            update_supervised_tokens=update_supervised_tokens,
        )


def _run_micro_impl(
    *,
    model: torch.nn.Module,
    amp_ctx,
    token_weighted: bool,
    batch: Mapping[str, PretrainBatchValue],
    accum_steps: int,
) -> torch.Tensor | MicroStepResult:
    model_inputs: PretrainBatch = dict(batch)
    if "labels" in model_inputs:
        model_inputs["compute_loss"] = True
    mark_step = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    if callable(mark_step):
        mark_step()
    with amp_ctx:
        out = model(**model_inputs)
        loss = out.loss
    if not bool(token_weighted):
        loss_detached = loss.detach()
        loss = loss / float(max(int(accum_steps), 1))
        loss.backward()
        return loss_detached

    labels = batch.get("labels")
    supervised = supervised_token_count(
        labels if torch.is_tensor(labels) else None, ignore_index=-100
    )
    if supervised.device != loss.device:
        supervised = supervised.to(device=loss.device)

    loss_sum = loss * supervised.to(dtype=loss.dtype)
    loss_sum.backward()
    return MicroStepResult(
        loss_detached=loss.detach(),
        loss_sum_detached=loss_sum.detach(),
        supervised_tokens=supervised.detach(),
    )


class EagerStepRunner(StepRunner):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        base_dtype: torch.dtype,
        token_weighted: bool = False,
    ) -> None:
        self._model = model
        self._token_weighted = bool(token_weighted)
        require_compute_loss(model, context=type(self).__name__)
        try:
            dev = next(model.parameters()).device
        except StopIteration:  # pragma: no cover - models always have params
            dev = torch.device("cuda")
        self._amp_ctx = _build_precision_context(
            base_dtype=base_dtype,
            device=dev,
        )

    @property
    def zero_grad_set_to_none(self) -> bool:
        return True

    def run_micro(
        self, batch: Mapping[str, PretrainBatchValue], *, accum_steps: int
    ) -> torch.Tensor | MicroStepResult:
        return _run_micro_impl(
            model=self._model,
            amp_ctx=self._amp_ctx,
            token_weighted=self._token_weighted,
            batch=batch,
            accum_steps=accum_steps,
        )


def build_step_runner(
    *,
    model: torch.nn.Module,
    base_dtype: torch.dtype,
    execution_plan: StepExecutionPlan,
    token_weighted: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
) -> StepRunner:
    if execution_plan.backend == SM120_GRAPH_BACKEND:
        from ml.training.pretrain.engine.sm120_graph_runner import SM120GraphRunner

        return SM120GraphRunner(
            model=model,
            base_dtype=base_dtype,
            token_weighted=token_weighted,
            execution_plan=execution_plan,
            optimizer=optimizer,
        )
    return EagerStepRunner(
        model=model,
        base_dtype=base_dtype,
        token_weighted=token_weighted,
    )


__all__ = [
    "AccumulatedUpdate",
    "EagerStepRunner",
    "MicroStepResult",
    "StepRunner",
    "build_step_runner",
]
