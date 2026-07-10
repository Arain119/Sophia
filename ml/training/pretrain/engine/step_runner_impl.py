from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from ml.training.pretrain.resources import PretrainBatch, PretrainBatchValue
from ml.training.loss_stats import supervised_token_count
from ml.training.model_contracts import require_compute_loss
from ml.training.pretrain.engine.step_execution import (
    StepExecutionPlan,
    resolve_torch_inductor_compile_options,
    resolve_torch_inductor_compile_mode,
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


class StepRunner:
    @property
    def zero_grad_set_to_none(self) -> bool:
        raise NotImplementedError

    def begin_update(self, *, accum_steps: int) -> None:
        del accum_steps

    def run_micro(
        self, batch: Mapping[str, PretrainBatchValue], *, accum_steps: int
    ) -> torch.Tensor | MicroStepResult:
        raise NotImplementedError


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


class CompiledStepRunner(EagerStepRunner):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        base_dtype: torch.dtype,
        token_weighted: bool = False,
        execution_plan: StepExecutionPlan,
    ) -> None:
        if str(execution_plan.backend) != "inductor":
            raise ValueError(
                "CompiledStepRunner requires an inductor execution plan, "
                f"got backend={execution_plan.backend!r}"
            )
        compile_mode = resolve_torch_inductor_compile_mode()
        compile_options = resolve_torch_inductor_compile_options()
        if compile_options:
            try:
                import torch._inductor as torch_inductor

                mode_options = dict(torch_inductor.list_mode_options(compile_mode))
            except Exception as exc:  # pragma: no cover - depends on torch internals
                raise RuntimeError(
                    "failed to resolve torch.compile mode options for "
                    f"compile_mode={compile_mode!r}"
                ) from exc
            mode_options.update(compile_options)
            compiled_model = torch.compile(model, options=mode_options)
        else:
            compiled_model = torch.compile(model, mode=compile_mode)
        super().__init__(
            model=compiled_model,
            base_dtype=base_dtype,
            token_weighted=token_weighted,
        )
        self.execution_plan = execution_plan


def build_step_runner(
    *,
    model: torch.nn.Module,
    base_dtype: torch.dtype,
    execution_plan: StepExecutionPlan,
    token_weighted: bool = False,
) -> StepRunner:
    if str(execution_plan.backend) == "inductor":
        return CompiledStepRunner(
            model=model,
            base_dtype=base_dtype,
            token_weighted=token_weighted,
            execution_plan=execution_plan,
        )
    return EagerStepRunner(
        model=model,
        base_dtype=base_dtype,
        token_weighted=token_weighted,
    )


__all__ = [
    "CompiledStepRunner",
    "EagerStepRunner",
    "MicroStepResult",
    "StepRunner",
    "build_step_runner",
]
