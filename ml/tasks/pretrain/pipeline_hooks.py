from __future__ import annotations

import gc

import torch

from ml.training.pretrain.run_config import PretrainRunConfig
from ml.tasks.pretrain.run_spec_builder import build_pretrain_run_spec
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.training.pretrain.engine.step_execution import resolve_step_execution_plan
from ml.training.pretrain.engine.step_runner_impl import (
    StepRunner,
    build_step_runner,
)


def clear_cuda() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def preferred_runtime_backend(
    runtime_state: PretrainRuntimeState | None,
) -> str | None:
    if runtime_state is None:
        return None
    backend = str(runtime_state.step_execution_backend).strip()
    return None if not backend else backend


def build_pretrain_runner(
    *,
    args: PretrainRunConfig,
    runtime_state: PretrainRuntimeState | None,
    model: torch.nn.Module,
    base_dtype: torch.dtype,
    optimizer: torch.optim.Optimizer | None = None,
) -> StepRunner:
    try:
        device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover - models always have params
        device = torch.device("cuda")
    execution_plan = resolve_step_execution_plan(
        policy=build_pretrain_run_spec(args).step_execution,
        device=device,
        execution_target="runtime",
        preferred_backend=preferred_runtime_backend(runtime_state),
    )
    runner = build_step_runner(
        model=model,
        base_dtype=base_dtype,
        execution_plan=execution_plan,
        token_weighted=True,
        optimizer=optimizer,
    )
    return runner


__all__ = [
    "build_pretrain_runner",
    "clear_cuda",
    "preferred_runtime_backend",
]
