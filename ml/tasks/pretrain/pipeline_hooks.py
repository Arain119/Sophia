from __future__ import annotations

import gc

import torch

from ml.core.spec import build_release_pretrain_schedule_spec
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.tasks.pretrain.run_spec_builder import build_pretrain_run_spec
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.training.pretrain.engine.step_execution import (
    StepExecutionPlan,
    resolve_step_execution_plan,
)
from ml.training.pretrain.engine.step_runner_impl import (
    StepRunner,
    build_step_runner,
)

_BASE_SEQ_LEN = int(build_release_pretrain_schedule_spec().train_seq_len)


def clear_cuda() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def resolve_pretrain_seq_len(
    *,
    args: PretrainRunConfig,
    runtime_state: PretrainRuntimeState | None,
) -> int:
    seq_len = int(build_pretrain_run_spec(args).seq_len)
    if runtime_state is not None and int(runtime_state.seq_len) > 0:
        seq_len = int(runtime_state.seq_len)
    return int(seq_len)


def resolve_pretrain_step_execution_plan(
    *,
    args: PretrainRunConfig,
    device: torch.device,
    seq_len: int,
    preferred_backend: str | None = None,
) -> StepExecutionPlan:
    # Inductor is only validated at the base train seq. Longer curriculum stages
    # hit Inductor recompile/triton-shared-memory failures at seq>base, so run them
    # eager on the fixed long-stage BF16 path.
    effective_backend = preferred_backend
    if int(seq_len) != _BASE_SEQ_LEN:
        effective_backend = "eager"
    return resolve_step_execution_plan(
        policy=build_pretrain_run_spec(args).step_execution,
        device=device,
        execution_target="runtime",
        preferred_backend=effective_backend,
    )


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
) -> StepRunner:
    stage_seq_len = resolve_pretrain_seq_len(
        args=args,
        runtime_state=runtime_state,
    )
    try:
        device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover - models always have params
        device = torch.device("cuda")
    execution_plan = resolve_pretrain_step_execution_plan(
        args=args,
        device=device,
        seq_len=int(stage_seq_len),
        preferred_backend=preferred_runtime_backend(runtime_state),
    )
    return build_step_runner(
        model=model,
        base_dtype=base_dtype,
        execution_plan=execution_plan,
        token_weighted=True,
    )


__all__ = [
    "build_pretrain_runner",
    "clear_cuda",
    "preferred_runtime_backend",
    "resolve_pretrain_seq_len",
    "resolve_pretrain_step_execution_plan",
]
