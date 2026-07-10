from __future__ import annotations

import os
from dataclasses import dataclass

import torch


def normalize_step_execution_backend(
    value: object,
    *,
    default: str = "eager",
) -> str:
    resolved = str(value or "").strip().lower()
    if not resolved:
        resolved = str(default or "").strip().lower()
    if resolved not in {"eager", "inductor"}:
        raise ValueError(
            "step execution backend must be 'eager' or 'inductor', "
            f"got {value!r}"
        )
    return resolved


@dataclass(frozen=True)
class StepExecutionPlan:
    backend: str = "eager"
    reason: str = "configured_eager"

    def __post_init__(self) -> None:
        backend = normalize_step_execution_backend(self.backend)
        object.__setattr__(self, "backend", backend)
        reason = str(self.reason or "").strip()
        if not reason:
            raise ValueError("step execution reason must be non-empty")
        object.__setattr__(self, "reason", reason)


TORCH_INDUCTOR_COMPILE_MODE_DEFAULT = "default"
TORCH_INDUCTOR_COMPILE_MODE_ENV = "SOPHIA_TORCH_INDUCTOR_COMPILE_MODE"
TORCH_INDUCTOR_DISABLE_CUDAGRAPHS_ENV = "SOPHIA_TORCH_INDUCTOR_DISABLE_CUDAGRAPHS"
SUPPORTED_TORCH_INDUCTOR_COMPILE_MODES = frozenset(
    {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }
)


def resolve_torch_inductor_compile_mode(value: object | None = None) -> str:
    raw = (
        os.environ.get(TORCH_INDUCTOR_COMPILE_MODE_ENV, "")
        if value is None
        else str(value or "")
    )
    mode = str(raw or "").strip().lower()
    if not mode:
        mode = TORCH_INDUCTOR_COMPILE_MODE_DEFAULT
    if mode not in SUPPORTED_TORCH_INDUCTOR_COMPILE_MODES:
        choices = ", ".join(sorted(SUPPORTED_TORCH_INDUCTOR_COMPILE_MODES))
        raise ValueError(
            f"unsupported torch inductor compile mode {mode!r}; expected one of: {choices}"
        )
    return mode


def resolve_torch_inductor_compile_options() -> dict[str, bool]:
    raw = os.environ.get(TORCH_INDUCTOR_DISABLE_CUDAGRAPHS_ENV, "")
    disable_cudagraphs = str(raw or "").strip().lower() in {"1", "true", "yes", "on"}
    if not disable_cudagraphs:
        return {}
    return {"triton.cudagraphs": False}


TORCH_INDUCTOR_COMPILE_MODE = TORCH_INDUCTOR_COMPILE_MODE_DEFAULT


@dataclass(frozen=True)
class StepExecutionPolicy:
    backend: str = "eager"

    def __post_init__(self) -> None:
        backend = normalize_step_execution_backend(self.backend)
        object.__setattr__(self, "backend", backend)

    def resolve_plan(
        self,
        *,
        device: torch.device,
        execution_target: str = "runtime",
    ) -> StepExecutionPlan:
        target_name = str(execution_target or "").strip().lower()
        if target_name != "runtime":
            raise ValueError(
                "step execution target must be 'runtime', "
                f"got {execution_target!r}"
            )
        if str(self.backend) == "eager":
            return StepExecutionPlan(
                backend="eager",
                reason="configured_eager",
            )
        if str(device.type) != "cuda":
            return StepExecutionPlan(
                backend="eager",
                reason="non_cuda_device",
            )
        return StepExecutionPlan(
            backend="inductor",
            reason="runtime_cuda",
        )


def resolve_step_execution_plan(
    *,
    policy: StepExecutionPolicy,
    device: torch.device,
    execution_target: str = "runtime",
    preferred_backend: str | None = None,
) -> StepExecutionPlan:
    canonical_plan = policy.resolve_plan(
        device=device,
        execution_target=execution_target,
    )
    if preferred_backend is None:
        return canonical_plan
    selected_backend = normalize_step_execution_backend(
        preferred_backend,
        default=str(canonical_plan.backend),
    )
    if selected_backend == "inductor" and str(canonical_plan.backend) == "inductor":
        return canonical_plan
    if selected_backend == "eager":
        if str(canonical_plan.backend) == "eager":
            return canonical_plan
        return StepExecutionPlan(
            backend="eager",
            reason="machine_selected_backend",
        )
    return canonical_plan


def step_execution_allows_machine_backend_selection(plan: StepExecutionPlan) -> bool:
    return str(plan.backend) == "inductor"


__all__ = [
    "TORCH_INDUCTOR_COMPILE_MODE",
    "TORCH_INDUCTOR_COMPILE_MODE_DEFAULT",
    "TORCH_INDUCTOR_COMPILE_MODE_ENV",
    "TORCH_INDUCTOR_DISABLE_CUDAGRAPHS_ENV",
    "StepExecutionPlan",
    "StepExecutionPolicy",
    "normalize_step_execution_backend",
    "resolve_torch_inductor_compile_options",
    "resolve_torch_inductor_compile_mode",
    "resolve_step_execution_plan",
    "step_execution_allows_machine_backend_selection",
]
