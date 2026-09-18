from __future__ import annotations

from dataclasses import dataclass

import torch


SM120_GRAPH_BACKEND = "sm120_graph"
SUPPORTED_STEP_EXECUTION_BACKENDS = frozenset({"eager", SM120_GRAPH_BACKEND})


def normalize_step_execution_backend(
    value: object,
    *,
    default: str = "eager",
) -> str:
    resolved = str(value or default).strip().lower()
    if resolved not in SUPPORTED_STEP_EXECUTION_BACKENDS:
        choices = ", ".join(sorted(SUPPORTED_STEP_EXECUTION_BACKENDS))
        raise ValueError(
            f"unsupported step execution backend {value!r}; expected one of: {choices}"
        )
    return resolved


@dataclass(frozen=True)
class StepExecutionPlan:
    backend: str = "eager"
    reason: str = "configured_eager"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backend", normalize_step_execution_backend(self.backend)
        )
        reason = str(self.reason).strip()
        if not reason:
            raise ValueError("step execution reason must be non-empty")
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True)
class StepExecutionPolicy:
    backend: str = "eager"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backend", normalize_step_execution_backend(self.backend)
        )

    def resolve_plan(
        self,
        *,
        device: torch.device,
        execution_target: str = "runtime",
    ) -> StepExecutionPlan:
        if str(execution_target).strip().lower() != "runtime":
            raise ValueError(
                f"step execution target must be 'runtime', got {execution_target!r}"
            )
        if self.backend == SM120_GRAPH_BACKEND:
            if device.type != "cuda":
                raise RuntimeError("sm120_graph requires a CUDA device")
            return StepExecutionPlan(
                backend=SM120_GRAPH_BACKEND,
                reason="configured_sm120_graph",
            )
        return StepExecutionPlan(backend="eager", reason="configured_eager")


def resolve_step_execution_plan(
    *,
    policy: StepExecutionPolicy,
    device: torch.device,
    execution_target: str = "runtime",
    preferred_backend: str | None = None,
) -> StepExecutionPlan:
    plan = policy.resolve_plan(device=device, execution_target=execution_target)
    if preferred_backend is None:
        return plan
    selected = normalize_step_execution_backend(
        preferred_backend, default=plan.backend
    )
    if selected != plan.backend:
        raise RuntimeError(
            f"machine backend {selected!r} does not match configured backend {plan.backend!r}"
        )
    return plan


__all__ = [
    "SM120_GRAPH_BACKEND",
    "SUPPORTED_STEP_EXECUTION_BACKENDS",
    "StepExecutionPlan",
    "StepExecutionPolicy",
    "normalize_step_execution_backend",
    "resolve_step_execution_plan",
]
