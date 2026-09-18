from __future__ import annotations

import pytest
import torch

from ml.training.pretrain.engine.step_execution import (
    SM120_GRAPH_BACKEND,
    StepExecutionPolicy,
    normalize_step_execution_backend,
    resolve_step_execution_plan,
)


def test_sm120_graph_is_the_only_formal_cuda_backend() -> None:
    policy = StepExecutionPolicy(backend=SM120_GRAPH_BACKEND)

    plan = policy.resolve_plan(device=torch.device("cuda"))

    assert plan.backend == SM120_GRAPH_BACKEND
    assert plan.reason == "configured_sm120_graph"


def test_old_inductor_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported step execution backend"):
        normalize_step_execution_backend("inductor")


def test_sm120_graph_rejects_cpu_execution() -> None:
    policy = StepExecutionPolicy(backend=SM120_GRAPH_BACKEND)

    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        policy.resolve_plan(device=torch.device("cpu"))


def test_machine_backend_must_match_configured_backend() -> None:
    with pytest.raises(RuntimeError, match="does not match configured backend"):
        resolve_step_execution_plan(
            policy=StepExecutionPolicy(backend=SM120_GRAPH_BACKEND),
            device=torch.device("cuda"),
            preferred_backend="eager",
        )
