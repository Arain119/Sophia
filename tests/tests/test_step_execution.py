from __future__ import annotations

import pytest
import torch

from ml.training.pretrain.engine.step_execution import (
    TORCH_INDUCTOR_COMPILE_MODE_DEFAULT,
    TORCH_INDUCTOR_COMPILE_MODE_ENV,
    TORCH_INDUCTOR_DISABLE_CUDAGRAPHS_ENV,
    StepExecutionPlan,
    resolve_torch_inductor_compile_options,
    resolve_torch_inductor_compile_mode,
)
from ml.training.pretrain.engine.step_runner_impl import CompiledStepRunner


def test_resolve_torch_inductor_compile_mode_defaults_to_low_overhead() -> None:
    assert resolve_torch_inductor_compile_mode("") == TORCH_INDUCTOR_COMPILE_MODE_DEFAULT


def test_resolve_torch_inductor_compile_mode_reads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TORCH_INDUCTOR_COMPILE_MODE_ENV, "reduce-overhead")

    assert resolve_torch_inductor_compile_mode() == "reduce-overhead"


def test_resolve_torch_inductor_compile_mode_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported torch inductor compile mode"):
        resolve_torch_inductor_compile_mode("not-a-mode")


def test_resolve_torch_inductor_compile_options_disable_cudagraphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TORCH_INDUCTOR_DISABLE_CUDAGRAPHS_ENV, "1")

    assert resolve_torch_inductor_compile_options() == {"triton.cudagraphs": False}


def test_resolve_torch_inductor_compile_options_defaults_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TORCH_INDUCTOR_DISABLE_CUDAGRAPHS_ENV, raising=False)

    assert resolve_torch_inductor_compile_options() == {}


def test_compiled_step_runner_exposes_inductor_option_resolution_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch._inductor as torch_inductor

    monkeypatch.setenv(TORCH_INDUCTOR_DISABLE_CUDAGRAPHS_ENV, "1")

    def fail_list_mode_options(_mode: str) -> dict[str, object]:
        raise RuntimeError("mode inspection failed")

    monkeypatch.setattr(torch_inductor, "list_mode_options", fail_list_mode_options)

    with pytest.raises(RuntimeError, match="failed to resolve torch.compile mode options"):
        CompiledStepRunner(
            model=torch.nn.Linear(1, 1),
            base_dtype=torch.float32,
            execution_plan=StepExecutionPlan(backend="inductor", reason="test"),
        )
