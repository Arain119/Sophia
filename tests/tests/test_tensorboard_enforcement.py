from __future__ import annotations

from builtins import ExceptionGroup

import torch
import pytest

from ml.core.engine.machine_signature import build_machine_adaptive_signature
from ml.core.engine.session_tensorboard import _SessionTensorBoardLogger
from ml.training.posttrain.runtime import setup_stage_runtime
from ml.training.posttrain.types import PosttrainLaunchConfig
from ml.training.pretrain.runtime_setup import setup_train_runtime_and_device
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.core.engine import session as session_mod


def test_pretrain_runtime_setup_requires_tensorboard(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_ensure_standard_stack(*, mode, require_tensorboard=False):
        seen["mode"] = mode
        seen["require_tensorboard"] = require_tensorboard
        return {"tensorboard": "ok"}

    monkeypatch.setattr(
        "ml.training.pretrain.runtime_setup.ensure_standard_stack",
        _fake_ensure_standard_stack,
    )
    monkeypatch.setattr(
        "ml.training.pretrain.runtime_setup.require_cuda",
        lambda _device: "cuda:0",
    )
    monkeypatch.setattr(
        "ml.training.pretrain.runtime_setup.setup_torch_backends",
        lambda: None,
    )
    monkeypatch.setattr(
        "ml.training.pretrain.runtime_setup.setup_seed",
        lambda _seed: None,
    )

    setup_train_runtime_and_device(
        PretrainRunConfig(data_path="dataset/pretrain_tokens", device="cuda:0"),
        run_kind="pretrain",
    )

    assert seen == {"mode": "train", "require_tensorboard": True}


def test_posttrain_runtime_setup_requires_tensorboard(monkeypatch, tmp_path) -> None:
    seen: dict[str, object] = {}

    def _fake_ensure_standard_stack(*, mode, require_tensorboard=False):
        seen["mode"] = mode
        seen["require_tensorboard"] = require_tensorboard
        return {"tensorboard": "ok"}

    monkeypatch.setattr(
        "ml.training.posttrain.runtime.ensure_standard_stack",
        _fake_ensure_standard_stack,
    )
    monkeypatch.setattr(
        "ml.training.posttrain.runtime.require_cuda",
        lambda _device: "cuda:0",
    )
    monkeypatch.setattr(
        "ml.training.posttrain.runtime.setup_torch_backends",
        lambda: None,
    )
    monkeypatch.setattr(
        "ml.training.posttrain.runtime.setup_seed",
        lambda _seed: None,
    )
    monkeypatch.setattr(
        "ml.training.posttrain.runtime.build_machine_adaptive_signature",
        lambda **kwargs: build_machine_adaptive_signature(
            schema=str(kwargs["schema"]),
            device=torch.device("cpu"),
        ),
    )

    ctx = setup_stage_runtime(
        output_dir=str(tmp_path),
        resume_path=None,
        launch=PosttrainLaunchConfig(
            output_dir=str(tmp_path),
            resume_from_checkpoint="",
            overwrite_output_dir=True,
            export_dir="export",
            requested_device="cuda:0",
            seed=42,
            gradient_checkpointing=True,
            max_steps=1,
        ),
    )

    assert seen == {"mode": "train", "require_tensorboard": True}
    assert isinstance(ctx.device, torch.device)


def test_engine_session_training_loop_logs_tensorboard(monkeypatch) -> None:
    seen: dict[str, object] = {"rows": [], "closed": 0}

    class _FakeTB:
        def __init__(self, *, output_dir: str) -> None:
            seen["output_dir"] = output_dir

        def log_metrics_row(self, row) -> None:
            seen["rows"].append(dict(row))

        def close(self) -> None:
            seen["closed"] += 1

    def _fake_append_metrics_row(_output_dir: str, _row: dict[str, object]) -> None:
        return None

    def _fake_run_training_loop_impl(
        *,
        output_dir: str,
        start_step: int,
        max_steps: int,
        step_fn,
        finalize_fn,
        append_metrics_row_fn,
        save_checkpoint_fn=None,
        time_fn=None,
    ) -> None:
        del output_dir, start_step, max_steps, step_fn, save_checkpoint_fn, time_fn
        append_metrics_row_fn(
            "out",
            {"stage": "sft", "step": 1, "loss": 0.25},
        )
        finalize_fn()

    monkeypatch.setattr(session_mod, "_SessionTensorBoardLogger", _FakeTB)
    monkeypatch.setattr(session_mod, "append_metrics_row", _fake_append_metrics_row)
    monkeypatch.setattr(session_mod, "_run_training_loop_impl", _fake_run_training_loop_impl)

    session_mod.run_training_loop(
        output_dir="out",
        start_step=0,
        max_steps=1,
        step_fn=lambda *_args: None,
        finalize_fn=lambda: None,
    )

    assert seen["output_dir"] == "out"
    assert seen["rows"] == [{"stage": "sft", "step": 1, "loss": 0.25}]
    assert seen["closed"] == 1


def test_session_tensorboard_close_surfaces_all_writer_failures() -> None:
    class _BrokenWriter:
        def flush(self) -> None:
            raise OSError("flush failed")

        def close(self) -> None:
            raise OSError("close failed")

    logger = _SessionTensorBoardLogger.__new__(_SessionTensorBoardLogger)
    logger._writer = _BrokenWriter()

    with pytest.raises(ExceptionGroup, match="TensorBoard logger close failed"):
        logger.close()

    assert logger._writer is None
