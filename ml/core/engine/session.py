"""Shared task-session helpers for engine-owned training loops."""

from __future__ import annotations

from collections.abc import Callable


from ml.core.engine.checkpointing import EngineCheckpoint
from ml.core.engine.metrics import append_metrics_row
from ml.core.engine.session_batches import maybe_prefetch_batch_iterator
from ml.core.engine.session_batches import scale_and_clip_grads
from ml.core.engine.session_batches import to_device_batch
from ml.core.engine.session_eval import (
    maybe_restore_rng_from_checkpoint as _maybe_restore_rng_from_checkpoint_impl,
)
from ml.core.engine.session_loop import run_training_loop as _run_training_loop_impl
from ml.core.engine.session_tensorboard import _SessionTensorBoardLogger
from ml.core.engine.session_runtime import generation_inference_mode
from ml.core.engine.session_runtime import reset_runtime_state
from ml.core.engine.types import StatePayload
from ml.core.common.rng import restore_rng_state


def run_training_loop(
    *,
    output_dir: str,
    start_step: int,
    max_steps: int,
    step_fn: Callable[[int, float, float], object],
    finalize_fn: Callable[[], None],
    save_checkpoint_fn: Callable[[int, StatePayload], None] | None = None,
) -> None:
    tb = _SessionTensorBoardLogger(output_dir=str(output_dir))

    def _append_metrics_row_and_tb(
        logged_output_dir: str,
        row: dict[str, object],
    ) -> None:
        append_metrics_row(str(logged_output_dir), dict(row))
        tb.log_metrics_row(dict(row))

    try:
        _run_training_loop_impl(
            output_dir=output_dir,
            start_step=start_step,
            max_steps=max_steps,
            step_fn=step_fn,
            finalize_fn=finalize_fn,
            append_metrics_row_fn=_append_metrics_row_and_tb,
            save_checkpoint_fn=save_checkpoint_fn,
        )
    finally:
        tb.close()


def maybe_restore_rng_from_checkpoint(resume_checkpoint: EngineCheckpoint | None) -> None:
    _maybe_restore_rng_from_checkpoint_impl(
        resume_checkpoint,
        restore_rng_state_fn=restore_rng_state,
    )


__all__ = [
    "generation_inference_mode",
    "maybe_prefetch_batch_iterator",
    "maybe_restore_rng_from_checkpoint",
    "reset_runtime_state",
    "run_training_loop",
    "scale_and_clip_grads",
    "to_device_batch",
]
