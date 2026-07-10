"""Evaluation and resume helpers shared by engine-owned tasks."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch

from ml.core.engine.checkpointing import EngineCheckpoint
from ml.core.common.device_batch import DeviceBatch
from ml.training.pretrain.engine.eval import evaluate_loss


def maybe_restore_rng_from_checkpoint(
    resume_checkpoint: EngineCheckpoint | None,
    *,
    restore_rng_state_fn: Callable[[object], None],
) -> None:
    if resume_checkpoint is None or resume_checkpoint.rng is None:
        return
    restore_rng_state_fn(resume_checkpoint.rng)


def supervised_eval(
    *,
    model: torch.nn.Module,
    data_iter: Iterator[DeviceBatch],
    steps: int,
    base_dtype: torch.dtype,
    to_device_batch_fn: Callable[..., DeviceBatch],
    evaluate_loss_fn: Callable[..., float] = evaluate_loss,
) -> float:
    try:
        device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover - trainable models always have params
        device = torch.device("cuda")

    class _DeviceEvalIterator:
        def __init__(self, source: Iterator[DeviceBatch]) -> None:
            self._source = source

        def __iter__(self) -> _DeviceEvalIterator:
            return self

        def __next__(self) -> DeviceBatch:
            batch = next(self._source)
            return to_device_batch_fn(batch, device=device)

    return evaluate_loss_fn(
        model=model,
        data_iter=_DeviceEvalIterator(data_iter),
        steps=int(steps),
        base_dtype=base_dtype,
    )


__all__ = [
    "maybe_restore_rng_from_checkpoint",
    "supervised_eval",
]
