"""Runtime-state helpers shared by engine-owned tasks."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator

import torch

from ml.runtime.controls import (
    maybe_refresh_runtime_state_buffers,
    maybe_reset_runtime_cache,
)


def reset_runtime_state(
    model: torch.nn.Module,
    *,
    refresh_runtime_state_buffers_fn: Callable[[torch.nn.Module], bool] = maybe_refresh_runtime_state_buffers,
    reset_runtime_cache_fn: Callable[[torch.nn.Module], None] = maybe_reset_runtime_cache,
) -> None:
    if refresh_runtime_state_buffers_fn(model):
        return
    reset_runtime_cache_fn(model)


@contextlib.contextmanager
def generation_inference_mode(
    model: torch.nn.Module,
    *,
    reset_runtime_state_fn: Callable[[torch.nn.Module], None],
) -> Iterator[None]:
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        reset_runtime_state_fn(model)
        model.train(was_training)


__all__ = [
    "generation_inference_mode",
    "reset_runtime_state",
]
