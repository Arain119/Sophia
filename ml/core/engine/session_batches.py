"""Batch/device helpers shared by engine-owned tasks."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ml.core.common.device_batch import move_batch_to_device
from ml.training.pretrain.engine.grad_clip import (
    PINNED_AGC_CLIP,
    PINNED_AGC_EPS,
    PINNED_AGC_EXCLUDE_BIAS_AND_NORM,
    PINNED_GRAD_CLIP_MODE,
    clip_gradients_,
    scale_grads_by_token_count_,
)
from ml.core.common.device_batch import (
    DeviceBatch,
    ResumableBatchIterator,
    ResumableCudaPrefetcher,
)


def to_device_batch(
    batch: DeviceBatch,
    *,
    device: torch.device,
    move_batch_to_device_fn: Callable[..., DeviceBatch] = move_batch_to_device,
) -> DeviceBatch:
    return move_batch_to_device_fn(batch, device=device)


def maybe_prefetch_batch_iterator(
    data_iter: ResumableBatchIterator,
    *,
    device: torch.device,
    move_batch_to_device_fn: Callable[..., DeviceBatch] = move_batch_to_device,
) -> ResumableBatchIterator:
    if torch.device(device).type != "cuda":
        return data_iter
    return ResumableCudaPrefetcher(
        data_iter,
        device=device,
        move_batch_to_device_fn=move_batch_to_device_fn,
    )


def scale_and_clip_grads(
    *,
    model: torch.nn.Module,
    supervised_tokens: torch.Tensor,
    max_grad_norm: float,
    scale_grads_by_token_count_fn: Callable[..., None] = scale_grads_by_token_count_,
    clip_gradients_fn: Callable[..., torch.Tensor | None] = clip_gradients_,
    pinned_grad_clip_mode: str = PINNED_GRAD_CLIP_MODE,
    pinned_agc_clip: float = PINNED_AGC_CLIP,
    pinned_agc_eps: float = PINNED_AGC_EPS,
    pinned_agc_exclude_bias_and_norm: int = PINNED_AGC_EXCLUDE_BIAS_AND_NORM,
) -> torch.Tensor | None:
    scale_grads_by_token_count_fn(model=model, supervised_tokens=supervised_tokens)
    return clip_gradients_fn(
        model=model,
        grad_clip_mode=pinned_grad_clip_mode,
        max_grad_norm=float(max_grad_norm),
        agc_clip=pinned_agc_clip,
        agc_eps=pinned_agc_eps,
        agc_exclude_bias_and_norm=pinned_agc_exclude_bias_and_norm,
    )


__all__ = [
    "maybe_prefetch_batch_iterator",
    "scale_and_clip_grads",
    "to_device_batch",
]
