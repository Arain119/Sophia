from __future__ import annotations

import warnings

import numpy as np
import torch


def np_dtype(dtype: str) -> np.dtype:
    if dtype == "int32":
        return np.dtype("<i4")
    raise ValueError(f"Unsupported token shard dtype: {dtype!r}. Expected 'int32'.")


def torch_from_numpy_readonly(arr: np.ndarray) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The given NumPy array is not writable"
        )
        return torch.from_numpy(arr)


def shape_changed_resume_incompatibility(
    *,
    shard_tokens: list[int] | tuple[int, ...],
    order: list[int],
    order_pos: int,
    current_pos: int | None,
    block_tokens: int,
) -> str | None:
    unread_order_pos = int(order_pos)
    if unread_order_pos < 0 or unread_order_pos > len(order):
        raise ValueError(
            f"token-stream resume state has an invalid order_pos cursor: {order_pos}"
        )
    if unread_order_pos >= len(order):
        return None
    block_tokens = int(block_tokens)
    if block_tokens <= 0:
        raise ValueError(
            f"shape-changed token-stream validation requires block_tokens > 0, got {block_tokens}"
        )

    for pos in range(unread_order_pos, len(order)):
        shard_idx = int(order[pos])
        if shard_idx < 0 or shard_idx >= len(shard_tokens):
            raise ValueError(
                "token-stream resume state references a shard index outside the loaded manifest: "
                f"shard_idx={shard_idx} len={len(shard_tokens)}"
            )
        n_tokens = int(shard_tokens[shard_idx])
        if n_tokens < block_tokens:
            unread_kind = "current" if pos == unread_order_pos else "future"
            return (
                "shape-changed token-stream resume would skip an unread shard in the active epoch "
                f"because the new block_tokens={block_tokens} exceed "
                f"{unread_kind}_shard_tokens={n_tokens}. "
                "Bridge the prior stage before switching shapes."
            )
        if pos != unread_order_pos or current_pos is None:
            continue
        max_start = int(n_tokens) - int(block_tokens)
        if int(current_pos) > int(max_start):
            return (
                "shape-changed token-stream resume would skip an unread shard tail because the "
                f"saved cursor={current_pos} exceeds the new max_start={max_start}. "
                "Bridge the prior stage before switching shapes."
            )
    return None


__all__ = [
    "np_dtype",
    "shape_changed_resume_incompatibility",
    "torch_from_numpy_readonly",
]
