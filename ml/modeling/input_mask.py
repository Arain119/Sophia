from __future__ import annotations

import torch


def _as_bool_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask if attention_mask.dtype == torch.bool else (attention_mask != 0)


def is_all_ones_mask(attention_mask: torch.Tensor) -> bool:
    mask = _as_bool_mask(attention_mask)
    return bool(mask.all().item())


def slice_valid_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[tuple[int, int, torch.Tensor]]:
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must be [B,T]")
    mask = _as_bool_mask(attention_mask)
    if mask.shape != input_ids.shape:
        raise ValueError("attention_mask shape must match input_ids")

    rows: list[tuple[int, int, torch.Tensor]] = []
    for batch_index in range(int(input_ids.size(0))):
        idx = mask[batch_index].nonzero(as_tuple=False).squeeze(-1)
        if int(idx.numel()) == 0:
            raise ValueError("attention_mask row has no valid tokens")
        start = int(idx[0].item())
        end = int(idx[-1].item()) + 1
        if int(idx.numel()) != (end - start):
            raise ValueError("Sophia only supports contiguous padding masks")
        rows.append((start, end, input_ids[batch_index : batch_index + 1, start:end]))
    return rows


__all__ = ["is_all_ones_mask", "slice_valid_tokens"]
