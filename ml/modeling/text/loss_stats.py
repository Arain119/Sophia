from __future__ import annotations

import torch
import torch.nn.functional as functional


def supervised_token_count(
    labels: torch.Tensor | None,
    *,
    label_offset: int = 0,
    ignore_index: int = -100,
) -> torch.Tensor:
    if labels is None or not torch.is_tensor(labels):
        return torch.zeros((), dtype=torch.int64)
    if labels.ndim != 2:
        raise ValueError(
            f"labels must be 2D [B,S] to count supervised tokens (got shape={tuple(labels.shape)})"
        )
    target_start = int(label_offset) + 1
    if int(labels.size(1)) <= target_start:
        return torch.zeros((), device=labels.device, dtype=torch.int64)
    shifted = labels[:, target_start:]
    return (shifted != int(ignore_index)).sum(dtype=torch.int64)


def shifted_loss_sum_and_count(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_offset: int = 0,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(logits.size(1)) <= 1 or int(labels.size(1)) <= int(label_offset) + 1:
        zero = logits.new_zeros(())
        return zero, zero
    shift_logits = logits[:, :-1, :].contiguous()
    target_start = int(label_offset) + 1
    target_end = target_start + int(shift_logits.size(1))
    if int(labels.size(1)) < target_end:
        shift_logits = shift_logits[:, : max(int(labels.size(1)) - target_start, 0), :]
        target_end = target_start + int(shift_logits.size(1))
    if int(shift_logits.size(1)) <= 0:
        zero = logits.new_zeros(())
        return zero, zero
    shift_labels = labels[:, target_start:target_end].contiguous()
    flat_labels = shift_labels.reshape(-1).to(dtype=torch.long)
    flat_logits = shift_logits.reshape(-1, int(shift_logits.size(-1)))
    loss_sum = functional.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=int(ignore_index),
        reduction="sum",
    )
    count = (flat_labels != int(ignore_index)).sum().to(dtype=loss_sum.dtype)
    return loss_sum, count


def mean_loss_from_sum_and_count(
    *,
    loss_sum: torch.Tensor,
    count: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        count > 0,
        loss_sum / count.clamp_min(1.0),
        reference.new_zeros(()),
    )
