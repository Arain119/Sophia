"""Shared logit-filtering helpers for the Sophia inference samplers."""

from __future__ import annotations

import torch


def apply_top_k_top_p(
    logits: torch.Tensor,
    *,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Top-k + nucleus (top-p) logit filtering on the last dim.

    Shared by the runtime generate() sampler and the eval sampler so both
    filter identically. ``top_k <= 0`` disables top-k; ``top_p >= 1.0`` disables
    nucleus filtering. At least the single highest-probability token is always kept.
    """
    if int(top_k) > 0:
        k = min(int(top_k), int(logits.size(-1)))
        if k > 0:
            topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(dim=-1, index=topk_idx, src=topk_vals)
            logits = filtered

    p = float(top_p)
    if 0.0 < p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        sorted_mask = cum > p
        # Shift so the first token over the threshold is always kept.
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        unsorted = torch.full_like(logits, float("-inf"))
        unsorted.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
        logits = unsorted

    return logits
