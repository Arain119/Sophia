"""
Guard for the runtime sampler's top-k / nucleus (top-p) filtering.

The runtime generate() path samples through ``_apply_top_k_top_p``; these tests
pin that it filters the same way as the eval sampler so the two stay consistent.
"""

from __future__ import annotations

import torch

from ml.runtime.generation import sample_next_token
from ml.runtime.generation.sampler import apply_top_k_top_p


_LOGITS = torch.tensor([[10.0, 9.0, 1.0, 0.0, -5.0]])


def _kept(logits: torch.Tensor) -> int:
    return int((logits > float("-inf")).sum().item())


def test_top_p_one_is_a_noop() -> None:
    out = apply_top_k_top_p(_LOGITS.clone(), top_k=0, top_p=1.0)
    assert torch.equal(out, _LOGITS)


def test_top_p_small_keeps_only_top_token() -> None:
    out = apply_top_k_top_p(_LOGITS.clone(), top_k=0, top_p=0.01)
    assert _kept(out) == 1
    assert int(out.argmax(dim=-1).item()) == 0


def test_top_p_mid_keeps_nucleus() -> None:
    # softmax([10,9,1,...]) ~ [0.73, 0.27, ...]; top-2 already exceeds 0.9.
    out = apply_top_k_top_p(_LOGITS.clone(), top_k=0, top_p=0.9)
    assert _kept(out) == 2


def test_top_k_limits_survivors() -> None:
    out = apply_top_k_top_p(_LOGITS.clone(), top_k=2, top_p=1.0)
    assert _kept(out) == 2


def test_greedy_ignores_top_p() -> None:
    # temperature == 0 -> argmax, independent of top_p.
    token = sample_next_token(_LOGITS.clone(), temperature=0.0, top_k=0, top_p=0.5)
    assert int(token.item()) == 0


def test_top_p_restricts_sampling_support() -> None:
    # With a tiny top_p, sampling must always return the top token.
    torch.manual_seed(0)
    for _ in range(50):
        token = sample_next_token(_LOGITS.clone(), temperature=1.0, top_k=0, top_p=0.01)
        assert int(token.item()) == 0
