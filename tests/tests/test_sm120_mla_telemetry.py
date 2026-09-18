from __future__ import annotations

import importlib.util

import pytest
import torch

from ml.training.pretrain.engine.sm120_mla_telemetry import (
    mla_attention_logit_max_sm120,
)


def _sm120_triton_available() -> bool:
    return bool(
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (12, 0)
        and importlib.util.find_spec("triton") is not None
    )


@pytest.mark.skipif(
    not _sm120_triton_available(),
    reason="requires the formal SM120 Triton runtime",
)
def test_sm120_mla_logit_max_matches_causal_reference_cuda() -> None:
    torch.manual_seed(42)
    query = torch.randn(
        (1, 2, 257, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn_like(query)
    actual = torch.full((2,), float("-inf"), device="cuda", dtype=torch.float32)

    mla_attention_logit_max_sm120(query, key, actual)
    logits = torch.matmul(query, key.transpose(-2, -1)) * (128**-0.5)
    causal = torch.ones((257, 257), device="cuda", dtype=torch.bool).tril_()
    logits.masked_fill_(~causal.view(1, 1, 257, 257), float("-inf"))
    expected = logits.amax(dim=(0, 2, 3)).float()

    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.05)
