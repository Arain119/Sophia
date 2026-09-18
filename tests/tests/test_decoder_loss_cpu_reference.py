"""The loss the model reports on a host without a Triton device."""

from __future__ import annotations

import torch
from torch import nn

from ml.modeling.decoder_loss import (
    _loss_chunk_stats,
    _reference_linear_ce_loss_chunk_stats,
)


def _pieces(vocab: int = 37, dim: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    norm = nn.LayerNorm(dim)
    output = nn.Linear(dim, vocab, bias=False)
    return norm, output


def test_reference_matches_cross_entropy_computed_directly() -> None:
    norm, output = _pieces()
    hidden = torch.randn(2, 5, 8)
    labels = torch.tensor([[3, -100, 7, 1, -100], [0, 9, -100, 2, 4]])

    loss_sum, count = _reference_linear_ce_loss_chunk_stats(
        hidden, labels, norm=norm, output=output
    )

    logits = output(norm(hidden)).reshape(-1, 37).float()
    expected = torch.nn.functional.cross_entropy(
        logits, labels.reshape(-1), ignore_index=-100, reduction="sum"
    )
    assert torch.allclose(loss_sum, expected, atol=1e-5)
    # ten label positions, three of them ignored
    assert count.item() == 7


def test_reference_counts_only_supervised_positions() -> None:
    norm, output = _pieces()
    hidden = torch.randn(1, 4, 8)
    all_ignored = torch.full((1, 4), -100)

    loss_sum, count = _reference_linear_ce_loss_chunk_stats(
        hidden, all_ignored, norm=norm, output=output
    )

    assert count.item() == 0
    assert loss_sum.item() == 0.0


def test_reference_is_independent_of_the_tile_size() -> None:
    """Tiling is a memory decision; it must not change the number."""
    norm, output = _pieces(seed=3)
    hidden = torch.randn(3, 9, 8)
    labels = torch.randint(0, 37, (3, 9))
    labels[0, 0] = -100

    whole, count_whole = _reference_linear_ce_loss_chunk_stats(
        hidden, labels, norm=norm, output=output, tile=4096
    )
    tiled, count_tiled = _reference_linear_ce_loss_chunk_stats(
        hidden, labels, norm=norm, output=output, tile=2
    )

    assert torch.allclose(whole, tiled, atol=1e-5)
    assert count_whole.item() == count_tiled.item()


def test_chunk_stats_take_the_reference_path_off_cuda() -> None:
    """The dispatch is what makes a CPU host able to score a checkpoint."""
    norm, output = _pieces(seed=5)
    hidden = torch.randn(2, 6, 8)
    labels = torch.randint(0, 37, (2, 6))

    dispatched = _loss_chunk_stats(hidden, labels, norm=norm, output=output)
    direct = _reference_linear_ce_loss_chunk_stats(
        hidden, labels, norm=norm, output=output
    )

    assert torch.allclose(dispatched[0], direct[0], atol=1e-6)
    assert dispatched[1].item() == direct[1].item()


def test_reference_carries_a_gradient() -> None:
    """Scoring is read-only today, but a CPU smoke test should still train."""
    norm, output = _pieces(seed=7)
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    labels = torch.tensor([[1, 2, -100, 3]])

    loss_sum, _count = _reference_linear_ce_loss_chunk_stats(
        hidden, labels, norm=norm, output=output
    )
    loss_sum.backward()

    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
