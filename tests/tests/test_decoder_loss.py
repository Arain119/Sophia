from __future__ import annotations

import pytest
import torch

from ml.modeling.decoder_loss import Z_LOSS_TOKEN_CHUNK_SIZE
from ml.modeling.decoder_loss import chunked_loss_stats_from_hidden
from ml.modeling.decoder_loss import z_loss_sum


class _IdentityMixer(torch.nn.Module):
    def forward(self, hidden, *, norm, output):
        del norm, output
        return hidden


class _IdentityModule(torch.nn.Module):
    def forward(self, x):
        return x


class _StandardMixer(torch.nn.Module):
    def forward(self, hidden, *, norm, output):
        return output(norm(hidden))


class _Cfg:
    def __init__(
        self,
        *,
        loss_chunk_size: int,
        z_loss_weight: float,
    ) -> None:
        self.loss_chunk_size = int(loss_chunk_size)
        self.z_loss_weight = float(z_loss_weight)
        self.return_logits_in_train = True
        self.vocab_size = 3


def _patch_fake_liger(monkeypatch) -> None:
    def fake_liger_linear_ce(
        input,
        weight,
        target,
        *,
        bias,
        ignore_index,
        reduction,
        lse_square_scale,
    ):
        assert ignore_index == -100
        assert reduction == "sum"
        logits = torch.nn.functional.linear(input, weight, bias)
        keep = target != ignore_index
        ce = torch.nn.functional.cross_entropy(
            logits,
            target,
            ignore_index=ignore_index,
            reduction="sum",
        )
        if float(lse_square_scale) <= 0.0:
            return ce
        return ce + (
            torch.logsumexp(logits.float(), dim=-1).square()
            * keep.to(dtype=torch.float32)
        ).sum() * float(lse_square_scale)

    import ml.modeling.decoder_loss as loss_mod

    monkeypatch.setattr(loss_mod, "_liger_fused_linear_ce_func", lambda: fake_liger_linear_ce)


def test_z_loss_sum_ignores_masked_targets() -> None:
    logits = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.5, -0.5, 1.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[1, -100]], dtype=torch.long)

    got = z_loss_sum(logits, labels)
    expected = torch.logsumexp(logits[:, :1, :].reshape(1, 3), dim=-1).square().sum()

    torch.testing.assert_close(got, expected)


def test_z_loss_sum_chunking_matches_full_reference() -> None:
    torch.manual_seed(0)
    logits = torch.randn((2, Z_LOSS_TOKEN_CHUNK_SIZE + 7, 11), dtype=torch.float32)
    labels = torch.randint(0, 11, (2, Z_LOSS_TOKEN_CHUNK_SIZE + 7), dtype=torch.long)
    labels[0, 3] = -100
    labels[1, 9] = -100

    got = z_loss_sum(logits, labels, token_chunk_size=13)
    flat_labels = labels.reshape(-1)
    flat_logits = logits.reshape(-1, logits.size(-1))
    expected = torch.logsumexp(
        flat_logits[flat_labels != -100].float(),
        dim=-1,
    ).square().sum()

    torch.testing.assert_close(got, expected)


def test_chunked_loss_stats_adds_z_loss(monkeypatch) -> None:
    _patch_fake_liger(monkeypatch)
    torch.manual_seed(0)
    hidden = torch.randn((1, 3, 5), dtype=torch.float32)
    labels = torch.tensor([[0, 1, 2]], dtype=torch.long)
    norm = _IdentityModule()
    output = torch.nn.Linear(5, 7, bias=False)

    base_sum, count = chunked_loss_stats_from_hidden(
        hidden,
        labels,
        label_offset=0,
        norm=norm,
        output=output,
        config=_Cfg(loss_chunk_size=0, z_loss_weight=0.0),
    )
    z_weight = 1e-4
    total_sum, total_count = chunked_loss_stats_from_hidden(
        hidden,
        labels,
        label_offset=0,
        norm=norm,
        output=output,
        config=_Cfg(loss_chunk_size=0, z_loss_weight=z_weight),
    )

    logits = output(hidden[:, :-1])
    expected_z = z_loss_sum(logits, labels[:, 1:]) * z_weight
    torch.testing.assert_close(total_count, count)
    torch.testing.assert_close(total_sum, base_sum + expected_z)


def test_chunked_loss_stats_keeps_zero_when_no_targets() -> None:
    hidden = torch.zeros((1, 1, 3), dtype=torch.float32)
    labels = torch.full((1, 1), -100, dtype=torch.long)
    norm = _IdentityModule()
    output = _IdentityModule()

    loss_sum, count = chunked_loss_stats_from_hidden(
        hidden,
        labels,
        label_offset=0,
        norm=norm,
        output=output,
        config=_Cfg(loss_chunk_size=0, z_loss_weight=1e-4),
    )

    assert float(loss_sum) == pytest.approx(0.0)
    assert float(count) == pytest.approx(0.0)


def test_liger_linear_ce_matches_reference_with_fake_provider(monkeypatch) -> None:
    _patch_fake_liger(monkeypatch)
    torch.manual_seed(0)
    hidden = torch.randn((2, 7, 5), dtype=torch.float32)
    labels = torch.randint(0, 11, (2, 7), dtype=torch.long)
    labels[0, 3] = -100
    norm = _IdentityModule()
    output = torch.nn.Linear(5, 11, bias=False)
    logits = output(hidden[:, :-1])
    flat_labels = labels[:, 1:].reshape(-1)
    torch_sum = torch.nn.functional.cross_entropy(
        logits.reshape(-1, int(logits.size(-1))),
        flat_labels,
        ignore_index=-100,
        reduction="sum",
    )
    torch_sum = torch_sum + z_loss_sum(logits, labels[:, 1:]) * 1e-4
    torch_count = (flat_labels != -100).sum().to(dtype=torch.float32)
    liger_sum, liger_count = chunked_loss_stats_from_hidden(
        hidden,
        labels,
        label_offset=0,
        norm=norm,
        output=output,
        config=_Cfg(loss_chunk_size=3, z_loss_weight=1e-4),
    )

    torch.testing.assert_close(liger_count, torch_count)
    torch.testing.assert_close(liger_sum, torch_sum, atol=1e-5, rtol=1e-5)


def test_liger_linear_ce_requires_provider(monkeypatch) -> None:
    import ml.modeling.decoder_loss as loss_mod

    def missing_liger(_module_name: str):
        raise ModuleNotFoundError(
            "No module named 'liger_kernel'",
            name="liger_kernel",
        )

    monkeypatch.setattr(loss_mod.importlib, "import_module", missing_liger)
    loss_mod._liger_fused_linear_ce_func.cache_clear()
    hidden = torch.randn((1, 3, 5), dtype=torch.float32)
    labels = torch.randint(0, 11, (1, 3), dtype=torch.long)

    with pytest.raises(RuntimeError, match="requires liger_kernel"):
        chunked_loss_stats_from_hidden(
            hidden,
            labels,
            label_offset=0,
            norm=_IdentityModule(),
            output=torch.nn.Linear(5, 11, bias=False),
            config=_Cfg(loss_chunk_size=2, z_loss_weight=0.0),
        )
    loss_mod._liger_fused_linear_ce_func.cache_clear()
