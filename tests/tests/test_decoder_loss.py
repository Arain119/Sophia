from __future__ import annotations

import pytest
import torch

from ml.modeling.decoder_loss import chunked_loss_stats_from_hidden


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
    ) -> None:
        self.loss_chunk_size = int(loss_chunk_size)
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
    ):
        assert ignore_index == -100
        assert reduction == "sum"
        logits = torch.nn.functional.linear(input, weight, bias)
        return torch.nn.functional.cross_entropy(
            logits,
            target,
            ignore_index=ignore_index,
            reduction="sum",
        )

    import ml.modeling.decoder_loss as loss_mod

    monkeypatch.setattr(loss_mod, "_liger_fused_linear_ce_func", lambda: fake_liger_linear_ce)


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
        config=_Cfg(loss_chunk_size=0),
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
    torch_count = (flat_labels != -100).sum().to(dtype=torch.float32)
    liger_sum, liger_count = chunked_loss_stats_from_hidden(
        hidden,
        labels,
        label_offset=0,
        norm=norm,
        output=output,
        config=_Cfg(loss_chunk_size=3),
    )

    torch.testing.assert_close(liger_count, torch_count)
    torch.testing.assert_close(liger_sum, torch_sum, atol=1e-5, rtol=1e-5)


def test_linear_ce_releases_allocator_cache_once_when_armed(monkeypatch) -> None:
    _patch_fake_liger(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    hidden = torch.randn((1, 3, 5), dtype=torch.float32)
    labels = torch.randint(0, 11, (1, 3), dtype=torch.long)
    output = torch.nn.Linear(5, 11, bias=False)
    output._sophia_release_cuda_cache_before_linear_ce = True

    for _ in range(2):
        chunked_loss_stats_from_hidden(
            hidden,
            labels,
            label_offset=0,
            norm=_IdentityModule(),
            output=output,
            config=_Cfg(loss_chunk_size=0),
        )

    assert calls == ["empty"]
    assert output._sophia_release_cuda_cache_before_linear_ce is False


def test_graph_safe_linear_ce_matches_reference_gradients() -> None:
    import ml.modeling.decoder_loss as loss_mod

    torch.manual_seed(9)
    hidden = torch.randn((7, 5), requires_grad=True)
    weight = torch.randn((11, 5), requires_grad=True)
    labels = torch.tensor([1, 3, -100, 7, 2, 5, 0])
    bias = hidden.new_empty((0,))
    actual = loss_mod._GraphSafeLinearCE.apply(hidden, weight, labels, bias)
    actual.backward()
    actual_hidden_grad = hidden.grad.detach().clone()
    actual_weight_grad = weight.grad.detach().clone()

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(reference_hidden, reference_weight),
        labels,
        ignore_index=-100,
        reduction="sum",
    )
    reference.backward()

    torch.testing.assert_close(actual.detach(), reference.detach())
    torch.testing.assert_close(actual_hidden_grad, reference_hidden.grad)
    torch.testing.assert_close(actual_weight_grad, reference_weight.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="liger linear-ce path is CUDA-only")
def test_liger_linear_ce_requires_provider(monkeypatch) -> None:
    import ml.modeling.decoder_loss as loss_mod

    def missing_liger(_module_name: str):
        raise ModuleNotFoundError(
            "No module named 'liger_kernel'",
            name="liger_kernel",
        )

    monkeypatch.setattr(loss_mod.importlib, "import_module", missing_liger)
    loss_mod._liger_fused_linear_ce_func.cache_clear()
    hidden = torch.randn((1, 3, 5), dtype=torch.float32, device="cuda")
    labels = torch.randint(0, 11, (1, 3), dtype=torch.long, device="cuda")

    with pytest.raises(RuntimeError, match="requires liger_kernel"):
        chunked_loss_stats_from_hidden(
            hidden,
            labels,
            label_offset=0,
            norm=_IdentityModule(),
            output=torch.nn.Linear(5, 11, bias=False),
            config=_Cfg(loss_chunk_size=2),
        )
    loss_mod._liger_fused_linear_ce_func.cache_clear()
