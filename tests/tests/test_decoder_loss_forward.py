from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from ml.modeling import decoder_loss_forward as mod


class _RuntimeModel:
    def __init__(self) -> None:
        self.calls = 0
        self.norm = nn.Identity()
        self.output = nn.Linear(4, 8, bias=False)

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert start_pos == 0
        self.calls += 1
        hidden = torch.nn.functional.one_hot(
            input_ids.long() % 4, num_classes=4
        ).float()
        return hidden, hidden.new_empty(0)


def test_right_padded_loss_uses_one_batched_model_forward(monkeypatch) -> None:
    runtime_model = _RuntimeModel()
    seen_shapes: list[tuple[int, ...]] = []

    def _fake_loss_stats(hidden, labels, **_):
        seen_shapes.append(tuple(hidden.shape))
        keep = labels[:, 1:] != -100
        return hidden[:, :-1].square().sum(), keep.sum().to(dtype=hidden.dtype)

    monkeypatch.setattr(mod, "chunked_loss_stats_from_hidden", _fake_loss_stats)
    input_ids = torch.tensor([[1, 2, 3, 0], [2, 3, 0, 0]])
    labels = torch.tensor([[1, 2, 3, -100], [2, 3, -100, -100]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

    loss = mod.forward_loss(
        runtime_model=runtime_model,
        config=SimpleNamespace(),
        training=True,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        output_weight=runtime_model.output.weight,
    )

    assert torch.isfinite(loss)
    assert runtime_model.calls == 1
    assert seen_shapes == [(2, 4, 4)]


def test_loss_rejects_non_right_padding_without_fallback() -> None:
    runtime_model = _RuntimeModel()
    input_ids = torch.tensor([[0, 1, 2]])
    labels = torch.tensor([[-100, 1, 2]])
    attention_mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)

    try:
        mod.forward_loss(
            runtime_model=runtime_model,
            config=SimpleNamespace(),
            training=True,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_weight=runtime_model.output.weight,
        )
    except ValueError as exc:
        assert "right-padded" in str(exc)
    else:
        raise AssertionError("non-right padding must fail explicitly")

    assert runtime_model.calls == 0
