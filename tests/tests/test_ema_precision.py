"""
Regression guard for EMA numerical precision.

The pretrain entrypoint enables EMA with a long half-life, so the per-update
factor ``(1 - decay)`` is ~1e-5 -- far below bf16's relative resolution. The EMA
shadow must therefore be kept in fp32, otherwise every update rounds away in bf16
and the shadow stays frozen at its initial value (which would then be exported as
the "best"/final weights).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ml.training.ema import ModelEMA


# Same regime as the release pretrain schedule: half-life -> decay close to 1.
_NEAR_ONE_DECAY = math.exp(math.log(0.5) / 20_000)  # ~0.99997, (1 - decay) ~ 3e-5


def _drift_and_average(model: nn.Module, ema: ModelEMA, *, steps: int = 2000) -> None:
    weight = model.weight
    with torch.no_grad():
        for _ in range(steps):
            weight.add_((0.01 * torch.randn(weight.shape)).to(weight.dtype))
            ema.update()


def test_shadow_is_fp32_for_bf16_model() -> None:
    model = nn.Linear(128, 128, bias=False).to(torch.bfloat16)
    ema = ModelEMA(model, decay=_NEAR_ONE_DECAY)
    assert all(s.dtype == torch.float32 for s in ema._shadow)


def test_ema_tracks_under_near_one_decay_in_bf16_model() -> None:
    model = nn.Linear(128, 128, bias=False).to(torch.bfloat16)
    init = model.weight.detach().float().clone()
    ema = ModelEMA(model, decay=_NEAR_ONE_DECAY)
    _drift_and_average(model, ema)
    moved = (ema._shadow[0].float() - init).abs().mean().item()
    # A bf16 shadow would stay ~1e-4 frozen; fp32 tracking moves orders more.
    assert moved > 1e-3, f"EMA shadow barely moved ({moved:.2e}) -- bf16 rounding regression"


def test_apply_to_model_preserves_param_dtype() -> None:
    model = nn.Linear(64, 64, bias=False).to(torch.bfloat16)
    ema = ModelEMA(model, decay=_NEAR_ONE_DECAY)
    ema.update()
    with ema.apply_to_model():
        assert model.weight.dtype == torch.bfloat16  # not silently promoted to fp32
    assert model.weight.dtype == torch.bfloat16


def test_state_dict_roundtrip_keeps_fp32_shadow() -> None:
    model = nn.Linear(64, 64, bias=False).to(torch.bfloat16)
    ema = ModelEMA(model, decay=_NEAR_ONE_DECAY)
    _drift_and_average(model, ema, steps=100)
    state = ema.state_dict()

    restored = ModelEMA(model, decay=_NEAR_ONE_DECAY)
    restored.load_state_dict(state)
    assert all(s.dtype == torch.float32 for s in restored._shadow)
    for a, b in zip(ema._shadow, restored._shadow, strict=True):
        assert torch.equal(a, b)
