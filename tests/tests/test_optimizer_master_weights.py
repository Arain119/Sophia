"""
Regression guard for optimizer master-weight precision.

At the pinned release LR the per-step Muon/AdamW update has magnitude below half a
bf16 ULP at the typical weight scale. If the optimizer wrote updates straight back
into the bf16 model weights, every update would round to zero and the weights would
stay frozen at init -- the same failure mode the fp32 EMA shadow already guards
against (see ``test_ema_precision``). ``TorchMuonFusedAdamW`` therefore keeps fp32
master weights: updates accumulate in fp32 and the bf16 model is refreshed from the
masters after each step.
"""

from __future__ import annotations

import torch
import pytest
from torch import nn

from ml.training.pretrain.optimizer import create_torch_muon_optimizer


def _matrix_model(dim: int = 64) -> nn.Module:
    # Bias-free linear -> a single 2D weight, routed entirely to Muon (no AdamW),
    # which keeps the test free of the CUDA-only fused AdamW path.
    return nn.Linear(dim, dim, bias=False).to(torch.bfloat16)


def _opt(model: nn.Module, *, lr: float) -> object:
    return create_torch_muon_optimizer(
        model,
        lr=float(lr),
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        muon_target_rms=0.18,
    )


def _hybrid_model(vocab_size: int = 32, dim: int = 16) -> nn.Module:
    return nn.Sequential(
        nn.Embedding(vocab_size, dim),
        nn.LayerNorm(dim),
        nn.Linear(dim, dim),
    )


def test_master_weights_are_fp32_for_bf16_model() -> None:
    model = _matrix_model(32)
    opt = _opt(model, lr=1e-3)
    assert opt._master_pairs  # type: ignore[attr-defined]
    assert all(master.dtype == torch.float32 for _, master in opt._master_pairs)  # type: ignore[attr-defined]
    # The model itself stays bf16 for fast compute.
    assert model.weight.dtype == torch.bfloat16


def test_sub_ulp_updates_accumulate_via_fp32_masters() -> None:
    torch.manual_seed(0)
    model = _matrix_model(64)
    with torch.no_grad():
        model.weight.fill_(0.5)  # bf16 ULP here is ~2e-3
    init = model.weight.detach().float().clone()

    opt = _opt(model, lr=1e-3)  # per-step update RMS ~ target_rms*lr = 1.8e-4 << ULP
    grad = torch.randn(64, 64)  # fixed direction so updates stay coherent
    for _ in range(400):
        model.weight.grad = grad.to(torch.bfloat16).clone()
        opt.step()

    moved = (model.weight.detach().float() - init).abs().mean().item()
    # A bf16-direct optimizer would freeze near 0; fp32 accumulation moves it orders more.
    assert moved > 1e-2, f"sub-ULP updates not accumulating (moved={moved:.2e})"


def test_model_weight_tracks_master_after_step() -> None:
    torch.manual_seed(0)
    model = _matrix_model(32)
    opt = _opt(model, lr=1e-3)
    model.weight.grad = torch.randn(32, 32).to(torch.bfloat16)
    opt.step()
    for model_param, master in opt._master_pairs:  # type: ignore[attr-defined]
        assert torch.equal(model_param.data, master.data.to(model_param.dtype))


def test_state_dict_roundtrip_restores_masters() -> None:
    torch.manual_seed(0)
    model = _matrix_model(32)
    opt = _opt(model, lr=1e-3)
    grad = torch.randn(32, 32)
    for _ in range(50):
        model.weight.grad = grad.to(torch.bfloat16).clone()
        opt.step()
    state = opt.state_dict()

    restored_model = _matrix_model(32)
    restored_opt = _opt(restored_model, lr=1e-3)
    restored_opt.load_state_dict(state)

    for (_, master), (_, restored_master) in zip(
        opt._master_pairs, restored_opt._master_pairs, strict=True  # type: ignore[attr-defined]
    ):
        assert master.dtype == torch.float32
        assert torch.equal(master, restored_master)
    # The bf16 model is resynced from the restored masters.
    assert torch.equal(restored_model.weight.detach(), model.weight.detach())


def _state_tensors(payload: object):
    if torch.is_tensor(payload):
        yield payload
    elif isinstance(payload, dict):
        for value in payload.values():
            yield from _state_tensors(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            yield from _state_tensors(value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_state_dict_offloads_checkpoint_tensors_to_cpu() -> None:
    torch.manual_seed(0)
    model = _matrix_model(16).to("cuda")
    opt = _opt(model, lr=1e-3)
    model.weight.grad = torch.randn_like(model.weight)
    opt.step()

    state = opt.state_dict()

    tensors = list(_state_tensors(state))
    assert tensors
    assert all(tensor.device.type == "cpu" for tensor in tensors)
    assert all(master.device.type == "cpu" for master in state["masters"])


def test_state_dict_roundtrip_restores_masters_for_hybrid_groups() -> None:
    torch.manual_seed(0)
    model = _hybrid_model()
    opt = create_torch_muon_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    out = model(input_ids).float().sum()
    out.backward()
    opt.step()
    state = opt.state_dict()

    restored_model = _hybrid_model()
    restored_opt = create_torch_muon_optimizer(
        restored_model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    restored_opt.load_state_dict(state)

    for (_, master), (_, restored_master) in zip(
        opt._master_pairs, restored_opt._master_pairs, strict=True  # type: ignore[attr-defined]
    ):
        assert torch.equal(master, restored_master)


def test_load_state_dict_rejects_missing_or_truncated_masters() -> None:
    model = _hybrid_model()
    opt = create_torch_muon_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    state = opt.state_dict()

    missing = dict(state)
    missing.pop("masters", None)
    with pytest.raises(RuntimeError, match="Missing master weights"):
        opt.load_state_dict(missing)

    truncated = dict(state)
    truncated["masters"] = list(state["masters"])[:-1]
    with pytest.raises(RuntimeError, match="do not match optimizer parameters"):
        opt.load_state_dict(truncated)
