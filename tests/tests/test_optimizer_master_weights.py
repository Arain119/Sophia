"""
Regression guard for optimizer master-weight precision.

The optimizer keeps fp32 master weights so sub-bf16-ULP updates accumulate instead of
rounding away. The bf16 model remains the compute copy and is refreshed after each
optimizer step.
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


def test_formal_hybrid_muon_momentum_is_bf16() -> None:
    model = _matrix_model(16)
    opt = _opt(model, lr=1e-3)
    model.weight.grad = torch.randn_like(model.weight)
    opt.step()

    state = next(iter(opt._muon_opt.state.values()))  # type: ignore[attr-defined]
    assert state["momentum_buffer"].dtype == torch.bfloat16


def test_sub_ulp_updates_accumulate_via_fp32_masters() -> None:
    torch.manual_seed(0)
    model = _matrix_model(64)
    with torch.no_grad():
        model.weight.fill_(0.5)  # bf16 ULP here is ~2e-3
    init = model.weight.detach().float().clone()

    opt = _opt(model, lr=1e-3)
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


def test_fp32_main_gradient_buffer_drives_update_and_is_transient() -> None:
    torch.manual_seed(3)
    model = _matrix_model(16)
    opt = _opt(model, lr=1e-3)
    main_grad = torch.randn_like(model.weight, dtype=torch.float32)
    model.weight.grad = torch.zeros_like(model.weight)
    opt.set_gradient_buffers(((model.weight, main_grad),))
    master_before = opt._master_pairs[0][1].detach().clone()  # type: ignore[attr-defined]

    opt.step()

    master = opt._master_pairs[0][1]  # type: ignore[attr-defined]
    assert not torch.equal(master, master_before)
    assert master.grad is None
    assert "gradient_buffers" not in opt.state_dict()

    opt.zero_grad(set_to_none=False)
    assert torch.count_nonzero(main_grad).item() == 0
    assert torch.count_nonzero(model.weight.grad).item() == 0


def test_fp32_main_gradient_buffer_contract_is_exact() -> None:
    model = _matrix_model(8)
    opt = _opt(model, lr=1e-3)

    with pytest.raises(RuntimeError, match="do not match optimizer parameters"):
        opt.set_gradient_buffers(())
    with pytest.raises(TypeError, match="float32"):
        opt.set_gradient_buffers(
            ((model.weight, torch.zeros_like(model.weight, dtype=torch.bfloat16)),)
        )
    with pytest.raises(ValueError, match="shape/device"):
        opt.set_gradient_buffers(
            ((model.weight, torch.zeros(1, dtype=torch.float32)),)
        )


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


def test_state_dict_roundtrip_restores_muon_momentum_dtype() -> None:
    torch.manual_seed(1)
    model = _matrix_model(16)
    opt = _opt(model, lr=1e-3)
    model.weight.grad = torch.randn_like(model.weight)
    opt.step()
    state = opt.state_dict()

    restored_model = _matrix_model(16)
    restored_opt = _opt(restored_model, lr=1e-3)
    restored_opt.load_state_dict(state)

    restored_state = next(iter(restored_opt._muon_opt.state.values()))  # type: ignore[attr-defined]
    assert restored_state["momentum_buffer"].dtype == torch.bfloat16


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


def test_hybrid_sequential_master_grad_sync_matches_all_at_once_reference() -> None:
    torch.manual_seed(7)
    model = _hybrid_model()
    reference_model = _hybrid_model()
    reference_model.load_state_dict(model.state_dict())
    opt = create_torch_muon_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    reference_opt = create_torch_muon_optimizer(
        reference_model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    model(input_ids).float().sum().backward()
    reference_model(input_ids).float().sum().backward()

    opt.step()
    reference_opt._sync_grads_to_masters()  # type: ignore[attr-defined]
    reference_opt._muon_opt.step()  # type: ignore[attr-defined]
    reference_opt._adamw_opt.step()  # type: ignore[attr-defined]
    reference_opt._sync_masters_to_model()  # type: ignore[attr-defined]

    for parameter, reference_parameter in zip(
        model.parameters(), reference_model.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter, reference_parameter, rtol=0.0, atol=0.0)
    for (_model_param, master), (_reference_model_param, reference_master) in zip(
        opt._master_pairs, reference_opt._master_pairs, strict=True  # type: ignore[attr-defined]
    ):
        torch.testing.assert_close(master, reference_master, rtol=0.0, atol=0.0)
        assert master.grad is None


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
