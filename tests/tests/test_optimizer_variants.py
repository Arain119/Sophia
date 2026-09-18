import pytest
import torch

from ml.modeling.sophia_decoder import SophiaDecoder, SophiaDecoderConfig
from ml.runtime.model.ops import RMSNorm
from ml.runtime.model.attention import SophiaMLA
from ml.runtime.model.config import ModelArgs
from ml.training.pretrain.engine.gradients import (
    global_grad_norm,
    scale_gradients_by_token_count,
)
from ml.training.pretrain.engine.train_step import check_finite_update_state
from ml.training.pretrain.muon import (
    Muon,
    adjust_lr_for_muon,
    zeropower_via_newtonschulz5,
)
from ml.training.pretrain.optimizer import (
    TorchMuonFusedAdamW,
    _split_params_for_hybrid,
    create_torch_muon_optimizer,
)
from ml.training.pretrain.optimizer_diagnostics import adamw_group_diagnostics
from ml.training.runtime_tools import create_optimizer, move_optimizer_state_to_device


def _toy_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Embedding(32, 8),
        torch.nn.Linear(8, 8),
        torch.nn.LayerNorm(8),
    )


def test_create_torch_muon_hybrid_optimizer() -> None:
    model = _toy_model()
    opt = create_torch_muon_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
        muon_ns_steps=5,
    )
    assert isinstance(opt, TorchMuonFusedAdamW)

    state = opt.state_dict()
    opt.load_state_dict(state)
    move_optimizer_state_to_device(opt, "cpu")


def test_hybrid_optimizer_materializes_state_without_taking_a_step() -> None:
    model = _toy_model()
    opt = create_torch_muon_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    masters_before = [master.detach().clone() for _, master in opt._master_pairs]

    opt.initialize_state()

    assert all(
        "momentum_buffer" in opt._muon_opt.state[parameter]
        for group in opt._muon_opt.param_groups
        for parameter in group["params"]
    )
    assert opt._adamw_opt is not None
    assert all(
        set(opt._adamw_opt.state[parameter]) == {"step", "exp_avg", "exp_avg_sq"}
        and float(opt._adamw_opt.state[parameter]["step"]) == 0.0
        for group in opt._adamw_opt.param_groups
        for parameter in group["params"]
    )
    for before, (_model_parameter, master) in zip(
        masters_before, opt._master_pairs, strict=True
    ):
        torch.testing.assert_close(master, before)


def test_qk_clip_scales_model_and_master_weights_per_head() -> None:
    module = SophiaMLA(
        ModelArgs(
            vocab_size=32,
            dim=8,
            n_layers=4,
            num_heads=2,
            head_dim=4,
            ffn_hidden=16,
            kda_decay_rank=4,
            kda_output_gate_rank=4,
            mla_q_rank=4,
            mla_kv_rank=4,
            max_seq_len=8,
            max_batch_size=1,
            kda_backend="reference",
        )
    ).to(dtype=torch.bfloat16)
    optimizer = create_torch_muon_optimizer(
        module,
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    with torch.no_grad():
        for projection in (module.q_up, module.k_up):
            projection.weight.fill_(1.0)
            optimizer._master_by_model_id[id(projection.weight)].fill_(1.0)

    optimizer.apply_mla_qk_clip(
        (module,),
        torch.tensor([[400.0, 25.0]]),
        threshold=100.0,
    )

    for projection in (module.q_up, module.k_up):
        model_heads = projection.weight.view(2, 4, -1).float()
        master_heads = optimizer._master_by_model_id[id(projection.weight)].view(
            2, 4, -1
        )
        torch.testing.assert_close(model_heads[0], torch.full_like(model_heads[0], 0.5))
        torch.testing.assert_close(model_heads[1], torch.ones_like(model_heads[1]))
        torch.testing.assert_close(master_heads, model_heads)


def test_global_grad_norm_is_finite() -> None:
    model = _toy_model()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    loss = model(input_ids).float().sum()
    loss.backward()

    grad_norm = global_grad_norm(model.parameters())

    assert grad_norm is not None
    assert torch.isfinite(grad_norm)


def test_gradient_helpers_operate_on_explicit_fp32_buffers() -> None:
    model = _toy_model()
    gradients = tuple(
        torch.ones_like(parameter, dtype=torch.float32)
        for parameter in model.parameters()
    )

    scale_gradients_by_token_count(
        model=model,
        supervised_tokens=torch.tensor(4),
        gradients=gradients,
    )
    grad_norm = global_grad_norm(model.parameters(), gradients=gradients)

    assert grad_norm is not None
    expected = sum(gradient.numel() for gradient in gradients) ** 0.5 / 4.0
    assert float(grad_norm) == pytest.approx(expected)


def test_torch_muon_hybrid_optimizer_exposes_diagnostics() -> None:
    model = _toy_model()
    opt = create_torch_muon_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    loss = model(input_ids).float().sum()
    loss.backward()
    opt.step()

    diagnostics = opt.diagnostics()

    assert float(diagnostics["optimizer/muon/group_count"]) >= 1.0
    assert float(diagnostics["optimizer/adamw/group_count"]) >= 1.0
    effective_lr_keys = [
        key for key in diagnostics if str(key).endswith("/effective_lr_max")
    ]
    assert effective_lr_keys
    assert max(float(diagnostics[key]) for key in effective_lr_keys) > 0.0


def test_adamw_diagnostics_reduce_before_scalar_transforms() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3))
    payload = adamw_group_diagnostics(
        param_groups=(
            {
                "name": "adamw_test",
                "params": [parameter],
                "lr": 0.01,
                "eps": 1e-8,
            },
        ),
        state={parameter: {"exp_avg_sq": torch.tensor([9.0, 4.0, 16.0])}},
    )

    assert payload["optimizer/adamw_test/denom_min"] == pytest.approx(2.0 + 1e-8)
    assert payload["optimizer/adamw_test/effective_lr_max"] == pytest.approx(
        0.01 / (2.0 + 1e-8)
    )


def _initialized_hybrid_optimizer() -> tuple[torch.nn.Module, TorchMuonFusedAdamW]:
    model = _toy_model()
    opt = create_torch_muon_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    model(input_ids).float().sum().backward()
    opt.step()
    assert isinstance(opt, TorchMuonFusedAdamW)
    return model, opt


def test_hybrid_finite_check_covers_master_weights() -> None:
    model, opt = _initialized_hybrid_optimizer()
    model_param, master = opt._master_pairs[0]
    with torch.no_grad():
        master.fill_(float("nan"))

    with pytest.raises(RuntimeError, match="master"):
        check_finite_update_state(model=model, optimizer=opt)


def test_hybrid_finite_check_covers_model_parameters() -> None:
    model, opt = _initialized_hybrid_optimizer()
    model_param, _master = opt._master_pairs[0]
    with torch.no_grad():
        model_param.fill_(float("nan"))

    with pytest.raises(RuntimeError, match="parameter"):
        check_finite_update_state(model=model, optimizer=opt)


def test_hybrid_finite_check_covers_muon_state() -> None:
    model, opt = _initialized_hybrid_optimizer()
    state = next(iter(opt._muon_opt.state.values()))
    with torch.no_grad():
        state["momentum_buffer"].fill_(float("inf"))

    with pytest.raises(RuntimeError, match="muon"):
        check_finite_update_state(model=model, optimizer=opt)


def test_hybrid_finite_check_covers_adamw_state() -> None:
    model, opt = _initialized_hybrid_optimizer()
    assert opt._adamw_opt is not None
    state = next(iter(opt._adamw_opt.state.values()))
    with torch.no_grad():
        state["exp_avg"].fill_(float("nan"))

    with pytest.raises(RuntimeError, match="adamw"):
        check_finite_update_state(model=model, optimizer=opt)


def test_hybrid_finite_check_materializes_only_scalar_finite_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, opt = _initialized_hybrid_optimizer()
    torch_isfinite = torch.isfinite

    def scalar_isfinite_only(tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.numel() == 1
        return torch_isfinite(tensor)

    monkeypatch.setattr(torch, "isfinite", scalar_isfinite_only)
    check_finite_update_state(model=model, optimizer=opt)


def test_global_grad_norm_requires_foreach_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _toy_model()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    loss = model(input_ids).float().sum()
    loss.backward()
    monkeypatch.delattr(torch, "_foreach_norm", raising=False)

    with pytest.raises(RuntimeError, match="torch._foreach_norm is required"):
        global_grad_norm(model.parameters())


def test_token_grad_scaling_requires_foreach_mul(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _toy_model()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    loss = model(input_ids).float().sum()
    loss.backward()
    monkeypatch.delattr(torch, "_foreach_mul_", raising=False)

    with pytest.raises(RuntimeError, match="torch._foreach_mul_ is required"):
        scale_gradients_by_token_count(
            model=model,
            supervised_tokens=torch.tensor(3),
        )


def test_create_optimizer_uses_torch_muon_hybrid() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for create_optimizer")

    model = _toy_model()
    opt = create_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
        muon_ns_steps=5,
    )
    assert isinstance(opt, TorchMuonFusedAdamW)


def test_muon_learning_rate_matches_moonlight_formula() -> None:
    lr = 4e-4
    expected = lr * 0.2 * (3968**0.5)

    assert adjust_lr_for_muon(lr, torch.Size((3968, 1536))) == pytest.approx(expected)
    assert adjust_lr_for_muon(lr, torch.Size((1536, 3968))) == pytest.approx(expected)


def test_muon_decoupled_weight_decay_shrinks_params_without_gradients() -> None:
    param = torch.nn.Parameter(
        torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    )
    param.grad = torch.zeros_like(param)
    opt = Muon(
        [param],
        lr=0.1,
        momentum=0.0,
        weight_decay=0.2,
        eps=1e-7,
    )

    opt.step()

    expected = torch.tensor([[1.96, 0.0], [0.0, 1.96]], dtype=torch.float32)
    torch.testing.assert_close(param.detach(), expected)


def test_muon_weight_decay_does_not_change_momentum_buffer_when_grad_is_zero() -> None:
    param = torch.nn.Parameter(torch.eye(2, dtype=torch.float32))
    param.grad = torch.zeros_like(param)
    opt = Muon(
        [param],
        lr=0.1,
        momentum=0.9,
        weight_decay=0.2,
        eps=1e-7,
    )

    opt.step()

    momentum_buffer = opt.state[param]["momentum_buffer"]
    torch.testing.assert_close(momentum_buffer, torch.zeros_like(momentum_buffer))


def test_muon_nesterov_does_not_mutate_gradient_tensor() -> None:
    param = torch.nn.Parameter(torch.eye(3, dtype=torch.float32))
    gradient = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    param.grad = gradient.clone()
    opt = Muon(
        [param],
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0,
        eps=1e-7,
    )

    opt.step()

    torch.testing.assert_close(param.grad, gradient)


def test_newton_schulz_matches_explicit_muon_polynomial() -> None:
    torch.manual_seed(123)
    gradient = torch.randn(3, 5, dtype=torch.float32)
    gradient_before = gradient.clone()
    coeffs = (3.4445, -4.7750, 2.0315)
    actual = zeropower_via_newtonschulz5(
        gradient, steps=1, eps=1e-7, coeffs=coeffs
    )

    normalized = gradient / (gradient.norm() + 1e-7)
    gram = normalized @ normalized.T
    expected = (
        coeffs[0] * normalized
        + (coeffs[1] * gram + coeffs[2] * (gram @ gram)) @ normalized
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(gradient, gradient_before, atol=0.0, rtol=0.0)


def test_rank_one_muon_update_scale_follows_ns5_scalar_trajectory() -> None:
    eps = 1e-7
    base_lr = 8.825e-4
    gradient = torch.ones((1, 1536), dtype=torch.float32)
    actual = zeropower_via_newtonschulz5(
        gradient,
        steps=5,
        eps=eps,
        coeffs=(3.4445, -4.7750, 2.0315),
    )

    singular_value = float(gradient.norm()) / (float(gradient.norm()) + eps)
    for _ in range(5):
        singular_value = (
            3.4445 * singular_value
            - 4.7750 * singular_value**3
            + 2.0315 * singular_value**5
        )

    assert float(actual.norm()) == pytest.approx(singular_value, rel=1e-5)
    assert singular_value == pytest.approx(0.696435, rel=1e-5)

    scaled_update = actual * adjust_lr_for_muon(base_lr, gradient.shape)
    assert float(scaled_update.square().mean().sqrt()) == pytest.approx(
        0.2 * singular_value * base_lr,
        rel=1e-5,
    )


def test_newton_schulz_rejects_invalid_iteration_contract() -> None:
    matrix = torch.ones(2, 3)
    with pytest.raises(ValueError, match="exactly three"):
        zeropower_via_newtonschulz5(matrix, steps=1, eps=1e-7, coeffs=(1.0, 2.0))
    with pytest.raises(ValueError, match="non-negative"):
        zeropower_via_newtonschulz5(matrix, steps=-1, eps=1e-7)
    with pytest.raises(ValueError, match="positive"):
        zeropower_via_newtonschulz5(matrix, steps=1, eps=-1.0)


class _SophiaLikeTinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"n_layers": 2})()
        self.model = torch.nn.Module()
        self.model.tok_embeddings = torch.nn.Embedding(32, 8)
        self.model.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(8, 8, bias=False),
                torch.nn.Linear(8, 8, bias=False),
            ]
        )
        self.model.norm = RMSNorm(8)
        self.output = torch.nn.Linear(8, 32, bias=False)


def test_sophia_embedding_and_output_params_never_enter_muon_groups() -> None:
    model = _SophiaLikeTinyModel()

    muon_params, embed_params, decay_params, _ = _split_params_for_hybrid(model)
    muon_ids = {id(param) for param in muon_params}
    embed_ids = {id(param) for param in embed_params}
    decay_ids = {id(param) for param in decay_params}

    assert id(model.model.tok_embeddings.weight) not in muon_ids
    assert id(model.model.tok_embeddings.weight) in embed_ids
    assert id(model.model.tok_embeddings.weight) not in decay_ids
    assert (
        len(
            [
                param
                for param in embed_params
                if id(param) == id(model.model.tok_embeddings.weight)
            ]
        )
        == 1
    )


def test_rmsnorm_and_auxiliary_groups_use_zero_weight_decay() -> None:
    model = _SophiaLikeTinyModel()

    opt = create_torch_muon_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.05,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    embed_groups = [
        group
        for group in opt._adamw_opt.param_groups
        if str(group.get("name", "")) == "adamw_embed"
    ]
    assert len(embed_groups) == 1
    assert float(embed_groups[0]["lr"]) == pytest.approx(1e-3)
    assert float(embed_groups[0]["weight_decay"]) == pytest.approx(0.05)
    norm = model.model.norm.weight
    norm_master = next(
        master
        for model_param, master in opt._master_pairs
        if id(model_param) == id(norm)
    )
    norm_group = next(
        group
        for group in opt._adamw_opt.param_groups
        if any(id(param) == id(norm_master) for param in group["params"])
    )
    assert str(norm_group["name"]) == "adamw_no_decay"
    assert float(norm_group["weight_decay"]) == pytest.approx(0.0)


def test_release_model_rmsnorm_parameters_use_zero_decay() -> None:
    with torch.device("meta"):
        model = SophiaDecoder(SophiaDecoderConfig(), runtime_max_seq_len=4096)

    _, _, decay_params, no_decay_params = _split_params_for_hybrid(model)
    decay_ids = {id(param) for param in decay_params}
    no_decay_ids = {id(param) for param in no_decay_params}
    norm_params = [
        module.weight for module in model.modules() if isinstance(module, RMSNorm)
    ]

    assert len(norm_params) == 149
    assert all(id(param) in no_decay_ids for param in norm_params)
    assert not any(id(param) in decay_ids for param in norm_params)
