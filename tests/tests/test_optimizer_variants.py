import pytest
import torch

from ml.training.pretrain.engine.grad_clip import clip_gradients_
from ml.training.pretrain.engine.grad_clip import scale_grads_by_token_count_
from ml.training.pretrain.muon import (
    MUON_ORTHO_BATCH_MAX_DEFAULT,
    MUON_ORTHO_BATCH_MAX_ENV,
    Muon,
    _resolve_ortho_batch_max,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_batched,
)
from ml.training.pretrain.optimizer import (
    TorchMuonFusedAdamW,
    _group_params_for_hybrid_layerwise,
    _split_params_for_hybrid,
    create_torch_muon_optimizer,
)
from ml.training.pretrain.batch_size_autofit import create_optimizer, move_optimizer_state_to_device


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
    )
    assert isinstance(opt, TorchMuonFusedAdamW)

    state = opt.state_dict()
    opt.load_state_dict(state)
    move_optimizer_state_to_device(opt, "cpu")


def test_hybrid_auto_clip_routes_through_hybrid_optimizer() -> None:
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

    grad_norm = clip_gradients_(
        model=model,
        optimizer=opt,
        grad_clip_mode="hybrid_auto",
        max_grad_norm=1.0,
        agc_clip=0.01,
        agc_eps=1e-3,
        agc_exclude_bias_and_norm=True,
    )

    assert grad_norm is not None
    assert torch.isfinite(grad_norm)


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


def test_hybrid_auto_clip_requires_hybrid_optimizer() -> None:
    model = _toy_model()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    loss = model(input_ids).float().sum()
    loss.backward()

    with pytest.raises(ValueError, match="requires an optimizer"):
        clip_gradients_(
            model=model,
            optimizer=None,
            grad_clip_mode="hybrid_auto",
            max_grad_norm=1.0,
            agc_clip=0.01,
            agc_eps=1e-3,
            agc_exclude_bias_and_norm=True,
        )


def test_agc_requires_foreach_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _toy_model()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    loss = model(input_ids).float().sum()
    loss.backward()
    monkeypatch.delattr(torch, "_foreach_norm", raising=False)

    with pytest.raises(RuntimeError, match="torch._foreach_norm is required"):
        clip_gradients_(
            model=model,
            optimizer=None,
            grad_clip_mode="agc",
            max_grad_norm=1.0,
            agc_clip=0.01,
            agc_eps=1e-3,
            agc_exclude_bias_and_norm=True,
        )


def test_token_grad_scaling_requires_foreach_mul(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _toy_model()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    loss = model(input_ids).float().sum()
    loss.backward()
    monkeypatch.delattr(torch, "_foreach_mul_", raising=False)

    with pytest.raises(RuntimeError, match="torch._foreach_mul_ is required"):
        scale_grads_by_token_count_(
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
    )
    assert isinstance(opt, TorchMuonFusedAdamW)


def test_muon_target_rms_respects_learning_rate_magnitude() -> None:
    p1 = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
    p2 = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
    p1.grad = torch.ones_like(p1)
    p2.grad = torch.ones_like(p2)
    opt1 = Muon([p1], lr=0.1, momentum=0.0, weight_decay=0.0, adjust_lr_fn=None, target_rms=0.18)
    opt2 = Muon([p2], lr=0.2, momentum=0.0, weight_decay=0.0, adjust_lr_fn=None, target_rms=0.18)
    before1 = p1.detach().clone()
    before2 = p2.detach().clone()
    opt1.step()
    opt2.step()
    delta1 = (before1 - p1).float().square().mean().sqrt().item()
    delta2 = (before2 - p2).float().square().mean().sqrt().item()
    assert delta2 > delta1 * 1.5


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
        adjust_lr_fn=None,
        target_rms=0.18,
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
        adjust_lr_fn=None,
        target_rms=0.18,
    )

    opt.step()

    momentum_buffer = opt.state[param]["momentum_buffer"]
    torch.testing.assert_close(momentum_buffer, torch.zeros_like(momentum_buffer))


def test_muon_ortho_batch_max_defaults_to_single_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(MUON_ORTHO_BATCH_MAX_ENV, raising=False)
    assert MUON_ORTHO_BATCH_MAX_DEFAULT == 1
    assert _resolve_ortho_batch_max() == 1


def test_muon_ortho_batch_max_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MUON_ORTHO_BATCH_MAX_ENV, "8")
    assert _resolve_ortho_batch_max() == 8
    monkeypatch.setenv(MUON_ORTHO_BATCH_MAX_ENV, "-1")
    assert _resolve_ortho_batch_max() == MUON_ORTHO_BATCH_MAX_DEFAULT


def test_muon_orthogonalization_chunking_matches_batched_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    grads = [torch.randn(3, 5) for _ in range(5)]

    monkeypatch.setenv(MUON_ORTHO_BATCH_MAX_ENV, "0")
    batched = Muon._orthogonalized_updates(
        grads,
        ns_steps=2,
        eps=1e-7,
        ns_coefficients=(3.4445, -4.7750, 2.0315),
        target_rms=0.18,
    )
    monkeypatch.setenv(MUON_ORTHO_BATCH_MAX_ENV, "2")
    chunked = Muon._orthogonalized_updates(
        grads,
        ns_steps=2,
        eps=1e-7,
        ns_coefficients=(3.4445, -4.7750, 2.0315),
        target_rms=0.18,
    )

    assert len(chunked) == len(batched)
    for chunked_update, batched_update in zip(chunked, batched, strict=True):
        torch.testing.assert_close(chunked_update, batched_update)


def test_batched_zeropower_matches_per_matrix_reference() -> None:
    torch.manual_seed(42)
    mats = torch.randn(5, 8, 4, dtype=torch.float32)
    batched = zeropower_via_newtonschulz5_batched(mats, steps=4, eps=1e-7)
    ref = torch.stack(
        [
            zeropower_via_newtonschulz5(mat, steps=4, eps=1e-7)
            for mat in mats
        ],
        dim=0,
    )
    torch.testing.assert_close(batched, ref, atol=1e-5, rtol=1e-5)


class _SophiaLikeTinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"num_hidden_layers": 2})()
        self.model = torch.nn.Module()
        self.model.tok_embeddings = torch.nn.Embedding(32, 8)
        self.model.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(8, 8, bias=False),
                torch.nn.Linear(8, 8, bias=False),
            ]
        )
        self.model.norm = torch.nn.LayerNorm(8)
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


def test_sophia_embedding_and_output_params_stay_out_of_layerwise_muon_groups() -> None:
    model = _SophiaLikeTinyModel()

    muon_groups, adamw_groups = _group_params_for_hybrid_layerwise(
        model,
        lr=1e-3,
        weight_decay=0.01,
        layerwise_lr_decay=0.9,
    )

    muon_ids = {id(param) for group in muon_groups for param in group.params}
    adamw_ids = {id(param) for group in adamw_groups for param in group.params}

    assert id(model.model.tok_embeddings.weight) not in muon_ids
    assert id(model.model.tok_embeddings.weight) in adamw_ids
    assert sum(
        1
        for group in adamw_groups
        for param in group.params
        if id(param) == id(model.model.tok_embeddings.weight)
    ) == 1


def test_embedding_group_uses_dedicated_lr_scale() -> None:
    model = _SophiaLikeTinyModel()

    opt = create_torch_muon_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
        embedding_lr_scale=0.5,
    )

    embed_groups = [
        group
        for group in opt._adamw_opt.param_groups
        if str(group.get("name", "")) == "adamw_embed"
    ]
    assert len(embed_groups) == 1
    assert float(embed_groups[0]["lr"]) == pytest.approx(5e-4)


def test_embedding_group_has_no_weight_decay() -> None:
    model = _SophiaLikeTinyModel()

    opt = create_torch_muon_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.05,
        betas=(0.9, 0.95),
        eps=1e-8,
        embedding_lr_scale=0.5,
    )

    embed_groups = [
        group
        for group in opt._adamw_opt.param_groups
        if str(group.get("name", "")) == "adamw_embed"
    ]
    assert len(embed_groups) == 1
    # z-loss covers output-side regularization; decaying the tied embedding only
    # shrinks rare-token rows.
    assert float(embed_groups[0]["weight_decay"]) == pytest.approx(0.0)
