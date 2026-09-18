from __future__ import annotations

from dataclasses import fields
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as functional

from ml.runtime.model.attention import (
    CausalDepthwiseConv1d,
    SophiaKDA,
    SophiaMLA,
)
from ml.core.spec import ModelSpec
from ml.runtime.model.blocks import AttentionResidualMixer, FeedForward
from ml.runtime.model.config import ModelArgs
from ml.runtime.model.transformer import Transformer


def _model_args(**overrides: object) -> ModelArgs:
    values: dict[str, object] = {
        "max_batch_size": 2,
        "max_seq_len": 64,
        "vocab_size": 128,
        "dim": 64,
        "n_layers": 4,
        "num_heads": 2,
        "head_dim": 32,
        "ffn_hidden": 128,
        "kda_decay_rank": 16,
        "kda_output_gate_rank": 16,
        "mla_q_rank": 16,
        "mla_kv_rank": 16,
        "short_conv_kernel": 4,
        "kda_backend": "reference",
    }
    values.update(overrides)
    return ModelArgs(**values)


def test_runtime_uses_native_hybrid_layer_pattern_cpu() -> None:
    model = Transformer(_model_args()).eval()

    assert [layer.layer_type for layer in model.layers] == [
        "kda",
        "kda",
        "kda",
        "mla",
    ]
    assert all(isinstance(model.layers[index].attn, SophiaKDA) for index in range(3))
    assert isinstance(model.layers[3].attn, SophiaMLA)
    assert all(not hasattr(module, "rotary_embedding") for module in model.modules())
    assert "rope_theta" not in {field.name for field in fields(ModelSpec)}


def test_nope_model_remains_sensitive_to_token_order_cpu() -> None:
    torch.manual_seed(5)
    model = Transformer(_model_args(use_cache=False)).eval()
    tokens = torch.randint(0, model.args.vocab_size, (1, 12))
    reordered = tokens.clone()
    reordered[:, 2:10] = tokens[:, 2:10].flip(dims=(1,))

    with torch.no_grad():
        logits = model(tokens)
        reordered_logits = model(reordered)

    assert not torch.equal(logits[:, -1], reordered_logits[:, -1])


def test_mla_attention_logit_telemetry_matches_causal_reference_cpu() -> None:
    torch.manual_seed(17)
    module = SophiaMLA(_model_args(use_cache=False)).eval()
    module.enable_attention_logit_telemetry()
    x = torch.randn(2, 11, module.dim)

    with torch.no_grad():
        module(x)
        q = module.q_up(module.q_norm(module.q_down(x))).view(
            2, 11, module.num_heads, module.head_dim
        )
        latent = module.kv_norm(module.kv_down(x))
        k = module.k_up(latent).view(
            2, 11, module.num_heads, module.head_dim
        )
        logits = torch.einsum("bhqd,bhkd->bhqk", q.transpose(1, 2), k.transpose(1, 2))
        logits.mul_(module.head_dim**-0.5)
        causal = torch.ones((11, 11), dtype=torch.bool).tril_()
        logits.masked_fill_(~causal.view(1, 1, 11, 11), float("-inf"))
        expected = logits.amax(dim=(0, 2, 3)).float()

    torch.testing.assert_close(module.attention_logit_max, expected)


def test_mla_attention_logit_telemetry_matches_cached_causal_reference_cpu() -> None:
    torch.manual_seed(19)
    module = SophiaMLA(_model_args(use_cache=True)).eval()
    module.enable_attention_logit_telemetry()
    first = torch.randn(1, 5, module.dim)
    second = torch.randn(1, 3, module.dim)

    with torch.no_grad():
        module(first, start_pos=0)
        module.reset_attention_logit_max()
        module(second, start_pos=5)
        q = module.q_up(module.q_norm(module.q_down(second))).view(
            1, 3, module.num_heads, module.head_dim
        )
        latent = module.latent_cache[:1, :8]
        k = module.k_up(latent).view(1, 8, module.num_heads, module.head_dim)
        logits = torch.einsum(
            "bhqd,bhkd->bhqk",
            q.transpose(1, 2),
            k.transpose(1, 2),
        )
        logits.mul_(module.head_dim**-0.5)
        causal = torch.ones((3, 8), dtype=torch.bool).tril_(diagonal=5)
        logits.masked_fill_(~causal.view(1, 1, 3, 8), float("-inf"))
        expected = logits.amax(dim=(0, 2, 3)).float()

    torch.testing.assert_close(module.attention_logit_max, expected)


def test_runtime_forward_backward_is_finite_cpu() -> None:
    torch.manual_seed(0)
    model = Transformer(_model_args(use_cache=False)).train()
    tokens = torch.randint(0, model.args.vocab_size, (2, 8))

    logits = model(tokens)
    loss = logits.float().square().mean()
    loss.backward()

    assert logits.shape == (2, 8, model.args.vocab_size)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert model.layers[0].attn.q_proj.weight.grad is not None
    assert model.layers[3].attn.q_down.weight.grad is not None


def test_runtime_is_causal_cpu() -> None:
    torch.manual_seed(1)
    model = Transformer(_model_args(use_cache=False)).eval()
    left = torch.randint(0, model.args.vocab_size, (1, 12))
    right = left.clone()
    right[:, 6:] = torch.randint(0, model.args.vocab_size, (1, 6))

    with torch.no_grad():
        left_logits = model(left)
        right_logits = model(right)

    torch.testing.assert_close(left_logits[:, :6], right_logits[:, :6])


def test_cached_decode_matches_full_prompt_cpu() -> None:
    torch.manual_seed(2)
    full = Transformer(_model_args(use_cache=True)).eval()
    incremental = Transformer(_model_args(use_cache=True)).eval()
    incremental.load_state_dict(full.state_dict(), strict=True)
    tokens = torch.randint(0, full.args.vocab_size, (2, 10))

    with torch.no_grad():
        full_logits = full(tokens)
        pieces = [
            incremental(tokens[:, index : index + 1], start_pos=index)
            for index in range(tokens.size(1))
        ]

    torch.testing.assert_close(full_logits, torch.cat(pieces, dim=1), atol=1e-6, rtol=1e-5)


def test_runtime_initialization_cpu() -> None:
    torch.manual_seed(3)
    args = _model_args(n_layers=8, initializer_range=0.01)
    model = Transformer(args).eval()
    residual_std = args.initializer_range / (2.0 * args.n_layers) ** 0.5
    kda = model.layers[0].attn
    assert kda.q_proj.weight.float().std(unbiased=False).item() == pytest.approx(
        args.initializer_range, rel=0.25
    )
    assert kda.o_proj.weight.float().std(unbiased=False).item() == pytest.approx(
        residual_std, rel=0.35
    )
    assert hasattr(kda, "output_gate")
    assert not hasattr(kda, "output_gate_down")
    assert torch.count_nonzero(model.layers[0].attn_residual.query.weight) == 0
    assert torch.count_nonzero(model.output_attn_residual.query.weight) == 0


def test_attention_residual_starts_as_uniform_depth_mixing_cpu() -> None:
    mixer = AttentionResidualMixer(_model_args()).eval()
    torch.nn.init.zeros_(mixer.query.weight)
    completed = torch.randn(2, 3, 2, 64)
    partial = torch.randn(2, 3, 64)

    actual = mixer(partial, completed)
    expected = torch.cat((completed, partial.unsqueeze(2)), dim=2).mean(dim=2)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_attention_residual_autocast_preserves_source_dtype_cuda() -> None:
    mixer = AttentionResidualMixer(_model_args(dim=64)).to(
        device="cuda", dtype=torch.bfloat16
    )
    completed = torch.randn(1, 3, 2, 64, device="cuda", dtype=torch.bfloat16)
    partial = torch.randn(1, 3, 64, device="cuda", dtype=torch.bfloat16)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual = mixer(partial, completed)

    assert actual.dtype is torch.bfloat16


def test_situ_glu_matches_k3_softcap_formula_cpu() -> None:
    args = _model_args(dim=4, ffn_hidden=4)
    ffn = FeedForward(args).eval()
    with torch.no_grad():
        ffn.gate_up_proj.weight.zero_()
        ffn.down_proj.weight.copy_(torch.eye(4))
        ffn.gate_up_proj.weight[:4].copy_(torch.eye(4))
        ffn.gate_up_proj.weight[4:].copy_(2.0 * torch.eye(4))
    x = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]])

    actual = ffn(x)
    gate = x
    up = 2.0 * x
    expected = (
        args.situ_gate_softcap
        * torch.tanh(gate / args.situ_gate_softcap)
        * torch.sigmoid(gate)
        * args.situ_up_softcap
        * torch.tanh(up / args.situ_up_softcap)
    )

    torch.testing.assert_close(actual, expected)


def test_attention_residual_supports_gradient_checkpointing_cpu() -> None:
    torch.manual_seed(4)
    model = Transformer(_model_args(use_cache=False)).train()
    model.gradient_checkpointing = True
    tokens = torch.randint(0, model.args.vocab_size, (2, 8))

    loss = model(tokens).float().square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert model.layers[0].attn_residual.query.weight.grad is not None
    assert model.layers[-1].ffn_residual.query.weight.grad is not None


def test_short_convolution_kernel_one_has_empty_history_cpu() -> None:
    convolution = CausalDepthwiseConv1d(channels=4, kernel_size=1)
    output, history = convolution(torch.randn(2, 3, 4))

    assert output.shape == (2, 3, 4)
    assert history.shape == (2, 4, 0)


def test_runtime_block_uses_k3_depth_residual_names_cpu() -> None:
    block = Transformer(_model_args()).layers[0]

    assert hasattr(block, "attn_norm")
    assert hasattr(block, "attn")
    assert hasattr(block, "ffn_norm")
    assert hasattr(block, "ffn")
    assert hasattr(block.ffn, "gate_up_proj")
    assert hasattr(block, "attn_residual")
    assert hasattr(block, "ffn_residual")
    assert block.attn_res_block_size == 4


def _wide_fp32_mixer_forward(
    module: AttentionResidualMixer,
    partial_block: torch.Tensor,
    completed_blocks: torch.Tensor,
) -> torch.Tensor:
    values = torch.cat((completed_blocks, partial_block.unsqueeze(2)), dim=2)
    values_float = values.float()
    keys = module.norm(values_float)
    score_weight = module.query.weight.squeeze(0).float()
    scores = torch.einsum("bsth,h->bst", keys, score_weight)
    weights = scores.softmax(dim=2).unsqueeze(-1)
    return (weights * values_float).sum(dim=2).to(dtype=partial_block.dtype)


def _source_dtype_mla_gate_forward(
    module: SophiaMLA,
    x: torch.Tensor,
    *,
    start_pos: int = 0,
) -> torch.Tensor:
    bsz, seqlen, _ = x.shape
    q = module.q_up(module.q_norm(module.q_down(x))).view(
        bsz, seqlen, module.num_heads, module.head_dim
    )
    latent = module.kv_norm(module.kv_down(x))
    k = module.k_up(latent).view(
        bsz, int(latent.size(1)), module.num_heads, module.head_dim
    )
    v = module.v_up(latent).view_as(k)
    q_t, k_t, v_t = (value.transpose(1, 2) for value in (q, k, v))
    bias = module._causal_bias(q, k, int(start_pos))
    output = functional.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        attn_mask=bias,
        is_causal=(bias is None),
    ).transpose(1, 2)
    gate = torch.sigmoid(module.output_gate(x)).view_as(output)
    return module.o_proj((output * gate).reshape(bsz, seqlen, module.inner_dim))


def _loss_and_grad_norm(model: Transformer, tokens: torch.Tensor) -> tuple[float, float]:
    logits = model(tokens)
    loss = logits.float().square().mean()
    loss.backward()
    grad_norm = torch.sqrt(
        sum(
            parameter.grad.float().square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    return float(loss.detach()), float(grad_norm.detach())


def test_k3_fp32_scope_matches_wide_fp32_reference_bf16_cpu() -> None:
    torch.manual_seed(6)
    args = _model_args(use_cache=False)
    reference = Transformer(args).to(dtype=torch.bfloat16).train()
    candidate = Transformer(args).to(dtype=torch.bfloat16).train()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    tokens = torch.randint(0, args.vocab_size, (2, 8))

    with (
        patch.object(AttentionResidualMixer, "forward", _wide_fp32_mixer_forward),
        patch.object(SophiaMLA, "forward", _source_dtype_mla_gate_forward),
    ):
        reference_loss, reference_grad_norm = _loss_and_grad_norm(reference, tokens)
    candidate_loss, candidate_grad_norm = _loss_and_grad_norm(candidate, tokens)

    assert candidate_loss == pytest.approx(reference_loss, rel=5e-3, abs=5e-3)
    assert candidate_grad_norm == pytest.approx(
        reference_grad_norm, rel=5e-3, abs=5e-3
    )
