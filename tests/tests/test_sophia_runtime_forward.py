from __future__ import annotations

import pytest
import torch

from ml.runtime.model.config import ModelArgs
from ml.runtime.model.rope import precompute_freqs_cis
from ml.runtime.model.transformer import Transformer


def _model_args(**overrides: object) -> ModelArgs:
    base: dict[str, object] = {
        "max_batch_size": 2,
        "max_seq_len": 128,
        "vocab_size": 1024,
        "dim": 256,
        "n_layers": 2,
        "n_heads": 4,
        "head_dim": 32,
        "num_key_value_heads": 2,
        "rope_head_dim": 16,
        "ffn_hidden": 768,
        "use_qk_norm": True,
    }
    base.update(overrides)
    return ModelArgs(**base)


def test_runtime_scaled_rope_uses_configured_recipe_cpu() -> None:
    args = _model_args(
        max_seq_len=8192,
        dim=128,
        ffn_hidden=384,
        original_seq_len=4096,
        rope_factor=2.0,
    )
    model = Transformer(args).eval()

    expected = precompute_freqs_cis(
        args.rope_head_dim,
        args.max_seq_len,
        theta=args.rope_theta,
        original_seq_len=args.original_seq_len,
        rope_factor=args.rope_factor,
        beta_fast=args.beta_fast,
        beta_slow=args.beta_slow,
    )

    for layer in model.layers:
        torch.testing.assert_close(layer.attn.freqs_cis, expected)


def test_runtime_default_rope_recipe_uses_high_theta_defaults_cpu() -> None:
    args = _model_args(max_seq_len=512)
    model = Transformer(args).eval()

    assert int(args.original_seq_len) == 0
    expected = precompute_freqs_cis(
        args.rope_head_dim,
        args.max_seq_len,
        theta=args.rope_theta,
        original_seq_len=args.original_seq_len,
        rope_factor=args.rope_factor,
        beta_fast=args.beta_fast,
        beta_slow=args.beta_slow,
    )

    for layer in model.layers:
        torch.testing.assert_close(layer.attn.freqs_cis, expected)


def test_attention_refresh_state_resets_cache_cpu() -> None:
    args = _model_args(max_seq_len=64)
    model = Transformer(args).eval()
    attn = model.layers[0].attn

    attn.kv_cache.fill_(1)
    assert torch.count_nonzero(attn.kv_cache) > 0

    attn.refresh_state_buffers()
    assert torch.count_nonzero(attn.kv_cache) == 0


def test_model_args_reject_invalid_kv_head_count_cpu() -> None:
    with pytest.raises(ValueError, match="must be divisible by num_key_value_heads"):
        _model_args(num_key_value_heads=3)


def test_runtime_residual_output_projections_use_depth_scaled_init_cpu() -> None:
    torch.manual_seed(0)
    args = _model_args(dim=128, head_dim=32, n_layers=8, ffn_hidden=384)
    model = Transformer(args).eval()

    residual_std = 0.02 / ((2.0 * float(args.n_layers)) ** 0.5)
    attn_std = float(model.layers[0].attn.o_proj.weight.float().std(unbiased=False).item())
    down_std = float(model.layers[0].ffn.down_proj.weight.float().std(unbiased=False).item())
    qkv_std = float(model.layers[0].attn.qkv_proj.weight.float().std(unbiased=False).item())

    assert attn_std == pytest.approx(residual_std, rel=0.35)
    assert down_std == pytest.approx(residual_std, rel=0.35)
    assert qkv_std == pytest.approx(0.02, rel=0.25)


def test_runtime_uses_fused_qkv_and_gate_up_projections_cpu() -> None:
    layer = Transformer(_model_args()).layers[0]

    assert hasattr(layer.attn, "qkv_proj")
    assert not hasattr(layer.attn, "q_proj")
    assert not hasattr(layer.attn, "k_proj")
    assert not hasattr(layer.attn, "v_proj")
    assert hasattr(layer.ffn, "gate_up_proj")
    assert not hasattr(layer.ffn, "gate_proj")
    assert not hasattr(layer.ffn, "up_proj")


def test_runtime_block_uses_runtime_residual_names_cpu() -> None:
    block = Transformer(_model_args()).layers[0]

    assert hasattr(block, "attn_norm")
    assert hasattr(block, "attn")
    assert hasattr(block, "ffn_norm")
    assert hasattr(block, "ffn")
    assert not hasattr(block, "hc_attn")
    assert not hasattr(block, "hc_ffn")
