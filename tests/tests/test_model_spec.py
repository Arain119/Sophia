from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ml.integrations.adapters.hf.config import build_config
from ml.modeling.config import SophiaModelConfig
from ml.integrations.adapters.hf.config import SophiaConfig as AdapterConfig
from ml.modeling.sophia_decoder import SophiaDecoderConfig
from ml.core.spec import ModelSpec


def test_default_model_spec_uses_decoder_defaults() -> None:
    spec = ModelSpec()

    assert int(spec.rope_head_dim) == 64
    assert int(spec.dim) == 1536
    assert int(spec.n_layers) == 30
    assert int(spec.n_heads) == 12
    assert int(spec.ffn_hidden) == 4096
    assert int(spec.original_seq_len) == 0


def test_default_model_spec_uses_high_theta_rope() -> None:
    spec = ModelSpec.default()

    # Decoder trains up to max_seq_len with a high RoPE base;
    # original_seq_len == 0 keeps the standard RoPE recipe active by default.
    assert float(spec.rope_theta) == 500000.0
    assert int(spec.original_seq_len) == 0


def test_specs_are_frozen() -> None:
    spec = ModelSpec()
    with pytest.raises(FrozenInstanceError):
        spec.dim = 2048  # type: ignore[misc]


def test_model_spec_roundtrips_through_config_and_runtime_args() -> None:
    spec = ModelSpec.from_mapping(
        {
            "vocab_size": 256,
            "dim": 128,
            "n_layers": 2,
            "n_heads": 4,
            "head_dim": 32,
            "num_key_value_heads": 2,
            "rope_head_dim": 16,
            "max_seq_len": 64,
            "ffn_hidden": 384,
        }
    )

    cfg = spec.to_config()
    args = spec.to_model_args(runtime_max_seq_len=32)

    assert int(cfg.ffn_hidden) == 384
    assert int(args.ffn_hidden) == 384
    assert int(args.max_seq_len) == 32

    cfg_args = cfg.to_model_args(runtime_max_seq_len=32)
    assert cfg_args == args


def test_model_spec_with_overrides_applies_updates() -> None:
    spec = ModelSpec.default().with_overrides(
        dim=384,
        max_seq_len=8192,
    )

    assert int(spec.dim) == 384
    assert int(spec.max_seq_len) == 8192


def test_model_spec_rejects_unknown_runtime_fields() -> None:
    with pytest.raises(ValueError, match="unknown override keys: unknown_runtime_field"):
        ModelSpec.default().with_overrides(unknown_runtime_field=2)

    with pytest.raises(ValueError, match="unknown override keys: unknown_loss_field"):
        ModelSpec.default().with_overrides(unknown_loss_field=1)


def test_canonical_config_projects_to_hf_only_via_adapter() -> None:
    cfg = SophiaModelConfig(
        vocab_size=256,
        dim=128,
        n_layers=2,
        n_heads=4,
        head_dim=32,
        num_key_value_heads=2,
        rope_head_dim=16,
        ffn_hidden=384,
        max_seq_len=64,
    )

    assert not hasattr(cfg, "to_hf_config")

    class _DummyHFConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    projected = build_config(cfg, config_cls=_DummyHFConfig)

    assert isinstance(projected, _DummyHFConfig)
    assert projected.kwargs["vocab_size"] == 256
    assert projected.kwargs["dim"] == 128
    assert projected.kwargs["n_layers"] == 2


def test_sophia_decoder_config_can_be_built_directly_from_hf_config_object() -> None:
    hf_cfg = AdapterConfig(
        vocab_size=256,
        dim=128,
        n_layers=2,
        n_heads=4,
        head_dim=32,
        num_key_value_heads=2,
        rope_head_dim=16,
        ffn_hidden=384,
        max_seq_len=64,
    )

    sophia_cfg = SophiaDecoderConfig.from_object(hf_cfg)

    assert isinstance(sophia_cfg, SophiaDecoderConfig)
    assert int(sophia_cfg.vocab_size) == 256
    assert int(sophia_cfg.max_seq_len) == 64
    assert int(sophia_cfg.ffn_hidden) == 384


def test_model_config_rejects_hf_field_aliases_in_mapping_inputs() -> None:
    with pytest.raises(ValueError, match="canonical Sophia model fields"):
        SophiaModelConfig.from_mapping(
            {
                "hidden_size": 128,
                "num_hidden_layers": 2,
            }
        )
