from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ml.core.spec import ModelSpec
from ml.integrations.adapters.hf.config import SophiaConfig as AdapterConfig
from ml.integrations.adapters.hf.config import build_config
from ml.modeling.config import SophiaModelConfig
from ml.modeling.sophia_decoder import SophiaDecoderConfig


def _tiny_values() -> dict[str, object]:
    return {
        "vocab_size": 256,
        "dim": 64,
        "n_layers": 4,
        "num_heads": 2,
        "head_dim": 32,
        "ffn_hidden": 128,
        "kda_decay_rank": 16,
        "kda_output_gate_rank": 16,
        "mla_q_rank": 16,
        "mla_kv_rank": 16,
        "max_seq_len": 64,
    }


def test_default_model_spec_is_native_hybrid_1b() -> None:
    spec = ModelSpec.default()

    assert spec.vocab_size == 65536
    assert spec.dim == 1536
    assert spec.n_layers == 28
    assert spec.num_heads == 16
    assert spec.head_dim == 128
    assert spec.ffn_hidden == 3968
    assert spec.kda_decay_rank == 128
    assert spec.kda_output_gate_rank == 128
    assert spec.kda_output_gate_full_rank is True
    assert spec.kda_decay_lower_bound == pytest.approx(-5.0)
    assert spec.kda_dt_min == pytest.approx(1e-3)
    assert spec.kda_dt_max == pytest.approx(1e-1)
    assert spec.kda_dt_floor == pytest.approx(1e-4)
    assert spec.kda_a_log_init == pytest.approx(0.0)
    assert spec.mla_q_rank == 384
    assert spec.mla_kv_rank == 128
    assert spec.short_conv_kernel == 4
    assert spec.attn_res_block_size == 4
    assert spec.situ_gate_softcap == pytest.approx(4.0)
    assert spec.situ_up_softcap == pytest.approx(25.0)
    assert spec.max_seq_len == 4096
    assert spec.initializer_range == pytest.approx(0.02)


def test_specs_are_frozen() -> None:
    spec = ModelSpec.default()
    with pytest.raises(FrozenInstanceError):
        spec.dim = 2048  # type: ignore[misc]


def test_model_spec_roundtrips_through_config_and_runtime_args() -> None:
    spec = ModelSpec.from_mapping(_tiny_values())

    cfg = spec.to_config()
    args = spec.to_model_args(runtime_max_seq_len=32)

    assert cfg.ffn_hidden == 128
    assert args.ffn_hidden == 128
    assert args.max_seq_len == 32
    assert [args.layer_type(index) for index in range(4)] == [
        "kda",
        "kda",
        "kda",
        "mla",
    ]
    assert cfg.to_model_args(runtime_max_seq_len=32) == args


def test_model_spec_rejects_unknown_and_removed_fields() -> None:
    with pytest.raises(ValueError, match="unknown override keys"):
        ModelSpec.default().with_overrides(unknown_runtime_field=2)
    with pytest.raises(ValueError, match="unknown keys: n_heads"):
        ModelSpec.from_mapping({"n_heads": 4})
    with pytest.raises(ValueError, match="unknown keys: rope_scaling"):
        ModelSpec.from_mapping({"rope_scaling": {"rope_type": "linear"}})


def test_model_spec_validates_native_hybrid_ranges() -> None:
    with pytest.raises(ValueError, match="kda_decay_lower_bound"):
        ModelSpec.default().with_overrides(kda_decay_lower_bound=-6.0)
    with pytest.raises(ValueError, match="kda_dt_min"):
        ModelSpec.default().with_overrides(kda_dt_min=0.0)
    with pytest.raises(ValueError, match="kda_dt_min must be <="):
        ModelSpec.default().with_overrides(kda_dt_min=0.2)
    with pytest.raises(ValueError, match="short_conv_kernel"):
        ModelSpec.default().with_overrides(short_conv_kernel=0)
    with pytest.raises(ValueError, match="initializer_range"):
        ModelSpec.default().with_overrides(initializer_range=0.0)
    with pytest.raises(ValueError, match="attn_res_block_size"):
        ModelSpec.default().with_overrides(attn_res_block_size=0)
    with pytest.raises(ValueError, match="situ_gate_softcap"):
        ModelSpec.default().with_overrides(situ_gate_softcap=0.0)
    with pytest.raises(ValueError, match="kda_backend"):
        ModelSpec.default().with_overrides(kda_backend="unknown")
    assert ModelSpec.default().with_overrides(head_dim=127).head_dim == 127


def test_canonical_config_projects_to_hf_only_via_adapter() -> None:
    cfg = SophiaModelConfig(**_tiny_values())

    assert not hasattr(cfg, "to_hf_config")

    class _DummyHFConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    projected = build_config(cfg, config_cls=_DummyHFConfig)
    assert projected.kwargs["num_heads"] == 2
    assert projected.kwargs["kda_decay_rank"] == 16


def test_hf_adapter_accepts_native_fields_and_rejects_legacy_fields() -> None:
    hf_cfg = AdapterConfig(**_tiny_values())
    sophia_cfg = SophiaDecoderConfig.from_object(hf_cfg)

    assert sophia_cfg.num_heads == 2
    assert sophia_cfg.mla_q_rank == 16
    with pytest.raises(ValueError, match="legacy Sophia Transformer fields"):
        AdapterConfig(n_heads=2)
    with pytest.raises(ValueError, match="legacy Sophia Transformer fields"):
        AdapterConfig(rope_head_dim=16)
    with pytest.raises(ValueError, match="legacy Sophia Transformer fields"):
        AdapterConfig(rope_theta=10000.0)


def test_decoder_config_from_mapping_preserves_runtime_fields() -> None:
    config = SophiaDecoderConfig.from_mapping(
        {
            "gradient_checkpointing_exclude_first": 4,
            "gradient_checkpointing_exclude_last": 5,
            "loss_chunk_size": 256,
            "pad_token_id": 7,
        }
    )

    assert config.gradient_checkpointing_exclude_first == 4
    assert config.gradient_checkpointing_exclude_last == 5
    assert config.loss_chunk_size == 256
    assert config.pad_token_id == 7


def test_model_config_rejects_hf_aliases_in_mapping_inputs() -> None:
    with pytest.raises(ValueError, match="native Sophia Hybrid fields"):
        SophiaModelConfig.from_mapping(
            {"hidden_size": 128, "num_hidden_layers": 2}
        )
