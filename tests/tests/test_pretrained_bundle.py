from __future__ import annotations

import json

import pytest
import torch

from ml.modeling.pretrained_bundle import (
    load_canonical_config_from_pretrained,
    load_pretrained_state_dict,
    save_pretrained_state_dict,
)
from ml.modeling.sophia_decoder import SophiaDecoderConfig


def test_safe_state_dict_export_preserves_tied_keys_without_shared_storage(
    tmp_path,
) -> None:
    tied = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    state = {
        "model.tok_embeddings.weight": tied,
        "model.output.weight": tied,
    }

    save_pretrained_state_dict(
        state,
        save_directory=tmp_path,
        safe_serialization=True,
    )
    loaded = load_pretrained_state_dict(tmp_path)

    assert loaded is not None
    assert set(loaded) == set(state)
    torch.testing.assert_close(
        loaded["model.tok_embeddings.weight"],
        loaded["model.output.weight"],
    )
    assert (
        loaded["model.tok_embeddings.weight"].untyped_storage().data_ptr()
        != loaded["model.output.weight"].untyped_storage().data_ptr()
    )


def test_load_pretrained_state_dict_rejects_missing_weights(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="no supported model weights"):
        load_pretrained_state_dict(tmp_path)


def test_exported_config_rejects_unknown_fields(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"dim": 128, "n_layerz": 2}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown keys: n_layerz"):
        load_canonical_config_from_pretrained(
            tmp_path,
            config_cls=SophiaDecoderConfig,
        )


def _hf_export_config(**overrides: object) -> dict[str, object]:
    config = SophiaDecoderConfig(
        vocab_size=128,
        dim=64,
        n_layers=4,
        num_heads=2,
        head_dim=32,
        ffn_hidden=128,
        kda_decay_rank=16,
        kda_output_gate_rank=16,
        mla_q_rank=16,
        mla_kv_rank=16,
        max_seq_len=128,
    ).to_dict()
    config.update(
        {
            "architectures": ["SophiaForCausalLM"],
            "attention_dropout": config["dropout"],
            "auto_map": {
                "AutoConfig": "modeling_sophia.SophiaConfig",
                "AutoModelForCausalLM": "modeling_sophia.SophiaForCausalLM",
            },
            "hidden_size": config["dim"],
            "max_position_embeddings": config["max_seq_len"],
            "model_type": "sophia_hybrid",
            "num_attention_heads": config["num_heads"],
            "num_hidden_layers": config["n_layers"],
            "rms_norm_eps": config["norm_eps"],
            "sliding_window": config["max_seq_len"],
            "transformers_version": "5.10.2",
        }
    )
    config.update(overrides)
    return config


def test_exported_hf_config_loads_after_strict_metadata_validation(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(_hf_export_config()),
        encoding="utf-8",
    )

    config = load_canonical_config_from_pretrained(
        tmp_path,
        config_cls=SophiaDecoderConfig,
    )

    assert config.dim == 64
    assert config.n_layers == 4
    assert config.max_seq_len == 128


def test_exported_hf_config_rejects_alias_mismatch(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(_hf_export_config(hidden_size=65)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hidden_size != dim"):
        load_canonical_config_from_pretrained(
            tmp_path,
            config_cls=SophiaDecoderConfig,
        )
