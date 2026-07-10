from __future__ import annotations

import argparse

import pytest
import torch

from ml.core.engine import RunContext, RunSpec, RunState
from ml.core.engine.machine_adaptive import (
    machine_signature_payload,
    ensure_machine_adaptive_signature,
    load_json_object_from_path,
    resume_machine_signature,
)
from ml.modeling import SophiaModelConfig
from ml.modeling.sophia_decoder import SophiaDecoder, SophiaDecoderConfig
from ml.core.spec import ModelSpec
from ml.training.pretrain.model_runtime_controls import (
    sanitize_runtime_recipe_knobs,
)


def test_model_spec_roundtrip_matches_canonical_config() -> None:
    cfg = SophiaModelConfig()
    spec = ModelSpec.from_config(cfg)

    assert spec == ModelSpec.default()
    assert spec.to_config().to_dict() == cfg.to_dict()


def test_model_spec_from_mapping_resolves_defaults_and_sequences() -> None:
    spec = ModelSpec.from_mapping(
        {
            "vocab_size": 256,
            "dim": 128,
            "n_layers": 2,
            "n_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "rope_head_dim": 16,
            "max_seq_len": 64,
            "ffn_hidden": 384,
        }
    )

    assert int(spec.to_config().ffn_hidden) == 384


def test_run_spec_normalizes_fields() -> None:
    spec = RunSpec(
        run_kind=" SFT ",
        model=ModelSpec.default(),
        seed=123,
        max_steps=10,
        output_dir=" out/run ",
        resume_from_checkpoint="  ",
    )

    assert spec.run_kind == "sft"
    assert spec.output_dir == "out/run"
    assert spec.resume_from_checkpoint is None


def test_run_context_normalizes_runtime_metadata() -> None:
    ctx = RunContext(
        output_dir="out/test",
        device=torch.device("cpu"),
        base_dtype=torch.float32,
        runtime_metadata={"torch": "test"},
        machine_signature={"device": "cpu"},
    )

    assert ctx.output_dir == "out/test"
    assert ctx.resolved_device == "cpu"
    assert ctx.runtime_metadata == {"torch": "test"}
    assert ctx.machine_signature == {"device": "cpu"}


def test_run_state_copies_mutable_payloads() -> None:
    best_metrics = {"loss": 1.25}
    state = RunState(
        step=3,
        seen_tokens=128,
        train_state={"phase": "warmup"},
        best_metrics=best_metrics,
    )
    best_metrics["loss"] = 99.0

    assert state.step == 3
    assert state.seen_tokens == 128
    assert state.train_state == {"phase": "warmup"}
    assert state.best_metrics == {"loss": 1.25}


def test_resume_machine_signature_rejects_mismatched_explicit_schema() -> None:
    signature = resume_machine_signature(
        resume_args={"sig": {"schema": "invalid_v0", "device": "cpu"}},
        arg_key="sig",
        schema="current_v1",
    )

    assert signature is None


def test_ensure_machine_adaptive_signature_replaces_incompatible_schema() -> None:
    args = argparse.Namespace(sig={"schema": "invalid_v0", "device": "cpu"})

    signature = ensure_machine_adaptive_signature(
        args=args,
        device=torch.device("cpu"),
        arg_key="sig",
        schema="current_v1",
    )

    assert machine_signature_payload(signature)["schema"] == "current_v1"
    assert dict(args.sig) == machine_signature_payload(signature)


def test_load_json_object_from_path_skips_blank_recipe_path() -> None:
    assert (
        load_json_object_from_path(
            path_label="semantic_recipe",
            recipe_path="   ",
        )
        is None
    )


def test_runtime_recipe_controls_reject_non_runtime_model() -> None:
    with pytest.raises(TypeError, match="SupportsRuntimeRecipeControl"):
        sanitize_runtime_recipe_knobs(
            torch.nn.Linear(4, 4),
            loss_chunk_size=256,
            gradient_checkpointing_exclude_first=1,
            gradient_checkpointing_exclude_last=1,
        )


def test_runtime_recipe_knob_sanitization_follows_runtime_capabilities() -> None:
    model = SophiaDecoder(
        SophiaDecoderConfig(
            vocab_size=128,
            dim=64,
            n_layers=2,
            n_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            rope_head_dim=16,
            max_seq_len=64,
            ffn_hidden=128,
        )
    )

    loss_chunk_size, exclude_first, exclude_last = sanitize_runtime_recipe_knobs(
        model,
        loss_chunk_size=256,
        gradient_checkpointing_exclude_first=1,
        gradient_checkpointing_exclude_last=1,
    )

    assert loss_chunk_size == 256
    assert exclude_first == 1
    assert exclude_last == 1


def test_sophia_decoder_config_exposes_z_loss_default() -> None:
    cfg = SophiaDecoderConfig()

    assert float(cfg.z_loss_weight) == 1e-4


def test_sophia_decoder_config_ties_word_embeddings_by_default() -> None:
    cfg = SophiaDecoderConfig()

    assert bool(cfg.tie_word_embeddings) is True
