"""Canonical guard for the Sophia model API."""

from __future__ import annotations

import importlib.util
from dataclasses import fields

import pytest

from ml.modeling.config import SophiaModelConfig
from ml.runtime.model.config import ModelArgs
from ml.core.spec import (
    ModelSpec,
    build_release_pretrain_schedule_spec,
)
from ml.training.pretrain.profiles import RELEASE_PROFILE

_SKIP_SHARED_FIELDS: set[str] = set()


def _normalize(value: object) -> object:
    return list(value) if isinstance(value, (list, tuple)) else value


def test_training_pretrain_config_module_is_absent() -> None:
    assert importlib.util.find_spec("ml.training.pretrain.config") is None


def test_release_pretrain_profile_owns_model_structure_and_schedule_defaults() -> None:
    profile = RELEASE_PROFILE
    schedule = build_release_pretrain_schedule_spec(model=profile.model)

    default_model = ModelSpec.default()
    assert profile.model == default_model
    assert int(profile.training.max_seq_len) == int(profile.model.max_seq_len)
    assert int(profile.training.train_seq_len) == int(schedule.train_seq_len)


def test_runtime_modelargs_match_canonical_dataclass() -> None:
    cfg = SophiaModelConfig()
    args = ModelArgs()
    cfg_fields = {f.name for f in fields(cfg)}
    arg_fields = {f.name for f in fields(args)}
    shared = (cfg_fields & arg_fields) - _SKIP_SHARED_FIELDS
    assert shared, "expected overlapping fields between SophiaModelConfig and ModelArgs"
    mismatches = []
    for name in sorted(shared):
        cfg_value = _normalize(getattr(cfg, name))
        arg_value = _normalize(getattr(args, name))
        if cfg_value != arg_value:
            mismatches.append(
                f"{name}: SophiaModelConfig={cfg_value!r} != ModelArgs={arg_value!r}"
            )
    assert not mismatches, (
        "runtime ModelArgs drifted from SophiaModelConfig:\n" + "\n".join(mismatches)
    )


def test_to_model_args_forwards_every_shared_field() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed in this environment")
    from ml.integrations.adapters.hf.model import SophiaConfig as HFSophiaConfig

    cfg = HFSophiaConfig(
        max_batch_size=2,
        max_seq_len=64,
        vocab_size=256,
        dim=128,
        n_layers=4,
        num_heads=4,
        head_dim=32,
        ffn_hidden=320,
        kda_decay_rank=16,
        kda_output_gate_rank=16,
        mla_q_rank=16,
        mla_kv_rank=16,
        dropout=0.1,
    )
    args = cfg.to_model_args()

    arg_fields = {f.name for f in fields(args)}
    runtime_only = arg_fields - {f.name for f in fields(SophiaModelConfig())}

    mismatches = []
    for name in sorted(arg_fields - runtime_only):
        if not hasattr(cfg, name):
            continue
        cfg_value = _normalize(getattr(cfg, name))
        arg_value = _normalize(getattr(args, name))
        if cfg_value != arg_value:
            mismatches.append(f"{name}: HF config={cfg_value!r} != ModelArgs={arg_value!r}")
    assert not mismatches, "to_model_args dropped/altered shared fields:\n" + "\n".join(mismatches)

    defaults = ModelArgs()
    for name in runtime_only:
        assert getattr(args, name) == getattr(defaults, name), (
            f"runtime-only field {name} was clobbered by to_model_args"
        )
