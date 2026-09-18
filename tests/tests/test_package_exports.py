from __future__ import annotations

import importlib

import ml
from ml import data, runtime
from ml.core import engine
from ml.modeling import ModelSpec, SophiaModelConfig, create_model_from_spec
from ml.data.token_shards import (
    TokenShard,
    TokenShardManifest,
    TokenStreamDataset,
    compute_tokenizer_bundle_sha1,
)
from ml.core.engine import EngineCheckpoint as SharedEngineCheckpoint
from ml.core.engine import RunContext, RunSpec
from ml.training.pretrain.engine import StepRunner, train_loop


def test_top_level_packages_expose_stable_subpackages() -> None:
    assert ml.__version__ == "0.1.0"
    assert importlib.import_module("ml.integrations.adapters") is not None
    assert data.token_shards is not None
    assert importlib.import_module("ml.core.engine") is not None
    assert importlib.import_module("ml.runtime") is not None
    assert importlib.import_module("ml.tasks") is not None
    assert importlib.import_module("ml.core.spec.semantics") is not None
    assert importlib.import_module("ml.training.pretrain") is not None
    assert runtime is not None
    assert importlib.import_module("ml.runtime.inference") is not None
    assert callable(runtime.resolve_runtime_host)
    assert callable(runtime.resolve_runtime_lock)


def test_modeling_package_exposes_core_model_api() -> None:
    spec = ModelSpec.default()

    assert SophiaModelConfig is not None
    assert create_model_from_spec is not None
    assert ModelSpec.default() == spec.to_config().to_model_spec()
    assert spec.to_config() == SophiaModelConfig().to_model_spec().to_config()
    assert not hasattr(importlib.import_module("ml.modeling"), "SophiaConfig")
    assert not hasattr(importlib.import_module("ml.modeling"), "SophiaForCausalLM")


def test_engine_package_exports_stable_symbols() -> None:
    assert engine is not None
    assert SharedEngineCheckpoint is not None
    assert RunSpec is not None
    assert RunContext is not None


def test_token_shards_package_exports_stable_symbols() -> None:
    assert TokenShard is not None
    assert TokenShardManifest is not None
    assert TokenStreamDataset is not None
    assert callable(compute_tokenizer_bundle_sha1)


def test_training_package_exports_stable_entrypoints() -> None:
    assert callable(importlib.import_module("ml.cli.train").main)
    assert callable(importlib.import_module("ml.cli.eval").main)
    assert callable(importlib.import_module("ml.cli.pretrain_check").main)
    assert callable(importlib.import_module("ml.cli.shard_builder").main)
    assert callable(importlib.import_module("ml.tasks.pretrain.pipeline").run)
    assert callable(
        importlib.import_module("ml.tasks.pretrain.validate").build_validation_args
    )


def test_pretrain_engine_package_exports_stable_symbols() -> None:
    assert StepRunner is not None
    assert callable(train_loop)
