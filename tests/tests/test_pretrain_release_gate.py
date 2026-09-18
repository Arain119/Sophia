from __future__ import annotations

import json
import pytest

from ml.errors import SophiaUsageError
from ml.tasks.pretrain.pipeline import build_pretrain_args
from ml.training.pretrain.profiles import RELEASE_PROFILE
from ml.training.pretrain.release_config import (
    DEFAULT_PRETRAIN_MACHINE_RECIPE,
    DEFAULT_PRETRAIN_MACHINE_RUNTIME,
    PretrainReleaseSemantics,
)
from ml.training.pretrain.implementation_fingerprint import (
    current_pretrain_implementation_sha256,
)
from ml.training.pretrain.release_gate import (
    PRETRAIN_MACHINE_RECIPE_KIND,
    build_pretrain_data_signature,
    run_pretrain_release_preflight,
)
from ml.training.pretrain.runtime_bootstrap import PretrainRuntimeBootstrap


def _write_tokenizer_bundle(path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "chat_template.jinja").write_text("", encoding="utf-8")


def _write_manifest(path, *, tokenizer_sha1: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dtype": "int32",
                "eos_token_id": 2,
                "tokenizer_sha1": tokenizer_sha1,
                "total_tokens": 1024,
                "shards": [{"path": "shard_00000.bin", "tokens": 1024}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (path.parent / "shard_00000.bin").write_bytes(b"\x00\x00\x00\x00")


def _bootstrap(tmp_path) -> tuple[object, PretrainRuntimeBootstrap]:
    tokenizer_dir = tmp_path / "tokenizer"
    _write_tokenizer_bundle(tokenizer_dir)
    from ml.data.token_shards.tokenizer_fingerprint import compute_tokenizer_bundle_sha1

    tokenizer_sha1 = compute_tokenizer_bundle_sha1(str(tokenizer_dir))
    train_dir = tmp_path / "dataset" / "train"
    val_dir = tmp_path / "dataset" / "val"
    test_dir = tmp_path / "dataset" / "test"
    train_manifest = train_dir / "manifest.json"
    val_manifest = val_dir / "manifest.json"
    test_manifest = test_dir / "manifest.json"
    for path in (train_manifest, val_manifest, test_manifest):
        _write_manifest(path, tokenizer_sha1=tokenizer_sha1)
    args = build_pretrain_args(
        data_path=str(train_dir),
        output_dir=str(tmp_path / "out" / "release_pretrain"),
        machine_recipe_json=str(tmp_path / "recipe.json"),
        profile=RELEASE_PROFILE,
    )
    args.eval_data_path = str(val_dir)
    args.test_data_path = str(test_dir)
    args._sophia_run_kind = "pretrain"
    args._sophia_machine_signature = {
        "schema": "pretrain_machine_adaptive_v1",
        "device_type": "cuda",
        "torch": "2.8.0",
        "cuda": "12.8",
        "platform": "linux",
        "machine": "x86_64",
        "cuda_device_name": "NVIDIA GeForce RTX 5090",
        "cuda_capability": [12, 0],
        "cuda_total_memory_gb": 31.356689453125,
    }
    args._sophia_train_manifest_sha1 = "train_sha1"
    args._sophia_val_manifest_sha1 = "val_sha1"
    args._sophia_test_manifest_sha1 = "test_sha1"
    bootstrap = PretrainRuntimeBootstrap(
        output_dir=str(tmp_path / "out" / "release_pretrain"),
        resume_path=None,
        resume_checkpoint=None,
        resolved_resume_checkpoint="",
        data_path=str(train_dir),
        eval_data_path=str(val_dir),
        test_data_path=str(test_dir),
        manifest_path=str(train_manifest),
        manifest=object(),
        total_tokens=int(args.total_tokens),
        tokenizer=object(),
        tokenizer_path=str(tokenizer_dir),
        seq_len=int(args.seq_len),
        train_manifest_sha1="train_sha1",
        val_manifest_sha1="val_sha1",
        test_manifest_sha1="test_sha1",
    )
    return args, bootstrap


def _valid_recipe(args, bootstrap) -> dict[str, object]:
    return {
        "kind": PRETRAIN_MACHINE_RECIPE_KIND,
        "pretrain_implementation_sha256": current_pretrain_implementation_sha256(),
        "selected_name": "measured_pretrain_sophia_1b_seq4096",
        "release_semantics": PretrainReleaseSemantics.from_run_config(
            args=args
        ).to_payload(),
        "machine_signature": dict(args._sophia_machine_signature or {}),
        "data_signature": build_pretrain_data_signature(
            args=args, bootstrap=bootstrap
        ),
        "machine_recipe": dict(DEFAULT_PRETRAIN_MACHINE_RECIPE),
        "machine_runtime": dict(DEFAULT_PRETRAIN_MACHINE_RUNTIME),
        "artifacts": {
            "machine_recipe": {"best": {"batch_size": 1}},
            "machine_runtime": {"best": {"tokens_per_sec": 1.0}},
            "update_profile": {"tokens_per_update": 1_310_720},
        },
    }


def test_release_pretrain_preflight_requires_signed_machine_recipe(tmp_path) -> None:
    args, bootstrap = _bootstrap(tmp_path)

    with pytest.raises(SophiaUsageError, match="requires --machine_recipe_json"):
        run_pretrain_release_preflight(
            args=args,
            bootstrap=bootstrap,
            recipe_payload=None,
        )


def test_release_pretrain_rejects_recipe_token_batch_mismatch(tmp_path) -> None:
    args, bootstrap = _bootstrap(tmp_path)
    recipe = _valid_recipe(args, bootstrap)
    recipe["machine_recipe"]["accumulation_steps"] = 319

    with pytest.raises(SophiaUsageError, match="fixed token batch"):
        run_pretrain_release_preflight(
            args=args,
            bootstrap=bootstrap,
            recipe_payload=recipe,
        )


def test_release_pretrain_rejects_recipe_semantic_mismatch(tmp_path) -> None:
    args, bootstrap = _bootstrap(tmp_path)
    recipe = _valid_recipe(args, bootstrap)
    recipe["release_semantics"]["learning_rate"] = 0.1

    with pytest.raises(SophiaUsageError, match="release_semantics"):
        run_pretrain_release_preflight(
            args=args,
            bootstrap=bootstrap,
            recipe_payload=recipe,
        )


def test_build_pretrain_release_semantics_captures_optimizer_semantics(
    tmp_path,
) -> None:
    args, _bootstrap_state = _bootstrap(tmp_path)

    semantics = PretrainReleaseSemantics.from_run_config(args=args).to_payload()

    assert float(semantics["beta1"]) == pytest.approx(0.9)
    assert float(semantics["beta2"]) == pytest.approx(0.95)
    assert float(semantics["adam_eps"]) == pytest.approx(1e-8)
    assert str(semantics["optimizer_kind"]) == "torch_muon_hybrid"
    assert int(semantics["muon_ns_steps"]) == 5
    assert semantics["qk_clip_enabled"] is True
    assert float(semantics["qk_clip_threshold"]) == pytest.approx(100.0)
    assert "grad_clip_mode" not in semantics
    assert "agc_clip" not in semantics


def test_release_pretrain_preflight_applies_recipe_and_materializes_artifacts(
    tmp_path,
) -> None:
    args, bootstrap = _bootstrap(tmp_path)
    machine_signature = dict(args._sophia_machine_signature or {})
    recipe_payload = {
        "kind": PRETRAIN_MACHINE_RECIPE_KIND,
        "pretrain_implementation_sha256": current_pretrain_implementation_sha256(),
        "selected_name": "measured_pretrain_sophia_1b_seq4096",
        "release_semantics": PretrainReleaseSemantics.from_run_config(
            args=build_pretrain_args(
                data_path=str(args.data_path),
                output_dir=str(args.output_dir),
                machine_recipe_json=str(args.machine_recipe_json),
                profile=RELEASE_PROFILE,
            )
        ).to_payload(),
        "machine_signature": machine_signature,
        "data_signature": build_pretrain_data_signature(args=args, bootstrap=bootstrap),
        "machine_recipe": dict(DEFAULT_PRETRAIN_MACHINE_RECIPE),
        "machine_runtime": dict(DEFAULT_PRETRAIN_MACHINE_RUNTIME),
        "artifacts": {
            "machine_recipe": {"best": {"batch_size": 1}},
            "machine_runtime": {"best": {"tokens_per_sec": 123.0}},
            "update_profile": {
                "tokens_per_sec_est": 27334.2,
                "tokens_per_update": 270336,
            },
        },
    }

    resolved = run_pretrain_release_preflight(
        args=args,
        bootstrap=bootstrap,
        recipe_payload=recipe_payload,
    )

    assert int(resolved.batch_size) == 1
    assert int(resolved.accumulation_steps) == 320
    assert int(resolved.gradient_checkpointing) == 0
    assert int(resolved.gradient_checkpointing_exclude_last) == 0
    assert int(resolved.loss_chunk_size) == 0
    assert int(resolved.dataloader_num_workers) == 0
    assert int(resolved.shard_preload_bytes) == 0
    assert (tmp_path / "out" / "release_pretrain" / "machine_recipe.json").exists()
    assert (tmp_path / "out" / "release_pretrain" / "machine_runtime.json").exists()
    assert (tmp_path / "out" / "release_pretrain" / "update_profile.json").exists()


def test_release_pretrain_rejects_unmeasured_graph_shape(tmp_path) -> None:
    args, bootstrap = _bootstrap(tmp_path)
    recipe = _valid_recipe(args, bootstrap)
    recipe["machine_recipe"]["batch_size"] = 2
    recipe["machine_recipe"]["accumulation_steps"] = 160

    with pytest.raises(SophiaUsageError, match="fixed B1/320"):
        run_pretrain_release_preflight(
            args=args,
            bootstrap=bootstrap,
            recipe_payload=recipe,
        )


def test_release_pretrain_rejects_non_graph_backend(tmp_path) -> None:
    args, bootstrap = _bootstrap(tmp_path)
    recipe = _valid_recipe(args, bootstrap)
    recipe["machine_runtime"]["step_execution_backend"] = "eager"

    with pytest.raises(SophiaUsageError, match="SM120 graph backend"):
        run_pretrain_release_preflight(
            args=args,
            bootstrap=bootstrap,
            recipe_payload=recipe,
        )


def test_release_pretrain_preflight_rejects_other_cuda_machine(tmp_path) -> None:
    args, bootstrap = _bootstrap(tmp_path)
    recipe_signature = dict(args._sophia_machine_signature or {})
    current_signature = dict(args._sophia_machine_signature or {})
    current_signature["cuda_device_name"] = "NVIDIA A100-SXM4-80GB"
    current_signature["cuda_capability"] = [8, 0]
    current_signature["cuda_total_memory_gb"] = 80.0
    args._sophia_machine_signature = current_signature
    recipe_args = build_pretrain_args(
        data_path=str(args.data_path),
        output_dir=str(args.output_dir),
        machine_recipe_json=str(args.machine_recipe_json),
        profile=RELEASE_PROFILE,
    )
    recipe_args.eval_data_path = str(args.eval_data_path)
    recipe_args.test_data_path = str(args.test_data_path)
    recipe_args._sophia_run_kind = str(args._sophia_run_kind)
    recipe_args._sophia_machine_signature = dict(recipe_signature)

    recipe_payload = {
        "kind": PRETRAIN_MACHINE_RECIPE_KIND,
        "pretrain_implementation_sha256": current_pretrain_implementation_sha256(),
        "selected_name": "measured_pretrain_sophia_1b_seq4096",
        "release_semantics": PretrainReleaseSemantics.from_run_config(
            args=recipe_args
        ).to_payload(),
        "machine_signature": dict(recipe_signature),
        "data_signature": build_pretrain_data_signature(args=args, bootstrap=bootstrap),
        "machine_recipe": dict(DEFAULT_PRETRAIN_MACHINE_RECIPE),
        "machine_runtime": dict(DEFAULT_PRETRAIN_MACHINE_RUNTIME),
        "artifacts": {
            "machine_recipe": {"best": {"batch_size": 1}},
            "machine_runtime": {"best": {"tokens_per_sec": 29900.0}},
            "update_profile": {
                "tokens_per_sec_est": 29900.0,
                "tokens_per_update": 270336,
            },
        },
    }

    with pytest.raises(SophiaUsageError, match="RTX 5090"):
        run_pretrain_release_preflight(
            args=args,
            bootstrap=bootstrap,
            recipe_payload=recipe_payload,
        )


def test_release_pretrain_preflight_rejects_non_cuda_machine(tmp_path) -> None:
    args, bootstrap = _bootstrap(tmp_path)
    recipe_signature = dict(args._sophia_machine_signature or {})
    current_signature = dict(args._sophia_machine_signature or {})
    current_signature["device_type"] = "cpu"
    current_signature.pop("cuda_device_name", None)
    current_signature.pop("cuda_capability", None)
    current_signature.pop("cuda_total_memory_gb", None)
    args._sophia_machine_signature = current_signature
    recipe_args = build_pretrain_args(
        data_path=str(args.data_path),
        output_dir=str(args.output_dir),
        machine_recipe_json=str(args.machine_recipe_json),
        profile=RELEASE_PROFILE,
    )
    recipe_args.eval_data_path = str(args.eval_data_path)
    recipe_args.test_data_path = str(args.test_data_path)
    recipe_args._sophia_run_kind = str(args._sophia_run_kind)
    recipe_args._sophia_machine_signature = dict(recipe_signature)

    recipe_payload = {
        "kind": PRETRAIN_MACHINE_RECIPE_KIND,
        "pretrain_implementation_sha256": current_pretrain_implementation_sha256(),
        "selected_name": "measured_pretrain_sophia_1b_seq4096",
        "release_semantics": PretrainReleaseSemantics.from_run_config(
            args=recipe_args
        ).to_payload(),
        "machine_signature": dict(recipe_signature),
        "data_signature": build_pretrain_data_signature(args=args, bootstrap=bootstrap),
        "machine_recipe": dict(DEFAULT_PRETRAIN_MACHINE_RECIPE),
        "machine_runtime": dict(DEFAULT_PRETRAIN_MACHINE_RUNTIME),
        "artifacts": {
            "machine_recipe": {"best": {"batch_size": 1}},
            "machine_runtime": {"best": {"tokens_per_sec": 29900.0}},
            "update_profile": {
                "tokens_per_sec_est": 29900.0,
                "tokens_per_update": 270336,
            },
        },
    }

    with pytest.raises(SophiaUsageError, match="CUDA GPU"):
        run_pretrain_release_preflight(
            args=args,
            bootstrap=bootstrap,
            recipe_payload=recipe_payload,
        )


def test_release_pretrain_preflight_rejects_memory_signature_mismatch(
    tmp_path,
) -> None:
    args, bootstrap = _bootstrap(tmp_path)
    recipe_signature = dict(args._sophia_machine_signature or {})
    current_signature = dict(args._sophia_machine_signature or {})
    current_signature["cuda_total_memory_gb"] = 8.0
    args._sophia_machine_signature = current_signature
    recipe_args = build_pretrain_args(
        data_path=str(args.data_path),
        output_dir=str(args.output_dir),
        machine_recipe_json=str(args.machine_recipe_json),
        profile=RELEASE_PROFILE,
    )
    recipe_args.eval_data_path = str(args.eval_data_path)
    recipe_args.test_data_path = str(args.test_data_path)
    recipe_args._sophia_run_kind = str(args._sophia_run_kind)
    recipe_args._sophia_machine_signature = dict(recipe_signature)

    recipe_payload = {
        "kind": PRETRAIN_MACHINE_RECIPE_KIND,
        "pretrain_implementation_sha256": current_pretrain_implementation_sha256(),
        "selected_name": "measured_pretrain_sophia_1b_seq4096",
        "release_semantics": PretrainReleaseSemantics.from_run_config(
            args=recipe_args
        ).to_payload(),
        "machine_signature": dict(recipe_signature),
        "data_signature": build_pretrain_data_signature(args=args, bootstrap=bootstrap),
        "machine_recipe": dict(DEFAULT_PRETRAIN_MACHINE_RECIPE),
        "machine_runtime": dict(DEFAULT_PRETRAIN_MACHINE_RUNTIME),
        "artifacts": {
            "machine_recipe": {"best": {"batch_size": 1}},
            "machine_runtime": {"best": {"tokens_per_sec": 29900.0}},
            "update_profile": {
                "tokens_per_sec_est": 29900.0,
                "tokens_per_update": 270336,
            },
        },
    }

    with pytest.raises(SophiaUsageError, match="machine_signature"):
        run_pretrain_release_preflight(
            args=args,
            bootstrap=bootstrap,
            recipe_payload=recipe_payload,
        )
