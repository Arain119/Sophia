from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.tooling.scripts.machine_recipes import check_pretrain_machine_recipe as mod


def test_flatten_for_frontier_prefers_last_val_loss_over_best_val_loss() -> None:
    flat = mod._flatten_for_frontier(
        {
            "name": "release_pretrain",
            "status": "ok",
            "metrics": {
                "best_val_loss": 1.0,
                "last_val_loss": 1.5,
                "test_loss": 2.0,
            },
            "profile": {"tokens_per_sec_est": 100.0},
            "recipe": {"max_memory_gb": 8.0},
            "release_semantics": {
                "target_tokens_per_update": 8192,
                "learning_rate": 1e-4,
            },
        }
    )

    assert float(flat["val_loss"]) == 1.5
    assert float(flat["best_val_loss"]) == 1.0
    assert float(flat["test_loss"]) == 2.0


def test_sweep_parser_defaults_match_ml_repo_layout() -> None:
    parsed = mod._build_parser().parse_args([])

    assert str(parsed.data_path) == "dataset/pretrain_tokens"
    assert str(parsed.output_dir) == "out/pretrain_machine_recipe"


def test_config_builds_release_run_config_without_mutating_base() -> None:
    config = mod.PretrainMachineValidationConfig.from_namespace(
        argparse.Namespace(
            data_path="dataset/pretrain_tokens",
            tokenizer_path="tok",
            output_dir="out/sweep",
            device="cpu",
            seed=7,
            overwrite_output_dir=1,
            validation_total_tokens=0,
            target_tokens_per_update=0,
            batch_size=1,
            accumulation_steps=64,
        )
    )
    profile = mod.RELEASE_PROFILE
    base = config.build_release_run_config(profile=profile)
    original_output_dir = str(base.output_dir)
    original_total_tokens = int(base.total_tokens)

    run_config = config.run_config(
        base_run_config=base,
        run_dir=Path("out/sweep/release_pretrain"),
    )

    assert str(base.output_dir) == str(original_output_dir)
    assert int(base.total_tokens) == int(original_total_tokens)
    assert str(run_config.output_dir) == "out/sweep/release_pretrain"
    assert not hasattr(run_config, "machine_recipe_accumulation_candidates")
    assert not hasattr(run_config, "machine_recipe_loss_chunk_candidates")
    assert not hasattr(run_config, "machine_recipe_checkpoint_candidates")
    assert int(base.save_best) == 0
    assert int(base.enable_checkpoints) == 0
    assert int(base.batch_size) == 1
    assert int(base.accumulation_steps) == 64
    assert int(base.target_tokens_per_microbatch) == 4096


def test_config_can_build_isolated_target_token_candidate() -> None:
    config = mod.PretrainMachineValidationConfig.from_namespace(
        argparse.Namespace(
            data_path="dataset/pretrain_tokens",
            tokenizer_path="tok",
            output_dir="out/sweep",
            device="cpu",
            seed=7,
            overwrite_output_dir=1,
            validation_total_tokens=0,
            target_tokens_per_update=524_288,
            batch_size=1,
            accumulation_steps=128,
        )
    )
    profile = mod.RELEASE_PROFILE
    run_config = config.build_release_run_config(profile=profile)

    assert int(run_config.target_tokens_per_update) == 524_288
    assert (
        str(config.run_name(base_run_config=run_config))
        == "release_pretrain_tpu524288_eager"
    )
    assert float(run_config.learning_rate) > 0.0
    assert int(run_config.enable_checkpoints) == 0
    assert int(run_config.batch_size) == 1
    assert int(run_config.accumulation_steps) == 128


def test_sweep_can_override_validation_total_tokens(monkeypatch, tmp_path) -> None:
    seen: dict[str, object] = {}

    class _FakePipeline:
        def __init__(self, args):
            seen["total_tokens"] = int(args.total_tokens)
            self.output_dir = str(args.output_dir)

        def run(self) -> None:
            out = tmp_path / "run" / "release_pretrain"
            out.mkdir(parents=True, exist_ok=True)
            (out / "machine_recipe_summary.json").write_text(
                '{"selected":{"seq_len":128,"batch_size":1,"accumulation_steps":1,"gradient_checkpointing":0,"loss_chunk_size":0,"dataloader_num_workers":0,"dataloader_prefetch_factor":0,"dataloader_persistent_workers":0,"shard_preload":0,"shard_preload_bytes":0}}',
                encoding="utf-8",
            )
            (out / "machine_recipe.json").write_text(
                '{"best":{"batch_size":1,"accumulation_steps":1,"tokens_per_sec":1.0,"max_memory_gb":1.0,"step_time_s":1.0,"candidate":{"gradient_checkpointing":false,"gradient_checkpointing_exclude_first":0,"gradient_checkpointing_exclude_last":0,"loss_chunk_size":0}}}',
                encoding="utf-8",
            )
            (out / "machine_runtime.json").write_text(
                '{"best":{"tokens_per_sec":1.0,"step_time_s":1.0,"shard_preload":0,"shard_preload_bytes":0}}',
                encoding="utf-8",
            )
            (out / "update_profile.json").write_text(
                '{"tokens_per_sec_est":1.0,"tokens_per_update":1}',
                encoding="utf-8",
            )
            (out / "metrics.jsonl").write_text(
                '{"type":"eval","val_loss":1.0}\n',
                encoding="utf-8",
            )

    monkeypatch.setattr(mod, "PretrainPipeline", _FakePipeline)
    monkeypatch.setattr(mod, "_write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pretrain_machine_recipe.py",
            "--data_path",
            "dataset/pretrain_tokens",
            "--output_dir",
            str(tmp_path / "run"),
            "--validation_total_tokens",
            "65536",
        ],
    )
    mod.main()

    assert int(seen["total_tokens"]) == 65536


def test_sweep_candidate_target_updates_run_name_and_skips_release_recipe_export(
    monkeypatch, tmp_path
) -> None:
    seen: dict[str, object] = {}

    class _FakePipeline:
        def __init__(self, args):
            seen["output_dir"] = str(args.output_dir)
            seen["target_tokens_per_update"] = int(args.target_tokens_per_update)

        def run(self) -> None:
            out = tmp_path / "run" / "release_pretrain_tpu524288"
            out.mkdir(parents=True, exist_ok=True)
            (out / "machine_recipe_summary.json").write_text(
                '{"selected":{"seq_len":128,"batch_size":1,"accumulation_steps":1,"gradient_checkpointing":0,"loss_chunk_size":0,"dataloader_num_workers":0,"dataloader_prefetch_factor":0,"dataloader_persistent_workers":0,"shard_preload":0,"shard_preload_bytes":0}}',
                encoding="utf-8",
            )
            (out / "machine_recipe.json").write_text(
                '{"best":{"batch_size":1,"accumulation_steps":1,"tokens_per_sec":1.0,"max_memory_gb":1.0,"step_time_s":1.0,"candidate":{"gradient_checkpointing":false,"gradient_checkpointing_exclude_first":0,"gradient_checkpointing_exclude_last":0,"loss_chunk_size":0}}}',
                encoding="utf-8",
            )
            (out / "machine_runtime.json").write_text(
                '{"best":{"tokens_per_sec":1.0,"step_time_s":1.0,"shard_preload":0,"shard_preload_bytes":0}}',
                encoding="utf-8",
            )
            (out / "update_profile.json").write_text(
                '{"tokens_per_sec_est":1.0,"tokens_per_update":1}',
                encoding="utf-8",
            )
            (out / "metrics.jsonl").write_text(
                '{"type":"eval","val_loss":1.0}\n',
                encoding="utf-8",
            )

    monkeypatch.setattr(mod, "PretrainPipeline", _FakePipeline)
    monkeypatch.setattr(
        mod,
        "_recipe_payload",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("candidate target must not export release recipe")
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pretrain_machine_recipe.py",
            "--data_path",
            "dataset/pretrain_tokens",
            "--output_dir",
            str(tmp_path / "run"),
            "--target_tokens_per_update",
            "524288",
        ],
    )
    mod.main()

    assert int(seen["target_tokens_per_update"]) == 524288
    assert str(seen["output_dir"]).endswith("release_pretrain_tpu524288")
    assert not (tmp_path / "run" / "release_pretrain_machine_recipe.json").exists()


def test_sweep_defaults_to_quality_probe_token_budget(monkeypatch, tmp_path) -> None:
    seen: dict[str, object] = {}

    class _FakePipeline:
        def __init__(self, args):
            seen["total_tokens"] = int(args.total_tokens)

        def run(self) -> None:
            out = tmp_path / "run" / "release_pretrain"
            out.mkdir(parents=True, exist_ok=True)
            (out / "machine_recipe_summary.json").write_text(
                '{"selected":{"seq_len":128,"batch_size":1,"accumulation_steps":1,"gradient_checkpointing":0,"loss_chunk_size":0,"dataloader_num_workers":0,"dataloader_prefetch_factor":0,"dataloader_persistent_workers":0,"shard_preload":0,"shard_preload_bytes":0}}',
                encoding="utf-8",
            )
            (out / "machine_recipe.json").write_text(
                '{"best":{"batch_size":1,"accumulation_steps":1,"tokens_per_sec":1.0,"max_memory_gb":1.0,"step_time_s":1.0,"candidate":{"gradient_checkpointing":false,"gradient_checkpointing_exclude_first":0,"gradient_checkpointing_exclude_last":0,"loss_chunk_size":0}}}',
                encoding="utf-8",
            )
            (out / "machine_runtime.json").write_text(
                '{"best":{"tokens_per_sec":1.0,"step_time_s":1.0,"shard_preload":0,"shard_preload_bytes":0}}',
                encoding="utf-8",
            )
            (out / "update_profile.json").write_text(
                '{"tokens_per_sec_est":1.0,"tokens_per_update":1}',
                encoding="utf-8",
            )
            (out / "metrics.jsonl").write_text(
                '{"type":"eval","val_loss":1.0}\n',
                encoding="utf-8",
            )

    monkeypatch.setattr(mod, "PretrainPipeline", _FakePipeline)
    monkeypatch.setattr(mod, "_write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pretrain_machine_recipe.py",
            "--data_path",
            "dataset/pretrain_tokens",
            "--output_dir",
            str(tmp_path / "run"),
        ],
    )
    mod.main()

    assert int(seen["total_tokens"]) == 65536


def test_run_summary_marks_missing_objective_metrics_incomplete(tmp_path) -> None:
    run_dir = tmp_path / "release_pretrain"
    run_dir.mkdir(parents=True)
    (run_dir / "machine_recipe_summary.json").write_text(
        '{"selected":{"seq_len":128,"batch_size":1,"accumulation_steps":1,"gradient_checkpointing":0,"loss_chunk_size":0,"dataloader_num_workers":0,"dataloader_prefetch_factor":0,"dataloader_persistent_workers":0,"shard_preload":0,"shard_preload_bytes":0}}',
        encoding="utf-8",
    )
    (run_dir / "machine_recipe.json").write_text(
        '{"best":{"batch_size":1,"accumulation_steps":1,"tokens_per_sec":1.0,"max_memory_gb":1.0,"step_time_s":1.0,"candidate":{"gradient_checkpointing":false,"gradient_checkpointing_exclude_first":0,"gradient_checkpointing_exclude_last":0,"loss_chunk_size":0}}}',
        encoding="utf-8",
    )
    (run_dir / "machine_runtime.json").write_text(
        '{"best":{"tokens_per_sec":1.0,"step_time_s":1.0,"shard_preload":0,"shard_preload_bytes":0}}',
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")

    summary = mod._run_summary(
        name="release_pretrain",
        semantics={
            "target_tokens_per_update": 8192,
            "learning_rate": 1e-4,
            "weight_decay": 0.05,
            "warmup_steps": 0,
            "warmup_ratio": 0.1,
            "min_lr_ratio": 0.1,
            "wsd_stable_ratio": 0.9,
            "max_grad_norm": 1.0,
            "lr_schedule": "wsd",
        },
        run_dir=run_dir,
        wall_time_s=1.0,
        error="",
    )

    assert str(summary["status"]) == "incomplete"
    assert str(summary["error"]) == "missing_or_non_finite_objective_metrics"


def test_machine_baseline_row_contains_release_gate_fields() -> None:
    run_config = mod.build_pretrain_args(
        data_path="dataset/pretrain_tokens",
        output_dir="out/release_pretrain",
    )
    row = mod._machine_baseline_row(
        summary={
            "name": "release_pretrain",
            "release_semantics": {
                "backend": "inductor",
            },
            "machine_adaptive_selected": {
                "seq_len": 4096,
                "batch_size": 2,
                "accumulation_steps": 4,
                "gradient_checkpointing": 1,
                "loss_chunk_size": 256,
            },
            "recipe": {
                "tokens_per_sec": 1000.0,
                "step_time_s": 1.5,
                "max_memory_gb": 20.0,
            },
            "runtime": {},
        },
        run_config=run_config,
    )

    assert row["seq_len"] == 4096
    assert row["backend"] == "inductor"
    assert row["batch_size"] == 2
    assert row["accumulation_steps"] == 4


def test_main_writes_release_pretrain_machine_recipe_when_artifacts_exist(
    monkeypatch, tmp_path
) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tokenizer_dir / "chat_template.jinja").write_text("", encoding="utf-8")

    class _FakePipeline:
        def __init__(self, args):
            self.args = args

        def run(self) -> None:
            out = Path(self.args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "machine_recipe_summary.json").write_text(
                '{"selected":{"seq_len":128,"batch_size":1,"accumulation_steps":1,"gradient_checkpointing":0,"loss_chunk_size":0,"dataloader_num_workers":0,"dataloader_prefetch_factor":0,"dataloader_persistent_workers":0,"shard_preload":0,"shard_preload_bytes":0}}',
                encoding="utf-8",
            )
            (out / "machine_recipe.json").write_text(
                '{"best":{"batch_size":1,"accumulation_steps":1,"tokens_per_sec":1.0,"max_memory_gb":1.0,"step_time_s":1.0,"candidate":{"gradient_checkpointing":false,"gradient_checkpointing_exclude_first":0,"gradient_checkpointing_exclude_last":0,"loss_chunk_size":0}}}',
                encoding="utf-8",
            )
            (out / "machine_runtime.json").write_text(
                '{"best":{"tokens_per_sec":1.0,"step_time_s":1.0,"shard_preload":0,"shard_preload_bytes":0,"step_execution_backend":"inductor"}}',
                encoding="utf-8",
            )
            (out / "update_profile.json").write_text(
                '{"tokens_per_sec_est":1.0,"tokens_per_update":1}',
                encoding="utf-8",
            )
            (out / "metrics.jsonl").write_text(
                '{"type":"eval","val_loss":1.0}\n',
                encoding="utf-8",
            )
            (out / "run_args.json").write_text(
                json.dumps(
                    {
                        "data_path": str(tmp_path / "dataset" / "train"),
                        "eval_data_path": str(tmp_path / "dataset" / "val"),
                        "test_data_path": str(tmp_path / "dataset" / "test"),
                        "tokenizer_path": str(tokenizer_dir),
                        "seq_len": 4096,
                        "total_tokens": 65536,
                        "batch_size": 1,
                        "accumulation_steps": 64,
                        "gradient_checkpointing": 0,
                        "gradient_checkpointing_exclude_first": 0,
                        "gradient_checkpointing_exclude_last": 0,
                        "loss_chunk_size": 0,
                        "target_tokens_per_update": 270336,
                        "learning_rate": 1e-4,
                        "weight_decay": 0.05,
                        "warmup_steps": 800,
                        "warmup_ratio": 0.0,
                        "min_lr_ratio": 0.1,
                        "wsd_stable_ratio": 0.9,
                        "max_grad_norm": 1.0,
                        "lr_schedule": "wsd",
                        "dataloader_num_workers": 0,
                        "dataloader_prefetch_factor": 0,
                        "dataloader_persistent_workers": 0,
                        "shard_preload": 1,
                        "shard_preload_bytes": 4194304,
                        "_sophia_train_manifest_sha1": "train_sha1",
                        "_sophia_val_manifest_sha1": "val_sha1",
                        "_sophia_test_manifest_sha1": "test_sha1",
                        "_sophia_machine_signature": {
                            "schema": "pretrain_machine_adaptive_v1",
                            "device_type": "cuda",
                            "torch": "2.8.0",
                            "cuda": "12.8",
                            "platform": "linux",
                            "machine": "x86_64",
                                "cuda_device_name": "NVIDIA GeForce RTX 5090",
                                "cuda_capability": [12, 0],
                                "cuda_total_memory_gb": 31.356689,
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(mod, "PretrainPipeline", _FakePipeline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pretrain_machine_recipe.py",
            "--data_path",
            str(tmp_path / "dataset" / "train"),
            "--tokenizer_path",
            str(tokenizer_dir),
            "--output_dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
        ],
    )

    mod.main()

    recipe_path = tmp_path / "run" / "release_pretrain_machine_recipe.json"
    assert recipe_path.is_file()
    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert payload["kind"] == mod.PRETRAIN_MACHINE_RECIPE_KIND
    assert payload["release_semantics"]["backend"] == "inductor"
    assert payload["machine_runtime"]["dataloader_num_workers"] == 0
    assert payload["machine_runtime"]["dataloader_prefetch_factor"] == 0
    assert payload["machine_runtime"]["dataloader_persistent_workers"] == 0
    assert payload["machine_runtime"]["shard_preload"] == 1
    assert payload["machine_runtime"]["shard_preload_bytes"] == 4194304
