from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from ml.tooling.scripts.machine_recipes import check_pretrain_machine_recipe as mod


def _machine_measurement_metrics() -> str:
    events = [
        {"type": "start", "time": 0.0},
        {
            "type": "train",
            "time": 100.0,
            "step": 1,
            "dt_s": 100.0,
            "updates": 1,
            "tok_s": 13_107,
            "mem_gb": 17.25,
        },
        {
            "type": "train",
            "time": 150.0,
            "step": 2,
            "dt_s": 50.0,
            "updates": 1,
            "tok_s": 26_214,
            "mem_gb": 17.0,
        },
        {
            "type": "train",
            "time": 210.0,
            "step": 3,
            "dt_s": 60.0,
            "updates": 1,
            "tok_s": 21_845,
            "mem_gb": 17.0,
        },
        {"type": "eval", "time": 211.0, "val_loss": 1.0},
        {"type": "end", "time": 211.0},
    ]
    return "".join(json.dumps(event) + "\n" for event in events)


def test_measurement_parser_exposes_only_paths_and_device() -> None:
    parsed = mod._build_parser().parse_args([])

    assert str(parsed.data_path) == "dataset/pretrain"
    assert str(parsed.output_dir) == "out/pretrain_machine_recipe"
    assert not hasattr(parsed, "seq_len")
    assert not hasattr(parsed, "learning_rate")
    assert not hasattr(parsed, "target_tokens_per_update")
    assert not hasattr(parsed, "batch_size")
    assert not hasattr(parsed, "step_execution_backend")


def test_config_builds_release_run_config_without_mutating_base() -> None:
    config = mod.PretrainMachineValidationConfig.from_namespace(
        argparse.Namespace(
            data_path="dataset/pretrain_tokens",
            tokenizer_path="tok",
            output_dir="out/sweep",
            device="cpu",
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
    assert int(base.warmup_steps) == 153
    assert int(run_config.warmup_steps) == 0
    assert not hasattr(run_config, "machine_recipe_accumulation_candidates")
    assert not hasattr(run_config, "machine_recipe_loss_chunk_candidates")
    assert not hasattr(run_config, "machine_recipe_checkpoint_candidates")
    assert int(base.save_best) == 0
    assert int(base.enable_checkpoints) == 0
    assert int(base.batch_size) == 1
    assert int(base.accumulation_steps) == 320


def test_config_keeps_formal_semantics() -> None:
    config = mod.PretrainMachineValidationConfig.from_namespace(
        argparse.Namespace(
            data_path="dataset/pretrain_tokens",
            tokenizer_path="tok",
            output_dir="out/sweep",
            device="cpu",
        )
    )
    profile = mod.RELEASE_PROFILE
    run_config = config.build_release_run_config(profile=profile)

    assert int(run_config.target_tokens_per_update) == 1_310_720
    assert str(config.run_name(base_run_config=run_config)) == "release_pretrain_sm120_graph"
    assert float(run_config.learning_rate) == pytest.approx(8.825e-4)
    assert int(run_config.seed) == 42
    assert int(run_config.enable_checkpoints) == 0
    assert int(run_config.batch_size) == 1
    assert int(run_config.accumulation_steps) == 320
    assert int(run_config.gradient_checkpointing) == 0
    assert str(run_config.step_execution_backend) == "sm120_graph"


def test_parser_rejects_training_semantic_overrides() -> None:
    parser = mod._build_parser()
    for option, value in (
        ("--seq_len", "16384"),
        ("--release_total_tokens", "5000000000"),
        ("--target_tokens_per_update", "524288"),
        ("--learning_rate", "1e-4"),
        ("--seed", "7"),
        ("--batch_size", "2"),
        ("--gradient_checkpointing", "1"),
        ("--step_execution_backend", "eager"),
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([option, value])


def test_machine_validation_uses_fixed_probe_token_budget(monkeypatch, tmp_path) -> None:
    seen: dict[str, object] = {}

    class _FakePipeline:
        def __init__(self, args):
            seen["total_tokens"] = int(args.total_tokens)
            self.output_dir = str(args.output_dir)

        def run(self) -> None:
            out = tmp_path / "run" / "release_pretrain"
            out.mkdir(parents=True, exist_ok=True)
            (out / "machine_recipe_summary.json").write_text(
                '{"selected":{"seq_len":4096,"batch_size":1,"accumulation_steps":320,"gradient_checkpointing":0,"loss_chunk_size":0,"dataloader_num_workers":0,"dataloader_prefetch_factor":0,"dataloader_persistent_workers":0,"shard_preload":1,"shard_preload_bytes":4194304}}',
                encoding="utf-8",
            )
            (out / "machine_recipe.json").write_text(
                '{"best":{"batch_size":1,"accumulation_steps":320,"tokens_per_sec":1.0,"max_memory_gb":1.0,"step_time_s":1.0,"candidate":{"gradient_checkpointing":false,"gradient_checkpointing_exclude_first":0,"gradient_checkpointing_exclude_last":0,"loss_chunk_size":0}}}',
                encoding="utf-8",
            )
            (out / "machine_runtime.json").write_text(
                '{"best":{"tokens_per_sec":1.0,"step_time_s":1.0,"shard_preload":0,"shard_preload_bytes":0}}',
                encoding="utf-8",
            )
            (out / "update_profile.json").write_text(
                '{"tokens_per_sec_est":1.0,"tokens_per_update":1310720}',
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

    assert int(seen["total_tokens"]) == 3 * 1_310_720


def test_machine_measurement_keeps_fixed_update_tokens(
    monkeypatch, tmp_path
) -> None:
    seen: dict[str, object] = {}

    class _FakePipeline:
        def __init__(self, args):
            seen["output_dir"] = str(args.output_dir)
            seen["target_tokens_per_update"] = int(args.target_tokens_per_update)

        def run(self) -> None:
            out = tmp_path / "run" / "release_pretrain_sm120_graph"
            out.mkdir(parents=True, exist_ok=True)
            (out / "machine_recipe_summary.json").write_text(
                '{"selected":{"seq_len":4096,"batch_size":1,"accumulation_steps":320,"gradient_checkpointing":0,"loss_chunk_size":0,"dataloader_num_workers":0,"dataloader_prefetch_factor":0,"dataloader_persistent_workers":0,"shard_preload":1,"shard_preload_bytes":4194304}}',
                encoding="utf-8",
            )
            (out / "machine_recipe.json").write_text(
                '{"best":{"batch_size":1,"accumulation_steps":320,"tokens_per_sec":1.0,"max_memory_gb":1.0,"step_time_s":1.0,"candidate":{"gradient_checkpointing":false,"gradient_checkpointing_exclude_first":0,"gradient_checkpointing_exclude_last":0,"loss_chunk_size":0}}}',
                encoding="utf-8",
            )
            (out / "machine_runtime.json").write_text(
                '{"best":{"tokens_per_sec":1.0,"step_time_s":1.0,"shard_preload":0,"shard_preload_bytes":0}}',
                encoding="utf-8",
            )
            (out / "update_profile.json").write_text(
                '{"tokens_per_sec_est":1.0,"tokens_per_update":1310720}',
                encoding="utf-8",
            )
            (out / "metrics.jsonl").write_text(
                _machine_measurement_metrics(),
                encoding="utf-8",
            )

    monkeypatch.setattr(mod, "PretrainPipeline", _FakePipeline)
    monkeypatch.setattr(
        mod,
        "_recipe_payload",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("fixture intentionally omits signed-recipe inputs")
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
        ],
    )
    mod.main()

    assert int(seen["target_tokens_per_update"]) == 1_310_720
    assert str(seen["output_dir"]).endswith("release_pretrain_sm120_graph")
    assert not (tmp_path / "run" / "release_pretrain_machine_recipe.json").exists()


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
            "lr_schedule": "cosine",
        },
        run_dir=run_dir,
        wall_time_s=1.0,
        error="",
    )

    assert str(summary["status"]) == "incomplete"
    assert str(summary["error"]) == "missing_or_non_finite_objective_metrics"


def test_machine_metrics_exclude_cold_start_from_steady_throughput(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(_machine_measurement_metrics(), encoding="utf-8")

    summary = mod._metrics_summary(path)

    assert summary["train_events"] == 3
    assert summary["first_update_wall_time_s"] == pytest.approx(100.0)
    assert summary["first_update_tokens_per_sec"] == pytest.approx(13_107.0)
    assert summary["steady_train_events"] == 2
    assert summary["steady_updates"] == 2
    assert summary["peak_train_memory_gb"] == pytest.approx(17.25)
    assert summary["steady_update_wall_time_s"] == pytest.approx(55.0)
    assert summary["steady_tokens_per_sec"] == pytest.approx(
        (50.0 * 26_214.0 + 60.0 * 21_845.0) / 110.0
    )


def test_machine_baseline_row_contains_release_gate_fields() -> None:
    run_config = mod.build_pretrain_args(
        data_path="dataset/pretrain_tokens",
        output_dir="out/release_pretrain",
    )
    row = mod._machine_baseline_row(
        summary={
            "name": "release_pretrain",
            "release_semantics": {
                "backend": "sm120_graph",
            },
            "machine_adaptive_selected": {
                "seq_len": 4096,
                "batch_size": 2,
                "accumulation_steps": 4,
                "gradient_checkpointing": 0,
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
    assert row["backend"] == "sm120_graph"
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
                '{"selected":{"seq_len":4096,"batch_size":1,"accumulation_steps":320,"gradient_checkpointing":0,"loss_chunk_size":0,"dataloader_num_workers":0,"dataloader_prefetch_factor":0,"dataloader_persistent_workers":0,"shard_preload":1,"shard_preload_bytes":4194304}}',
                encoding="utf-8",
            )
            (out / "machine_recipe.json").write_text(
                '{"best":{"batch_size":1,"accumulation_steps":320,"tokens_per_sec":1.0,"max_memory_gb":1.0,"step_time_s":1.0,"candidate":{"gradient_checkpointing":false,"gradient_checkpointing_exclude_first":0,"gradient_checkpointing_exclude_last":0,"loss_chunk_size":0}}}',
                encoding="utf-8",
            )
            (out / "machine_runtime.json").write_text(
                '{"best":{"tokens_per_sec":1.0,"step_time_s":1.0,"shard_preload":1,"shard_preload_bytes":4194304,"step_execution_backend":"sm120_graph"}}',
                encoding="utf-8",
            )
            (out / "update_profile.json").write_text(
                '{"tokens_per_sec_est":1.0,"tokens_per_update":1310720}',
                encoding="utf-8",
            )
            (out / "metrics.jsonl").write_text(
                _machine_measurement_metrics(),
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
                        "total_tokens": 3 * 1_310_720,
                        "batch_size": 1,
                        "accumulation_steps": 320,
                        "gradient_checkpointing": 0,
                        "gradient_checkpointing_exclude_first": 0,
                        "gradient_checkpointing_exclude_last": 0,
                        "loss_chunk_size": 0,
                        "target_tokens_per_update": 1_310_720,
                        "learning_rate": 8.825e-4,
                        "weight_decay": 0.1,
                        "warmup_steps": 0,
                        "warmup_ratio": 0.0,
                        "min_lr_ratio": 0.1,
                        "lr_schedule": "cosine",
                        "dataloader_num_workers": 0,
                        "dataloader_prefetch_factor": 0,
                        "dataloader_persistent_workers": 0,
                        "shard_preload": 1,
                        "shard_preload_bytes": 4194304,
                        "step_execution_backend": "sm120_graph",
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
    assert payload["selected_name"] == "measured_pretrain_sophia_1b_seq4096"
    assert payload["release_semantics"]["warmup_steps"] == 153
    assert (
        payload["release_semantics"]["profile"]["training"][
            "default_total_tokens"
        ]
        == 20_000_000_000
    )
    assert payload["machine_runtime"]["step_execution_backend"] == "sm120_graph"
    assert payload["machine_runtime"]["dataloader_num_workers"] == 0
    assert payload["machine_runtime"]["dataloader_prefetch_factor"] == 0
    assert payload["machine_runtime"]["dataloader_persistent_workers"] == 0
    assert payload["machine_runtime"]["shard_preload"] == 1
    assert payload["machine_runtime"]["shard_preload_bytes"] == 4194304
    assert payload["artifacts"]["machine_runtime"]["best"][
        "tokens_per_sec"
    ] == pytest.approx(
        1_310_720 / 55.0
    )
