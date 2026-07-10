from __future__ import annotations

import importlib.util
import sys

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch is not installed in this environment", allow_module_level=True)

from ml.tooling.scripts.machine_recipes import select_sft_machine_recipe as mod


def test_release_semantics_use_fixed_examples_per_update(tmp_path) -> None:
    curriculum = tmp_path / "curriculum.json"
    curriculum.write_text(
        '{"sft_curriculum":[{"stage":"sft_core","max_seq_len":4096}]}',
        encoding="utf-8",
    )
    semantics = mod.build_fixed_sft_release_semantics(
        curriculum_path=str(curriculum),
        curriculum_stage="sft_core",
        learning_rate=5e-6,
        weight_decay=0.1,
        notes=(
            f"target_examples_per_update={int(mod.RELEASE_SFT_DEFAULTS.target_examples_per_update)} "
            "learning_rate=5.000e-06 weight_decay=0.100"
        ),
    )

    assert semantics.name == "release_sft"
    assert semantics.target_examples_per_update == 16


def test_machine_candidates_filter_to_divisible_example_updates() -> None:
    candidates = mod._machine_candidates(
        target_examples_per_update=16,
        micro_batch_candidates=(1, 2, 3, 4, 8, 16, 32),
        checkpoint_candidates=(False, True),
    )

    assert [
        (candidate.batch_size, candidate.accumulation_steps, candidate.gradient_checkpointing)
        for candidate in candidates
    ] == [
        (1, 16, True),
        (2, 8, True),
        (4, 4, True),
        (8, 2, True),
        (16, 1, True),
        (1, 16, False),
        (2, 8, False),
        (4, 4, False),
        (8, 2, False),
        (16, 1, False),
    ]


def test_sweep_selects_fastest_successful_machine_candidate(monkeypatch, tmp_path) -> None:
    seen_summary: dict[str, object] = {}

    def _fake_run_command(*, argv: list[str], cwd, timeout_s: float) -> None:
        del argv, cwd, timeout_s
        return None

    def _fake_run_summary(*, semantics, machine, run_dir, wall_time_s, error=""):
        del run_dir, wall_time_s, error
        speed = float(machine.batch_size * 10)
        return {
            "name": str(semantics.name),
            "status": "ok",
            "release_semantics": {
                "target_examples_per_update": int(semantics.target_examples_per_update),
            },
            "machine_selected": {
                "batch_size": int(machine.batch_size),
                "accumulation_steps": int(machine.accumulation_steps),
                "gradient_checkpointing": int(bool(machine.gradient_checkpointing)),
            },
            "metrics": {
                "wall_tokens_per_s": float(speed),
                "train_tokens_per_s": float(speed),
                "best_eval_loss": 1.0 - float(machine.batch_size) * 0.01,
                "last_eval_loss": 1.0 - float(machine.batch_size) * 0.01,
            },
            "report": {
                "val_loss_mean": 0.9 - float(machine.batch_size) * 0.01,
                "test_loss_mean": 0.95 - float(machine.batch_size) * 0.01,
            },
        }

    def _capture_write_json(path, payload) -> None:
        if str(path).endswith("summary.json"):
            seen_summary["payload"] = payload

    monkeypatch.setattr(mod, "_run_command", _fake_run_command)
    monkeypatch.setattr(mod, "_run_summary", _fake_run_summary)
    monkeypatch.setattr(mod, "_write_json", _capture_write_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_sft_machine_recipe.py",
            "--export_dir",
            str(tmp_path / "hf"),
            "--train_data",
            str(tmp_path / "train.jsonl"),
            "--eval_data",
            str(tmp_path / "val.jsonl"),
            "--test_data",
            str(tmp_path / "test.jsonl"),
            "--output_dir",
            str(tmp_path / "out"),
            "--micro_batch_candidates",
            "1,2,4,8,16",
            "--checkpoint_candidates",
            "false",
        ],
    )

    mod.main()

    payload = seen_summary["payload"]
    assert isinstance(payload, dict)
    selected = payload["selected_runs"][0]
    assert selected["machine_selected"]["batch_size"] == 16
    assert payload["frontier_quality_speed"][0]["batch_size"] == 16
    assert payload["machine_baseline_table"][0]["batch_size"] == 16
    assert payload["machine_baseline_table"][0]["gradient_checkpointing"] == 0


def test_run_summary_reads_machine_selection_from_posttrain_run_args(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_args.json").write_text(
        '{"batch_size":2,"accumulation_steps":3,"gradient_checkpointing":1}',
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(
        '{"stage":"sft","wall_tokens_per_s":10.0,"train_tokens_per_s":10.0,"eval_loss":1.0,"test_loss":1.1}\n',
        encoding="utf-8",
    )
    (run_dir / "sft_capability_report.json").write_text(
        '{"splits":{"val":{"overall":{"loss_mean":0.9}}}}',
        encoding="utf-8",
    )

    summary = mod._run_summary(
        semantics=mod.SftReleaseSemantics(
            name="release_sft",
            curriculum_stage="sft_core",
            target_examples_per_update=16,
            learning_rate=5e-6,
            weight_decay=0.1,
            eval_interval=1,
            save_interval=1,
            resolved_max_seq_len=4096,
            max_seq_len_source="curriculum:sft_core",
        ),
        machine=mod.MachineCandidate(
            batch_size=1,
            accumulation_steps=16,
            gradient_checkpointing=False,
        ),
        run_dir=run_dir,
        wall_time_s=1.0,
    )

    assert summary["machine_selected"] == {
        "batch_size": 2,
        "accumulation_steps": 3,
        "gradient_checkpointing": 1,
    }
    assert summary["metrics"]["test_loss"] == 1.1
