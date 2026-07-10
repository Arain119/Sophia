from __future__ import annotations

import json
from pathlib import Path

import torch

from ml.tooling.scripts.release_gate import audit_resume_drill as mod


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix:
        path.write_text("", encoding="utf-8")
    else:
        path.mkdir(parents=True, exist_ok=True)


def _write_resume_run(
    *,
    stage: str,
    run_dir: Path,
    train_state: dict[str, object],
) -> None:
    for rel in ("run_args.json", "run_meta.json"):
        _touch(run_dir / rel)
    if str(stage) == "sft":
        _touch(run_dir / "machine_recipe_summary.json")
    _touch(run_dir / "tensorboard")
    (run_dir / "metrics.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").write_text(
        '{"type":"start","start_step":3,"max_steps":8}\n'
        '{"stage":"resume","step":4}\n'
        '{"stage":"resume","step":5}\n',
        encoding="utf-8",
    )
    checkpoint_path = run_dir / "checkpoints" / "ckpt_step5.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": 5,
            "model": {},
            "optimizer": {},
            "scheduler": None,
            "args": {"_sophia_resolved_resume_checkpoint": "/tmp/previous_ckpt.pt"},
            "rng": {},
            "ema": None,
            "train_state": train_state,
        },
        checkpoint_path,
    )


def test_build_resume_drill_audit_passes_for_sft_machine_evidence(tmp_path: Path) -> None:
    scenario_runs: dict[str, Path] = {}
    for scenario in mod.REQUIRED_SCENARIOS:
        run_dir = tmp_path / scenario
        _write_resume_run(
            stage="sft",
            run_dir=run_dir,
            train_state={
                "stage": "sft",
                "train_iter_state": {"offset": 16},
                "eval_iter_state": {"offset": 8},
                "test_iter_state": {"offset": 4},
                "samples_seen": 24,
            },
        )
        scenario_runs[scenario] = run_dir

    payload = mod.build_resume_drill_audit(stage="sft", scenario_runs=scenario_runs)

    assert payload["kind"] == "resume_drill_audit"
    assert payload["stage"] == "sft"
    assert payload["summary"]["all_passed"] is True
    assert payload["summary"]["failed"] == 0


def test_build_resume_drill_audit_fails_for_missing_required_scenario(tmp_path: Path) -> None:
    run_dir = tmp_path / "normal_stop"
    _write_resume_run(
        stage="pretrain",
        run_dir=run_dir,
        train_state={"data_iter_state": {"cursor": 1}},
    )

    try:
        mod.build_resume_drill_audit(
            stage="pretrain",
            scenario_runs={"normal_stop": run_dir},
        )
    except ValueError as exc:
        assert "missing required scenarios" in str(exc)
    else:
        raise AssertionError("expected ValueError for incomplete resume drill")


def test_build_resume_drill_audit_rejects_missing_posttrain_recipe_summary(tmp_path: Path) -> None:
    scenario_runs: dict[str, Path] = {}
    for scenario in mod.REQUIRED_SCENARIOS:
        run_dir = tmp_path / scenario
        _write_resume_run(
            stage="sft",
            run_dir=run_dir,
            train_state={
                "stage": "sft",
                "train_iter_state": {"offset": 16},
                "eval_iter_state": {"offset": 8},
                "test_iter_state": {"offset": 4},
                "samples_seen": 24,
            },
        )
        if scenario == "forced_kill":
            (run_dir / "machine_recipe_summary.json").unlink()
        scenario_runs[scenario] = run_dir

    payload = mod.build_resume_drill_audit(stage="sft", scenario_runs=scenario_runs)

    assert payload["summary"]["all_passed"] is False
    forced = next(item for item in payload["scenario_reports"] if item["scenario"] == "forced_kill")
    checks = {item["name"]: item for item in forced["checks"]}
    assert checks["required_run_artifacts"]["passed"] is False
    assert "machine_recipe_summary.json" in checks["required_run_artifacts"]["detail"]


def test_build_resume_drill_audit_rejects_stale_tmp_artifacts(tmp_path: Path) -> None:
    scenario_runs: dict[str, Path] = {}
    for scenario in mod.REQUIRED_SCENARIOS:
        run_dir = tmp_path / scenario
        _write_resume_run(
            stage="sft",
            run_dir=run_dir,
            train_state={
                "stage": "sft",
                "train_iter_state": {"offset": 16},
                "eval_iter_state": {"offset": 8},
                "test_iter_state": {"offset": 4},
                "samples_seen": 24,
            },
        )
        scenario_runs[scenario] = run_dir
    (scenario_runs["post_checkpoint"] / "checkpoints" / "ckpt_step5.pt.tmp.789").write_text("", encoding="utf-8")

    payload = mod.build_resume_drill_audit(stage="sft", scenario_runs=scenario_runs)

    assert payload["summary"]["all_passed"] is False
    post_checkpoint = next(item for item in payload["scenario_reports"] if item["scenario"] == "post_checkpoint")
    checks = {item["name"]: item for item in post_checkpoint["checks"]}
    assert checks["no_stale_tmp"]["passed"] is False
