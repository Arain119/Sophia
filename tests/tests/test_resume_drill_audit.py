from __future__ import annotations

from pathlib import Path

from ml.core.engine.checkpointing import save_checkpoint_state
from ml.tooling.scripts.release_gate import audit_resume_drill as mod


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix:
        path.write_text("", encoding="utf-8")
    else:
        path.mkdir(parents=True, exist_ok=True)


def _write_resume_run(*, run_dir: Path) -> None:
    for rel in ("run_args.json", "run_meta.json"):
        _touch(run_dir / rel)
    _touch(run_dir / "tensorboard")
    (run_dir / "metrics.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").write_text(
        '{"type":"start","start_step":3,"max_steps":8}\n'
        '{"stage":"resume","step":4}\n'
        '{"stage":"resume","step":5}\n',
        encoding="utf-8",
    )
    save_checkpoint_state(
        output_dir=str(run_dir),
        step=5,
        state={
            "schema": "sophia_engine_checkpoint_v1",
            "kind": "full",
            "step": 5,
            "model": {},
            "optimizer": {},
            "scheduler": None,
            "args": {"_sophia_resolved_resume_checkpoint": "/tmp/previous_ckpt.pt"},
            "rng": {},
            "ema": None,
            "train_state": {"data_iter_state": {"cursor": 24}},
        },
        save_total_limit=1,
    )


def test_build_resume_drill_audit_passes_for_pretrain_machine_evidence(
    tmp_path: Path,
) -> None:
    scenario_runs: dict[str, Path] = {}
    for scenario in mod.REQUIRED_SCENARIOS:
        run_dir = tmp_path / scenario
        _write_resume_run(run_dir=run_dir)
        scenario_runs[scenario] = run_dir

    payload = mod.build_resume_drill_audit(
        stage="pretrain",
        scenario_runs=scenario_runs,
    )

    assert payload["kind"] == "resume_drill_audit"
    assert payload["stage"] == "pretrain"
    assert payload["summary"]["all_passed"] is True
    assert payload["summary"]["failed"] == 0


def test_build_resume_drill_audit_fails_for_missing_required_scenario(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "normal_stop"
    _write_resume_run(run_dir=run_dir)

    try:
        mod.build_resume_drill_audit(
            stage="pretrain",
            scenario_runs={"normal_stop": run_dir},
        )
    except ValueError as exc:
        assert "missing required scenarios" in str(exc)
    else:
        raise AssertionError("expected ValueError for incomplete resume drill")


def test_build_resume_drill_audit_rejects_stale_tmp_artifacts(
    tmp_path: Path,
) -> None:
    scenario_runs: dict[str, Path] = {}
    for scenario in mod.REQUIRED_SCENARIOS:
        run_dir = tmp_path / scenario
        _write_resume_run(run_dir=run_dir)
        scenario_runs[scenario] = run_dir
    (
        scenario_runs["post_checkpoint"] / "checkpoints" / "ckpt_step5.pt.tmp.789"
    ).write_text("", encoding="utf-8")

    payload = mod.build_resume_drill_audit(
        stage="pretrain",
        scenario_runs=scenario_runs,
    )

    assert payload["summary"]["all_passed"] is False
    post_checkpoint = next(
        item
        for item in payload["scenario_reports"]
        if item["scenario"] == "post_checkpoint"
    )
    checks = {item["name"]: item for item in post_checkpoint["checks"]}
    assert checks["no_stale_tmp"]["passed"] is False


def test_build_resume_drill_audit_rejects_unsupported_stage(tmp_path: Path) -> None:
    try:
        mod.build_resume_drill_audit(stage="unsupported", scenario_runs={})
    except ValueError as exc:
        assert "unsupported stage" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported stage")
