from __future__ import annotations

import json
from pathlib import Path

from ml.tooling.scripts.release_gate import audit_soak_test as mod


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix:
        path.write_text("", encoding="utf-8")
    else:
        path.mkdir(parents=True, exist_ok=True)


def test_build_soak_test_audit_passes_for_trial_run_plus_resume_report(tmp_path: Path) -> None:
    sft_output = tmp_path / "release_sft"
    for rel in ("metrics.jsonl", "run_args.json", "run_meta.json", "machine_recipe_summary.json"):
        _touch(sft_output / rel)
    _touch(sft_output / "tensorboard")
    _write_json(sft_output / "sft_capability_report.json", {"ok": True})

    resume_report = tmp_path / "sft_resume_drill.json"
    _write_json(
        resume_report,
        {
            "kind": "resume_drill_audit",
            "stage": "sft",
            "summary": {"all_passed": True, "passed": 3, "failed": 0},
        },
    )

    payload = mod.build_soak_test_audit(
        stage_outputs={"sft": sft_output},
        resume_report_paths=(resume_report,),
    )

    assert payload["kind"] == "soak_test_audit"
    assert payload["summary"]["all_passed"] is True


def test_build_soak_test_audit_fails_without_resume_report(tmp_path: Path) -> None:
    payload = mod.build_soak_test_audit(stage_outputs={}, resume_report_paths=())

    assert payload["summary"]["all_passed"] is False
    names = {item["name"]: item for item in payload["checks"]}
    assert names["resume_report_present"]["passed"] is False


def test_build_soak_test_audit_rejects_stale_tmp_artifacts(tmp_path: Path) -> None:
    sft_output = tmp_path / "release_sft"
    for rel in ("metrics.jsonl", "run_args.json", "run_meta.json", "machine_recipe_summary.json"):
        _touch(sft_output / rel)
    _touch(sft_output / "tensorboard")
    _write_json(sft_output / "sft_capability_report.json", {"ok": True})
    (sft_output / "checkpoints" / "ckpt_step10.pt.tmp.456").parent.mkdir(parents=True, exist_ok=True)
    (sft_output / "checkpoints" / "ckpt_step10.pt.tmp.456").write_text("", encoding="utf-8")

    resume_report = tmp_path / "sft_resume_drill.json"
    _write_json(
        resume_report,
        {
            "kind": "resume_drill_audit",
            "stage": "sft",
            "summary": {"all_passed": True, "passed": 3, "failed": 0},
        },
    )

    payload = mod.build_soak_test_audit(
        stage_outputs={"sft": sft_output},
        resume_report_paths=(resume_report,),
    )

    names = {item["name"]: item for item in payload["checks"]}
    assert names["sft_trial_run_no_stale_tmp"]["passed"] is False
