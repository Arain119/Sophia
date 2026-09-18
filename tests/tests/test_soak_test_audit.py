from __future__ import annotations

import json
from pathlib import Path

from ml.tooling.scripts.release_gate import audit_soak_test as mod
from ml.training.pretrain.artifacts import (
    PRETRAIN_MACHINE_RECIPE_ARTIFACT,
    PRETRAIN_MACHINE_RECIPE_SUMMARY,
    PRETRAIN_MACHINE_RUNTIME_ARTIFACT,
    PRETRAIN_UPDATE_PROFILE_ARTIFACT,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix:
        path.write_text("", encoding="utf-8")
    else:
        path.mkdir(parents=True, exist_ok=True)


def _prepare_pretrain_output(root: Path) -> Path:
    output = root / "release_pretrain"
    for rel in (
        "metrics.jsonl",
        "run_args.json",
        "run_meta.json",
        PRETRAIN_MACHINE_RECIPE_SUMMARY,
        PRETRAIN_MACHINE_RECIPE_ARTIFACT,
        PRETRAIN_MACHINE_RUNTIME_ARTIFACT,
        PRETRAIN_UPDATE_PROFILE_ARTIFACT,
    ):
        _touch(output / rel)
    _touch(output / "tensorboard")
    return output


def test_build_soak_test_audit_passes_for_trial_run_plus_resume_report(
    tmp_path: Path,
) -> None:
    pretrain_output = _prepare_pretrain_output(tmp_path)
    resume_report = tmp_path / "pretrain_resume_drill.json"
    _write_json(
        resume_report,
        {
            "kind": "resume_drill_audit",
            "stage": "pretrain",
            "summary": {"all_passed": True, "passed": 3, "failed": 0},
        },
    )

    payload = mod.build_soak_test_audit(
        stage_outputs={"pretrain": pretrain_output},
        resume_report_paths=(resume_report,),
    )

    assert payload["kind"] == "soak_test_audit"
    assert payload["summary"]["all_passed"] is True


def test_build_soak_test_audit_fails_without_resume_report(tmp_path: Path) -> None:
    payload = mod.build_soak_test_audit(stage_outputs={}, resume_report_paths=())

    assert payload["summary"]["all_passed"] is False
    names = {item["name"]: item for item in payload["checks"]}
    assert names["resume_report_present"]["passed"] is False


def test_build_soak_test_audit_rejects_resume_report_for_wrong_stage(
    tmp_path: Path,
) -> None:
    pretrain_output = _prepare_pretrain_output(tmp_path)
    resume_report = tmp_path / "wrong_stage_resume_drill.json"
    _write_json(
        resume_report,
        {
            "kind": "resume_drill_audit",
            "stage": "unsupported",
            "summary": {"all_passed": True, "passed": 3, "failed": 0},
        },
    )

    payload = mod.build_soak_test_audit(
        stage_outputs={"pretrain": pretrain_output},
        resume_report_paths=(resume_report,),
    )

    names = {item["name"]: item for item in payload["checks"]}
    assert names[f"resume_report::{resume_report.name}"]["passed"] is False
    assert names["resume_report_present"]["passed"] is False


def test_build_soak_test_audit_rejects_stale_tmp_artifacts(tmp_path: Path) -> None:
    pretrain_output = _prepare_pretrain_output(tmp_path)
    temp = pretrain_output / "checkpoints" / "ckpt_step10.pt.tmp.456"
    temp.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text("", encoding="utf-8")
    resume_report = tmp_path / "pretrain_resume_drill.json"
    _write_json(
        resume_report,
        {
            "kind": "resume_drill_audit",
            "stage": "pretrain",
            "summary": {"all_passed": True, "passed": 3, "failed": 0},
        },
    )

    payload = mod.build_soak_test_audit(
        stage_outputs={"pretrain": pretrain_output},
        resume_report_paths=(resume_report,),
    )

    names = {item["name"]: item for item in payload["checks"]}
    assert names["pretrain_trial_run_no_stale_tmp"]["passed"] is False


def test_build_soak_test_audit_rejects_unsupported_stage(tmp_path: Path) -> None:
    try:
        mod.build_soak_test_audit(
            stage_outputs={"unsupported": tmp_path},
            resume_report_paths=(),
        )
    except ValueError as exc:
        assert "unsupported stage" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported stage")
