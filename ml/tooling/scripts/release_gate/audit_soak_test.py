#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml.core.common.io import write_json_atomic
from ml.training.pretrain.artifacts import (
    PRETRAIN_MACHINE_RECIPE_ARTIFACT,
    PRETRAIN_MACHINE_RECIPE_SUMMARY,
    PRETRAIN_MACHINE_RUNTIME_ARTIFACT,
    PRETRAIN_UPDATE_PROFILE_ARTIFACT,
)


STAGE_CHOICES: tuple[str, ...] = ("pretrain", "sft")


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _summarize_checks(checks: list[GateCheck]) -> dict[str, Any]:
    passed = sum(1 for item in checks if bool(item.passed))
    return {
        "passed": int(passed),
        "failed": int(len(checks) - passed),
        "all_passed": bool(passed == len(checks)),
    }


def _stage_trial_check(*, stage: str, run_dir: Path) -> GateCheck:
    required = ["metrics.jsonl", "run_args.json", "run_meta.json", "tensorboard"]
    if str(stage) == "pretrain":
        required.extend(
            [
                PRETRAIN_MACHINE_RECIPE_SUMMARY,
                PRETRAIN_MACHINE_RECIPE_ARTIFACT,
                PRETRAIN_MACHINE_RUNTIME_ARTIFACT,
                PRETRAIN_UPDATE_PROFILE_ARTIFACT,
            ]
        )
    elif str(stage) == "sft":
        required.append("machine_recipe_summary.json")
        required.append("sft_capability_report.json")
    else:
        raise ValueError(f"unsupported stage: {stage!r}")
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        return GateCheck(
            name=f"{stage}_trial_run_artifacts",
            passed=False,
            detail=f"missing under {run_dir}: {', '.join(missing)}",
        )
    return GateCheck(
        name=f"{stage}_trial_run_artifacts",
        passed=True,
        detail=str(run_dir),
    )


def _stage_tmp_check(*, stage: str, run_dir: Path) -> GateCheck:
    if not run_dir.exists():
        return GateCheck(
            name=f"{stage}_trial_run_no_stale_tmp",
            passed=False,
            detail=f"missing directory: {run_dir}",
        )
    found: list[str] = []
    for path in run_dir.rglob("*"):
        if ".tmp." not in path.name:
            continue
        found.append(str(path))
        if len(found) >= 5:
            break
    if found:
        return GateCheck(
            name=f"{stage}_trial_run_no_stale_tmp",
            passed=False,
            detail="stale tmp artifacts: " + ", ".join(found),
        )
    return GateCheck(
        name=f"{stage}_trial_run_no_stale_tmp",
        passed=True,
        detail=str(run_dir),
    )


def _resume_report_check(*, path: Path) -> GateCheck:
    try:
        payload = _json_object(path)
    except Exception as exc:
        return GateCheck(
            name=f"resume_report::{path.name}",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )
    summary = payload.get("summary")
    passed = (
        payload.get("kind") == "resume_drill_audit"
        and isinstance(summary, dict)
        and bool(summary.get("all_passed"))
    )
    return GateCheck(
        name=f"resume_report::{path.name}",
        passed=passed,
        detail=(str(path) if passed else f"invalid or failing resume report: {path}"),
    )


def build_soak_test_audit(
    *,
    stage_outputs: dict[str, Path],
    resume_report_paths: tuple[Path, ...],
) -> dict[str, Any]:
    checks: list[GateCheck] = []
    normalized_stage_outputs: dict[str, str] = {}
    for stage, run_dir in stage_outputs.items():
        normalized_stage = str(stage or "").strip().lower()
        if normalized_stage not in STAGE_CHOICES:
            raise ValueError(f"unsupported stage: {stage!r}")
        resolved = Path(str(run_dir)).resolve()
        normalized_stage_outputs[normalized_stage] = str(resolved)
        checks.append(_stage_trial_check(stage=normalized_stage, run_dir=resolved))
        checks.append(_stage_tmp_check(stage=normalized_stage, run_dir=resolved))

    if not normalized_stage_outputs:
        checks.append(
            GateCheck(
                name="release_trial_run_present",
                passed=False,
                detail="no stage outputs were provided",
            )
        )
    else:
        successful_stages = [
            item.name.removesuffix("_trial_run_artifacts")
            for item in checks
            if item.name.endswith("_trial_run_artifacts") and bool(item.passed)
        ]
        checks.append(
            GateCheck(
                name="release_trial_run_present",
                passed=bool(successful_stages),
                detail=(
                    ",".join(successful_stages)
                    if successful_stages
                    else "no provided trial run passed artifact checks"
                ),
            )
        )

    if not resume_report_paths:
        checks.append(
            GateCheck(
                name="resume_report_present",
                passed=False,
                detail="no resume_report paths were provided",
            )
        )
    else:
        resume_checks = [_resume_report_check(path=Path(str(path)).resolve()) for path in resume_report_paths]
        checks.extend(resume_checks)
        checks.append(
            GateCheck(
                name="resume_report_present",
                passed=any(bool(item.passed) for item in resume_checks),
                detail=(
                    "at least one passing resume report found"
                    if any(bool(item.passed) for item in resume_checks)
                    else "all provided resume reports failed"
                ),
            )
        )

    return {
        "kind": "soak_test_audit",
        "stage_outputs": normalized_stage_outputs,
        "resume_reports": [str(Path(str(path)).resolve()) for path in resume_report_paths],
        "checks": [asdict(item) for item in checks],
        "summary": _summarize_checks(checks),
    }


def _parse_stage_outputs(raw_items: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in raw_items:
        raw = str(item or "").strip()
        if "=" not in raw:
            raise ValueError(f"expected STAGE=RUN_DIR, got: {item!r}")
        stage, run_dir = raw.split("=", 1)
        normalized_stage = str(stage or "").strip().lower()
        if normalized_stage not in STAGE_CHOICES:
            raise ValueError(
                f"unsupported stage {normalized_stage!r}; expected one of {STAGE_CHOICES!r}"
            )
        if normalized_stage in parsed:
            raise ValueError(f"duplicate stage: {normalized_stage}")
        parsed[normalized_stage] = Path(str(run_dir or "").strip()).resolve()
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit target-machine soak-test evidence for release training readiness."
    )
    parser.add_argument(
        "--stage_output",
        type=str,
        action="append",
        default=[],
        help="Stage/run mapping in the form STAGE=RUN_DIR. Example: sft=out/release_sft",
    )
    parser.add_argument(
        "--resume_report",
        type=str,
        action="append",
        default=[],
        help="Path to a passing resume drill audit report.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="out/training_soak_test.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = build_soak_test_audit(
        stage_outputs=_parse_stage_outputs(list(args.stage_output or [])),
        resume_report_paths=tuple(
            Path(str(path)).resolve() for path in list(args.resume_report or [])
        ),
    )
    output_path = Path(str(args.output_json)).resolve()
    write_json_atomic(
        output_path,
        payload,
        sort_keys=True,
        ensure_ascii=False,
        make_parents=True,
    )
    print(
        "[AUDIT][SOAK_TEST] "
        f"passed={int(payload['summary']['passed'])} "
        f"failed={int(payload['summary']['failed'])} "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
