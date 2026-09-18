#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml.core.common.io import write_json_atomic
from ml.core.engine.checkpointing import latest_checkpoint_path, load_checkpoint


REQUIRED_SCENARIOS: tuple[str, ...] = (
    "normal_stop",
    "forced_kill",
    "post_checkpoint",
)
STAGE_CHOICES: tuple[str, ...] = ("pretrain",)


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _summarize_checks(checks: list[GateCheck]) -> dict[str, Any]:
    passed = sum(1 for item in checks if bool(item.passed))
    return {
        "passed": int(passed),
        "failed": int(len(checks) - passed),
        "all_passed": bool(passed == len(checks)),
    }


def _artifact_check(*, stage: str, run_dir: Path) -> GateCheck:
    required = (
        "metrics.jsonl",
        "run_args.json",
        "run_meta.json",
        "tensorboard",
        "checkpoints",
    )
    if str(stage) != "pretrain":
        raise ValueError(f"unsupported stage: {stage!r}")
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        return GateCheck(
            name="required_run_artifacts",
            passed=False,
            detail=f"missing under {run_dir}: {', '.join(missing)}",
        )
    return GateCheck(name="required_run_artifacts", passed=True, detail=str(run_dir))


def _stale_tmp_check(*, run_dir: Path) -> GateCheck:
    if not run_dir.exists():
        return GateCheck(
            name="no_stale_tmp",
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
            name="no_stale_tmp",
            passed=False,
            detail="stale tmp artifacts: " + ", ".join(found),
        )
    return GateCheck(name="no_stale_tmp", passed=True, detail=str(run_dir))


def _resumed_start_event_check(*, run_dir: Path) -> GateCheck:
    metrics_path = run_dir / "metrics.jsonl"
    try:
        rows = _jsonl_rows(metrics_path)
    except Exception as exc:
        return GateCheck(
            name="resumed_start_event",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )
    resumed_rows = [
        row
        for row in rows
        if str(row.get("type", "") or "") == "start"
        and int(row.get("start_step", 0) or 0) > 0
    ]
    if not resumed_rows:
        return GateCheck(
            name="resumed_start_event",
            passed=False,
            detail=f"no start event with start_step>0 in {metrics_path}",
        )
    return GateCheck(
        name="resumed_start_event",
        passed=True,
        detail=f"start_step={int(resumed_rows[-1].get('start_step', 0) or 0)}",
    )


def _metric_step_continuity_check(*, run_dir: Path) -> GateCheck:
    metrics_path = run_dir / "metrics.jsonl"
    try:
        rows = _jsonl_rows(metrics_path)
    except Exception as exc:
        return GateCheck(
            name="metric_step_continuity",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )
    steps: list[int] = []
    for row in rows:
        if "step" not in row:
            continue
        try:
            steps.append(int(row.get("step", 0) or 0))
        except Exception:
            continue
    if not steps:
        return GateCheck(
            name="metric_step_continuity",
            passed=False,
            detail=f"no step rows in {metrics_path}",
        )
    for previous, current in zip(steps, steps[1:], strict=False):
        if int(current) < int(previous):
            return GateCheck(
                name="metric_step_continuity",
                passed=False,
                detail=f"non-monotonic steps in {metrics_path}: {previous} -> {current}",
            )
    return GateCheck(
        name="metric_step_continuity",
        passed=True,
        detail=f"steps={int(steps[0])}..{int(steps[-1])}",
    )


def _checkpoint_contract_checks(*, stage: str, run_dir: Path) -> list[GateCheck]:
    checkpoint_path_raw = latest_checkpoint_path(str(run_dir))
    if checkpoint_path_raw is None:
        return [
            GateCheck(
                name="latest_checkpoint_loadable",
                passed=False,
                detail=f"no checkpoint under {run_dir / 'checkpoints'}",
            ),
            GateCheck(
                name="checkpoint_resume_marker_present",
                passed=False,
                detail="checkpoint unavailable",
            ),
            GateCheck(
                name="checkpoint_train_state_present",
                passed=False,
                detail="checkpoint unavailable",
            ),
            GateCheck(
                name="stage_train_state_contract",
                passed=False,
                detail="checkpoint unavailable",
            ),
        ]

    checkpoint_path = Path(str(checkpoint_path_raw))
    try:
        checkpoint = load_checkpoint(str(checkpoint_path), expected_kind="full")
    except Exception as exc:
        return [
            GateCheck(
                name="latest_checkpoint_loadable",
                passed=False,
                detail=f"{type(exc).__name__}: {exc}",
            ),
            GateCheck(
                name="checkpoint_resume_marker_present",
                passed=False,
                detail="checkpoint load failed",
            ),
            GateCheck(
                name="checkpoint_train_state_present",
                passed=False,
                detail="checkpoint load failed",
            ),
            GateCheck(
                name="stage_train_state_contract",
                passed=False,
                detail="checkpoint load failed",
            ),
        ]

    checks = [
        GateCheck(
            name="latest_checkpoint_loadable",
            passed=bool(int(checkpoint.step) > 0),
            detail=f"path={checkpoint_path} step={int(checkpoint.step)}",
        )
    ]
    args_payload = checkpoint.args if isinstance(checkpoint.args, dict) else {}
    resolved_resume = str(
        args_payload.get("_sophia_resolved_resume_checkpoint", "") or ""
    ).strip()
    checks.append(
        GateCheck(
            name="checkpoint_resume_marker_present",
            passed=bool(resolved_resume),
            detail=(
                resolved_resume
                or f"missing _sophia_resolved_resume_checkpoint in {checkpoint_path}"
            ),
        )
    )
    train_state = (
        checkpoint.train_state if isinstance(checkpoint.train_state, dict) else None
    )
    checks.append(
        GateCheck(
            name="checkpoint_train_state_present",
            passed=bool(isinstance(train_state, dict) and train_state),
            detail=(
                f"keys={','.join(sorted(str(key) for key in train_state.keys()))}"
                if isinstance(train_state, dict) and train_state
                else f"missing train_state in {checkpoint_path}"
            ),
        )
    )
    checks.append(
        _stage_train_state_contract_check(
            stage=str(stage), train_state=train_state, checkpoint_path=checkpoint_path
        )
    )
    return checks


def _stage_train_state_contract_check(
    *,
    stage: str,
    train_state: dict[str, Any] | None,
    checkpoint_path: Path,
) -> GateCheck:
    if not isinstance(train_state, dict):
        return GateCheck(
            name="stage_train_state_contract",
            passed=False,
            detail=f"missing train_state in {checkpoint_path}",
        )
    if str(stage) != "pretrain":
        return GateCheck(
            name="stage_train_state_contract",
            passed=False,
            detail=f"unsupported stage {stage!r} in {checkpoint_path}",
        )
    data_iter_state = train_state.get("data_iter_state")
    passed = isinstance(data_iter_state, dict) and bool(data_iter_state)
    return GateCheck(
        name="stage_train_state_contract",
        passed=passed,
        detail=(
            "pretrain data_iter_state captured"
            if passed
            else f"missing data_iter_state in {checkpoint_path}"
        ),
    )


def build_resume_drill_audit(
    *,
    stage: str,
    scenario_runs: dict[str, Path],
) -> dict[str, Any]:
    normalized_stage = str(stage or "").strip().lower()
    if normalized_stage not in STAGE_CHOICES:
        raise ValueError(f"unsupported stage: {stage!r}")
    missing_scenarios = [
        scenario for scenario in REQUIRED_SCENARIOS if scenario not in scenario_runs
    ]
    if missing_scenarios:
        raise ValueError(
            "missing required scenarios: "
            + ", ".join(str(item) for item in missing_scenarios)
        )

    scenario_reports: list[dict[str, Any]] = []
    for scenario in REQUIRED_SCENARIOS:
        run_dir = Path(str(scenario_runs[scenario])).resolve()
        checks = [
            _artifact_check(stage=normalized_stage, run_dir=run_dir),
            _stale_tmp_check(run_dir=run_dir),
            _resumed_start_event_check(run_dir=run_dir),
            _metric_step_continuity_check(run_dir=run_dir),
            *_checkpoint_contract_checks(stage=normalized_stage, run_dir=run_dir),
        ]
        scenario_reports.append(
            {
                "scenario": str(scenario),
                "run_dir": str(run_dir),
                "checks": [asdict(item) for item in checks],
                "summary": _summarize_checks(checks),
            }
        )

    passed = sum(
        1
        for item in scenario_reports
        if bool(dict(item.get("summary", {})).get("all_passed"))
    )
    return {
        "kind": "resume_drill_audit",
        "stage": normalized_stage,
        "required_scenarios": list(REQUIRED_SCENARIOS),
        "scenario_reports": scenario_reports,
        "summary": {
            "passed": int(passed),
            "failed": int(len(scenario_reports) - passed),
            "all_passed": bool(passed == len(scenario_reports)),
        },
    }


def _parse_scenario_runs(raw_items: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in raw_items:
        raw = str(item or "").strip()
        if "=" not in raw:
            raise ValueError(f"expected SCENARIO=RUN_DIR, got: {item!r}")
        scenario, run_dir = raw.split("=", 1)
        normalized_scenario = str(scenario or "").strip().lower()
        if normalized_scenario not in REQUIRED_SCENARIOS:
            raise ValueError(
                f"unsupported scenario {normalized_scenario!r}; expected one of {REQUIRED_SCENARIOS!r}"
            )
        if normalized_scenario in parsed:
            raise ValueError(f"duplicate scenario: {normalized_scenario}")
        resolved_run_dir = Path(str(run_dir or "").strip()).resolve()
        parsed[normalized_scenario] = resolved_run_dir
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit machine-level resume drill evidence for pretraining."
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=STAGE_CHOICES,
    )
    parser.add_argument(
        "--scenario_run",
        type=str,
        action="append",
        default=[],
        help="Scenario/run mapping in the form SCENARIO=RUN_DIR. Required scenarios: normal_stop, forced_kill, post_checkpoint.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="out/resume_drill_audit.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = build_resume_drill_audit(
        stage=str(args.stage),
        scenario_runs=_parse_scenario_runs(list(args.scenario_run or [])),
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
        "[AUDIT][RESUME_DRILL] "
        f"stage={payload['stage']} "
        f"passed={int(payload['summary']['passed'])} "
        f"failed={int(payload['summary']['failed'])} "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
