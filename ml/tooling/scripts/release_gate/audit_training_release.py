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


def _baseline_table_check(
    *,
    name: str,
    path: Path,
    required_keys: tuple[str, ...],
) -> GateCheck:
    try:
        payload = _json_object(path)
    except Exception as exc:
        return GateCheck(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}")
    rows = payload.get("machine_baseline_table")
    if not isinstance(rows, list) or not rows:
        return GateCheck(
            name=name,
            passed=False,
            detail=f"missing_or_empty machine_baseline_table in {path}",
        )
    first = rows[0]
    if not isinstance(first, dict) or not all(key in first for key in required_keys):
        return GateCheck(
            name=name,
            passed=False,
            detail=f"baseline row is missing required keys in {path}",
        )
    return GateCheck(name=name, passed=True, detail=str(path))


def _artifact_check(*, name: str, root: Path, relative_paths: tuple[str, ...]) -> GateCheck:
    missing = [rel for rel in relative_paths if not (root / rel).exists()]
    if missing:
        return GateCheck(
            name=name,
            passed=False,
            detail=f"missing under {root}: {', '.join(missing)}",
        )
    return GateCheck(name=name, passed=True, detail=str(root))


def _stale_tmp_check(*, name: str, root: Path) -> GateCheck:
    if not root.exists():
        return GateCheck(name=name, passed=False, detail=f"missing directory: {root}")
    found: list[str] = []
    for path in root.rglob("*"):
        if ".tmp." not in path.name:
            continue
        found.append(str(path))
        if len(found) >= 5:
            break
    if found:
        return GateCheck(
            name=name,
            passed=False,
            detail="stale tmp artifacts: " + ", ".join(found),
        )
    return GateCheck(name=name, passed=True, detail=str(root))


def _external_eval_check(*, pretrain_output: Path) -> GateCheck:
    root = pretrain_output / "external_eval"
    if not root.is_dir():
        return GateCheck(
            name="pretrain_external_eval_reports",
            passed=False,
            detail=f"missing directory: {root}",
        )
    step_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not step_dirs:
        return GateCheck(
            name="pretrain_external_eval_reports",
            passed=False,
            detail=f"no step_* reports under {root}",
        )
    required = ("contamination_scan.json",)
    for step_dir in step_dirs:
        if all((step_dir / name).is_file() for name in required):
            return GateCheck(
                name="pretrain_external_eval_reports",
                passed=True,
                detail=str(step_dir),
            )
    return GateCheck(
        name="pretrain_external_eval_reports",
        passed=False,
        detail=f"no complete external eval step under {root}",
    )


def _checkpoint_capability_check(
    *,
    name: str,
    output_root: Path,
    report_name: str,
) -> GateCheck:
    root = output_root / "checkpoint_capability"
    if not root.is_dir():
        return GateCheck(name=name, passed=False, detail=f"missing directory: {root}")
    step_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not step_dirs:
        return GateCheck(name=name, passed=False, detail=f"no step_* reports under {root}")
    for step_dir in step_dirs:
        report_path = step_dir / report_name
        if report_path.is_file():
            return GateCheck(name=name, passed=True, detail=str(report_path))
    return GateCheck(name=name, passed=False, detail=f"no {report_name} under {root}")


def _metric_name_check(
    *,
    name: str,
    metrics_path: Path,
    required_names: tuple[str, ...],
) -> GateCheck:
    try:
        rows = _jsonl_rows(metrics_path)
    except Exception as exc:
        return GateCheck(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}")
    seen = {str(row.get("name", "") or "") for row in rows if "name" in row}
    missing = [metric for metric in required_names if metric not in seen]
    if missing:
        return GateCheck(
            name=name,
            passed=False,
            detail=f"missing metrics in {metrics_path}: {', '.join(missing)}",
        )
    return GateCheck(name=name, passed=True, detail=str(metrics_path))


def _stage_metric_check(
    *,
    name: str,
    metrics_path: Path,
    stage: str,
    field: str,
) -> GateCheck:
    try:
        rows = _jsonl_rows(metrics_path)
    except Exception as exc:
        return GateCheck(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}")
    for row in rows:
        if str(row.get("stage", "") or "") == str(stage) and field in row:
            return GateCheck(name=name, passed=True, detail=str(metrics_path))
    return GateCheck(
        name=name,
        passed=False,
        detail=f"missing stage={stage} field={field} in {metrics_path}",
    )


def _report_passed_check(
    *,
    name: str,
    path: Path,
    expected_kind: str,
    expected_stage: str | None = None,
) -> GateCheck:
    try:
        payload = _json_object(path)
    except Exception as exc:
        return GateCheck(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}")
    kind = str(payload.get("kind", "") or "")
    if kind != str(expected_kind):
        return GateCheck(
            name=name,
            passed=False,
            detail=f"unexpected kind in {path}: {kind!r}",
        )
    if expected_stage is not None:
        stage = str(payload.get("stage", "") or "")
        if stage != str(expected_stage):
            return GateCheck(
                name=name,
                passed=False,
                detail=f"unexpected stage in {path}: {stage!r}",
            )
    summary = payload.get("summary")
    if not isinstance(summary, dict) or not bool(summary.get("all_passed")):
        return GateCheck(
            name=name,
            passed=False,
            detail=f"summary.all_passed != true in {path}",
        )
    return GateCheck(name=name, passed=True, detail=str(path))


def build_release_audit(
    *,
    pretrain_recipe_root: Path,
    pretrain_output: Path,
    pretrain_resume_report: Path,
    soak_report: Path,
    sft_recipe_root: Path | None = None,
    sft_output: Path | None = None,
    sft_resume_report: Path | None = None,
    require_sft: bool = True,
) -> dict[str, Any]:
    checks = [
        _artifact_check(
            name="pretrain_machine_recipe_generated",
            root=pretrain_recipe_root,
            relative_paths=("release_pretrain_machine_recipe.json", "report.json"),
        ),
        _report_passed_check(
            name="pretrain_resume_drill_passed",
            path=pretrain_resume_report,
            expected_kind="resume_drill_audit",
            expected_stage="pretrain",
        ),
        _report_passed_check(
            name="soak_test_passed",
            path=soak_report,
            expected_kind="soak_test_audit",
        ),
        _baseline_table_check(
            name="pretrain_machine_baseline_table",
            path=pretrain_recipe_root / "report.json",
            required_keys=(
                "seq_len",
                "backend",
                "batch_size",
                "accumulation_steps",
                "gradient_checkpointing",
                "loss_chunk_size",
                "tokens_per_s",
                "step_time_s",
                "max_mem_gb",
            ),
        ),
        _artifact_check(
            name="pretrain_release_artifacts",
            root=pretrain_output,
            relative_paths=(
                "metrics.jsonl",
                "run_args.json",
                "run_meta.json",
                "tensorboard",
                PRETRAIN_MACHINE_RECIPE_SUMMARY,
                PRETRAIN_MACHINE_RECIPE_ARTIFACT,
                PRETRAIN_MACHINE_RUNTIME_ARTIFACT,
                PRETRAIN_UPDATE_PROFILE_ARTIFACT,
            ),
        ),
        _stale_tmp_check(name="pretrain_no_stale_tmp", root=pretrain_output),
        _external_eval_check(pretrain_output=pretrain_output),
        _metric_name_check(
            name="pretrain_external_eval_metrics",
            metrics_path=pretrain_output / "metrics.jsonl",
            required_names=(
                "pretrain_contamination_safe_rate",
            ),
        ),
    ]
    if bool(require_sft):
        if sft_recipe_root is None or sft_output is None or sft_resume_report is None:
            raise ValueError("SFT release audit inputs are required when require_sft=True")
        checks.extend(
            [
                _artifact_check(
                    name="sft_machine_recipe_generated",
                    root=sft_recipe_root,
                    relative_paths=("release_sft_recipe.json", "summary.json"),
                ),
                _report_passed_check(
                    name="sft_resume_drill_passed",
                    path=sft_resume_report,
                    expected_kind="resume_drill_audit",
                    expected_stage="sft",
                ),
                _baseline_table_check(
                    name="sft_machine_baseline_table",
                    path=sft_recipe_root / "summary.json",
                    required_keys=(
                        "batch_size",
                        "accumulation_steps",
                        "gradient_checkpointing",
                        "loss",
                        "val_test",
                        "speed",
                    ),
                ),
                _artifact_check(
                    name="sft_release_artifacts",
                    root=sft_output,
                    relative_paths=(
                        "metrics.jsonl",
                        "run_args.json",
                        "run_meta.json",
                        "tensorboard",
                        "machine_recipe_summary.json",
                        "sft_capability_report.json",
                    ),
                ),
                _stale_tmp_check(name="sft_no_stale_tmp", root=sft_output),
                _checkpoint_capability_check(
                    name="sft_checkpoint_capability_reports",
                    output_root=sft_output,
                    report_name="sft_capability_report.json",
                ),
                _stage_metric_check(
                    name="sft_eval_metric_present",
                    metrics_path=sft_output / "metrics.jsonl",
                    stage="sft",
                    field="eval_loss",
                ),
                _stage_metric_check(
                    name="sft_test_metric_present",
                    metrics_path=sft_output / "metrics.jsonl",
                    stage="sft",
                    field="test_loss",
                ),
            ]
        )
    passed = sum(1 for item in checks if bool(item.passed))
    return {
        "kind": "training_release_audit",
        "checks": [asdict(item) for item in checks],
        "summary": {
            "passed": int(passed),
            "failed": int(len(checks) - passed),
            "all_passed": bool(passed == len(checks)),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit training release artifacts against the repository release gate."
    )
    parser.add_argument(
        "--pretrain_recipe_root",
        type=str,
        default="out/pretrain_machine_recipe",
    )
    parser.add_argument(
        "--sft_recipe_root",
        type=str,
        default="out/sft_machine_recipe",
    )
    parser.add_argument(
        "--pretrain_output",
        type=str,
        default="out/release_pretrain",
    )
    parser.add_argument(
        "--sft_output",
        type=str,
        default="out/release_sft",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="out/training_release_audit.json",
    )
    parser.add_argument(
        "--pretrain_resume_report",
        type=str,
        default="out/pretrain_resume_drill.json",
    )
    parser.add_argument(
        "--sft_resume_report",
        type=str,
        default="out/sft_resume_drill.json",
    )
    parser.add_argument(
        "--require_sft",
        type=int,
        default=1,
        choices=[0, 1],
        help="Set to 0 for a pretrain-only release gate before formal pretrain.",
    )
    parser.add_argument(
        "--soak_report",
        type=str,
        default="out/training_soak_test.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = build_release_audit(
        pretrain_recipe_root=Path(str(args.pretrain_recipe_root)).resolve(),
        pretrain_output=Path(str(args.pretrain_output)).resolve(),
        pretrain_resume_report=Path(str(args.pretrain_resume_report)).resolve(),
        soak_report=Path(str(args.soak_report)).resolve(),
        sft_recipe_root=Path(str(args.sft_recipe_root)).resolve(),
        sft_output=Path(str(args.sft_output)).resolve(),
        sft_resume_report=Path(str(args.sft_resume_report)).resolve(),
        require_sft=bool(int(args.require_sft)),
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
        "[AUDIT][RELEASE_TRAINING] "
        f"passed={int(payload['summary']['passed'])} "
        f"failed={int(payload['summary']['failed'])} "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
