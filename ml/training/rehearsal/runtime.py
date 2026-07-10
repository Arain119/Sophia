from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ml.errors import SophiaUsageError
from ml.training.rehearsal.config import RehearsalConfig
from ml.training.rehearsal.defaults import REPO_ROOT
from ml.training.rehearsal.io import append_jsonl, plan_payload, write_json
from ml.training.rehearsal.plan import (
    RehearsalPlan,
    build_rehearsal_plan,
    validate_posttrain_stage_inputs,
)
from ml.training.rehearsal.process import run_stage_process


def run_rehearsal(plan: RehearsalPlan) -> dict[str, Any]:
    output_root = Path(plan.output_root)
    log_root = output_root / "logs"
    events_path = output_root / "rehearsal_events.jsonl"
    plan_path = output_root / "rehearsal_plan.json"
    report_path = output_root / "rehearsal_report.json"

    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, plan_payload(plan))

    report: dict[str, Any] = {
        "kind": "local_rehearsal_report",
        "repo_root": str(REPO_ROOT),
        "output_root": str(output_root),
        "started_at": float(time.time()),
        "completed_at": None,
        "status": "running",
        "stages": [],
    }
    write_json(report_path, report)

    active_stage_record: dict[str, Any] | None = None
    try:
        for stage in plan.stages:
            stage_record: dict[str, Any] = {
                "name": stage.name,
                "type": stage.stage_type,
                "output_dir": stage.output_dir,
                "argv": list(stage.argv),
                "reason": stage.reason,
                "started_at": float(time.time()),
                "completed_at": None,
                "status": "pending",
            }
            active_stage_record = stage_record
            if stage.stage_type == "skip":
                stage_record["status"] = "skipped"
                stage_record["completed_at"] = float(time.time())
                report["stages"].append(stage_record)
                append_jsonl(
                    events_path,
                    {
                        "stage": stage.name,
                        "status": "skipped",
                        "reason": stage.reason,
                        "time": float(time.time()),
                    },
                )
                write_json(report_path, report)
                active_stage_record = None
                continue

            stdout_log = log_root / f"{stage.name}.stdout.log"
            stderr_log = log_root / f"{stage.name}.stderr.log"
            stage_record["stdout_log"] = str(stdout_log)
            stage_record["stderr_log"] = str(stderr_log)
            report["stages"].append(stage_record)
            try:
                validate_posttrain_stage_inputs(stage)
            except SophiaUsageError as exc:
                stage_record["status"] = "failed"
                stage_record["reason"] = str(exc)
                stage_record["completed_at"] = float(time.time())
                report["status"] = "failed"
                report["completed_at"] = float(time.time())
                append_jsonl(
                    events_path,
                    {
                        "stage": stage.name,
                        "status": "failed_preflight",
                        "reason": str(exc),
                        "time": float(time.time()),
                    },
                )
                write_json(report_path, report)
                raise
            stage_record["status"] = "running"
            append_jsonl(
                events_path,
                {
                    "stage": stage.name,
                    "status": "start",
                    "time": float(time.time()),
                    "argv": list(stage.argv),
                },
            )
            write_json(report_path, report)
            print(f"[REHEARSAL] start {stage.name} -> {stage.output_dir}", flush=True)
            with stdout_log.open("w", encoding="utf-8") as stdout_handle:
                with stderr_log.open("w", encoding="utf-8") as stderr_handle:
                    try:
                        returncode = run_stage_process(
                            stage=stage,
                            stdout_handle=stdout_handle,
                            stderr_handle=stderr_handle,
                            progress_interval_seconds=float(
                                plan.progress_interval_seconds
                            ),
                            events_path=events_path,
                            stdout_log=stdout_log,
                            stderr_log=stderr_log,
                        )
                    except KeyboardInterrupt as exc:
                        stage_record["status"] = "interrupted"
                        stage_record["completed_at"] = float(time.time())
                        report["status"] = "interrupted"
                        report["completed_at"] = float(time.time())
                        append_jsonl(
                            events_path,
                            {
                                "stage": stage.name,
                                "status": "interrupted",
                                "time": float(time.time()),
                                "stdout_log": str(stdout_log),
                                "stderr_log": str(stderr_log),
                            },
                        )
                        write_json(report_path, report)
                        raise SophiaUsageError(
                            "[ERR] rehearsal interrupted during stage: "
                            f"{stage.name}. See {stdout_log} and {stderr_log}."
                        ) from exc
            stage_record["returncode"] = int(returncode)
            stage_record["completed_at"] = float(time.time())
            if int(returncode) != 0:
                stage_record["status"] = "failed"
                report["status"] = "failed"
                report["completed_at"] = float(time.time())
                append_jsonl(
                    events_path,
                    {
                        "stage": stage.name,
                        "status": "failed",
                        "returncode": int(returncode),
                        "time": float(time.time()),
                    },
                )
                write_json(report_path, report)
                raise SophiaUsageError(
                    "[ERR] rehearsal stage failed: "
                    f"{stage.name} (returncode={returncode}). "
                    f"See {stdout_log} and {stderr_log}."
                )
            stage_record["status"] = "completed"
            append_jsonl(
                events_path,
                {
                    "stage": stage.name,
                    "status": "completed",
                    "time": float(time.time()),
                },
            )
            write_json(report_path, report)
            print(f"[REHEARSAL] done {stage.name}", flush=True)
            active_stage_record = None
        report["status"] = "completed"
        report["completed_at"] = float(time.time())
        write_json(report_path, report)
        return report
    except BaseException:
        if active_stage_record is not None and str(active_stage_record.get("status") or "") == "running":
            active_stage_record["status"] = "failed"
            active_stage_record["completed_at"] = float(time.time())
        if report.get("completed_at") is None:
            report["completed_at"] = float(time.time())
        if str(report.get("status") or "") == "running":
            report["status"] = "failed"
        write_json(report_path, report)
        raise


def run(config: RehearsalConfig) -> None:
    plan = build_rehearsal_plan(config)
    run_rehearsal(plan)


__all__ = ["run", "run_rehearsal"]
