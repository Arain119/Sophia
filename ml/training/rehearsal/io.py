from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ml.training.rehearsal.defaults import REPO_ROOT
from ml.training.rehearsal.plan import RehearsalPlan


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def plan_payload(plan: RehearsalPlan) -> dict[str, Any]:
    return {
        "kind": "local_rehearsal_plan",
        "repo_root": str(REPO_ROOT),
        "output_root": str(plan.output_root),
        "progress_interval_seconds": float(plan.progress_interval_seconds),
        "stages": [
            {
                "name": stage.name,
                "type": stage.stage_type,
                "reason": stage.reason,
                "output_dir": stage.output_dir,
                "argv": list(stage.argv),
            }
            for stage in plan.stages
        ],
    }


__all__ = ["append_jsonl", "plan_payload", "write_json"]
