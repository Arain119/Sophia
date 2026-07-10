#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.errors import SophiaUsageError
from ml.training.posttrain.release_gate import (
    build_data_signature,
    build_export_signature,
)
from ml.training.posttrain.types import PosttrainStageArgs


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SophiaUsageError(
            "[ERR] unable to load machine-selection summary json.\n"
            f"path={path}\n"
            f"error={type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SophiaUsageError(
            "[ERR] machine-selection summary json must decode to an object.\n"
            f"path={path}"
        )
    return payload


def _select_run(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("selected_runs")
    rows = rows if isinstance(rows, list) else []
    valid = [row for row in rows if isinstance(row, dict)]
    if len(valid) != 1:
        raise SophiaUsageError(
            "[ERR] expected exactly one selected run in summary.json.\n"
            f"found={len(valid)}"
        )
    return dict(valid[0])


def _machine_recipe_payload(
    *,
    stage: str,
    row: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    machine = row.get("machine_selected")
    machine = machine if isinstance(machine, dict) else {}
    release_semantics = row.get("release_semantics")
    release_semantics = release_semantics if isinstance(release_semantics, dict) else {}
    recipe = PosttrainStageArgs.machine_selection_payload_from_mapping(machine)
    if min(recipe.values()) < 0 or int(recipe["batch_size"]) <= 0 or int(
        recipe["accumulation_steps"]
    ) <= 0:
        raise SophiaUsageError(
            "[ERR] selected run is missing a valid machine recipe.\n"
            f"source={source_path}"
        )
    payload = {
        "kind": "posttrain_machine_recipe",
        "stage": str(stage),
        "source_summary": str(source_path.resolve()),
        "selected_name": str(row.get("name", "") or ""),
        "release_semantics": release_semantics,
        "machine_recipe": recipe,
    }
    run_dir_value = str(row.get("output_dir", "") or "").strip()
    if not run_dir_value:
        raise SophiaUsageError(
            "[ERR] selected run is missing output_dir; cannot export a signed release machine recipe.\n"
            f"source={source_path}"
        )
    run_dir = Path(run_dir_value).resolve()
    run_args_path = run_dir / "run_args.json"
    if not run_args_path.is_file():
        raise SophiaUsageError(
            "[ERR] selected run is missing run_args.json; cannot export a signed release machine recipe.\n"
            f"run_dir={run_dir}"
        )
    run_args = _json_object(run_args_path)
    machine_signature = run_args.get("_sophia_machine_signature")
    if not isinstance(machine_signature, dict) or not machine_signature:
        raise SophiaUsageError(
            "[ERR] selected run is missing _sophia_machine_signature in run_args.json.\n"
            f"path={run_args_path}"
        )
    payload["machine_signature"] = dict(machine_signature)
    export_dir = str(run_args.get("export_dir", "") or "").strip()
    train_data = str(run_args.get("train_data", "") or "").strip()
    eval_data = str(run_args.get("eval_data", "") or "").strip()
    test_data = str(run_args.get("test_data", "") or "").strip()
    if not export_dir:
        raise SophiaUsageError(
            "[ERR] selected run is missing export_dir in run_args.json.\n"
            f"path={run_args_path}"
        )
    if not train_data:
        raise SophiaUsageError(
            "[ERR] selected run is missing train_data in run_args.json.\n"
            f"path={run_args_path}"
        )
    payload["export_signature"] = build_export_signature(export_dir=export_dir)
    payload["data_signature"] = build_data_signature(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a normalized machine-recipe json from an SFT machine-selection summary."
    )
    parser.add_argument("--summary_json", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    parser.add_argument("--stage", required=True, choices=["sft"], type=str)
    return parser


@dataclass(frozen=True)
class ExportMachineRecipeConfig:
    summary_json: str
    output_json: str
    stage: str

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> ExportMachineRecipeConfig:
        return cls(
            summary_json=str(args.summary_json),
            output_json=str(args.output_json),
            stage=str(args.stage),
        )


def main() -> None:
    config = ExportMachineRecipeConfig.from_namespace(_build_parser().parse_args())
    source_path = Path(str(config.summary_json)).resolve()
    if not source_path.is_file():
        raise SophiaUsageError(
            "[ERR] --summary_json path does not exist.\n"
            f"path={source_path}"
        )
    payload = _json_object(source_path)
    row = _select_run(payload)
    out = _machine_recipe_payload(
        stage=str(config.stage),
        row=row,
        source_path=source_path,
    )
    out_path = Path(str(config.output_json)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] wrote post-train machine recipe: {out_path}", flush=True)


if __name__ == "__main__":
    main()
