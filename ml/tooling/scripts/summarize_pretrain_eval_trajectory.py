"""Combine checkpoint evaluation reports into one comparable trajectory.

This is a diagnostic report, not a model-selection gate.  It deliberately
preserves each report's own dataset/suite hashes and records missing reports
instead of inventing zeros.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.core.common.io import write_json_atomic


SCHEMA = "sophia_pretrain_eval_trajectory_v1"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _parse_spec(value: str) -> tuple[int, Path]:
    step, separator, path = str(value).partition("=")
    if not separator or not step.isdigit() or not path.strip():
        raise ValueError(f"report must use STEP=/path/to/report.json: {value!r}")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing report: {resolved}")
    return int(step), resolved


def _rate(payload: dict[str, Any], dotted: str) -> float | None:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _generation_row(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    overall = summary.get("overall") if isinstance(summary.get("overall"), dict) else {}
    capabilities = summary.get("by_capability")
    capability_rows: dict[str, Any] = {}
    if isinstance(capabilities, dict):
        for name, values in sorted(capabilities.items()):
            if isinstance(values, dict):
                capability_rows[str(name)] = {
                    "cases": values.get("cases"),
                    "answer_accuracy": values.get("answer_accuracy"),
                    "high_repetition_rate": values.get("high_repetition_rate"),
                    "prompt_echo_rate": values.get("prompt_echo_rate"),
                }
    selection = report.get("selection")
    return {
        "schema": report.get("schema"),
        "suite_sha256": report.get("suite_sha256"),
        "cases": (
            selection.get("evaluated_case_count")
            if isinstance(selection, dict)
            else overall.get("cases")
        ),
        "overall": {
            "answer_accuracy": overall.get("answer_accuracy"),
            "high_repetition_rate": overall.get("high_repetition_rate"),
            "prompt_echo_rate": overall.get("prompt_echo_rate"),
            "empty_rate": overall.get("empty_rate"),
            "first_token_eos_rate": overall.get("first_token_eos_rate"),
        },
        "by_capability": capability_rows,
    }


def _perplexity_row(report: dict[str, Any]) -> dict[str, Any]:
    datasets = report.get("datasets")
    result: dict[str, Any] = {}
    if isinstance(datasets, dict):
        for name, values in sorted(datasets.items()):
            if isinstance(values, dict):
                result[str(name)] = {
                    "loss": values.get("loss"),
                    "perplexity": values.get("perplexity"),
                    "manifest_sha1": values.get("manifest_sha1"),
                }
    return {"schema": report.get("schema"), "datasets": result}


def _executable_row(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    return {
        "schema": report.get("schema"),
        "math_exact_match_rate": summary.get("math_exact_match_rate"),
        "code_pass_at_1": summary.get("code_pass_at_1"),
    }


def _chinese_row(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    overall = summary.get("overall")
    overall = overall if isinstance(overall, dict) else {}
    return {
        "schema": report.get("schema"),
        "accuracy": overall.get("accuracy"),
        "cases": overall.get("cases"),
        "benchmark_sha256": report.get("benchmark_sha256"),
    }


def build_trajectory(
    *,
    generation: list[str],
    perplexity: list[str],
    executable: list[str],
    chinese: list[str],
) -> dict[str, Any]:
    groups = {
        "generation": generation,
        "perplexity": perplexity,
        "executable": executable,
        "chinese_multiple_choice": chinese,
    }
    parsed: dict[str, dict[int, Path]] = {}
    for name, values in groups.items():
        rows: dict[int, Path] = {}
        for value in values:
            step, path = _parse_spec(value)
            if step in rows:
                raise ValueError(f"duplicate {name} report for step {step}")
            rows[step] = path
        parsed[name] = rows
    steps = sorted({step for rows in parsed.values() for step in rows})
    checkpoints: list[dict[str, Any]] = []
    for step in steps:
        row: dict[str, Any] = {"step": step}
        for name, reports in parsed.items():
            path = reports.get(step)
            if path is None:
                row[name] = None
                continue
            payload = _load(path)
            row[name] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "metrics": (
                    _generation_row(payload)
                    if name == "generation"
                    else _perplexity_row(payload)
                    if name == "perplexity"
                    else _executable_row(payload)
                    if name == "executable"
                    else _chinese_row(payload)
                ),
            }
        checkpoints.append(row)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "selection_policy": "diagnostic trajectory only; no checkpoint selection or thresholding",
        "steps": checkpoints,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize pretraining evaluation reports by checkpoint step")
    parser.add_argument("--generation", action="append", default=[])
    parser.add_argument("--perplexity", action="append", default=[])
    parser.add_argument("--executable", action="append", default=[])
    parser.add_argument("--chinese", action="append", default=[])
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_trajectory(
        generation=list(args.generation),
        perplexity=list(args.perplexity),
        executable=list(args.executable),
        chinese=list(args.chinese),
    )
    write_json_atomic(args.output, report, ensure_ascii=False, sort_keys=True, make_parents=True)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
