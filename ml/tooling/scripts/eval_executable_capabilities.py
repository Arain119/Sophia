from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ml.core.common.io import write_json_atomic


REPORT_SCHEMA = "sophia_executable_capability_eval_v1"
_CODE_ID_RE = re.compile(r"^(?:zh|en)_code_(\d{3})$")
_ALLOWED_NODES = (
    ast.Expression,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.BinOp,
    ast.Add,
    ast.Mult,
    ast.Mod,
    ast.Compare,
    ast.Eq,
    ast.Call,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing evaluation binding file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_binding(
    *,
    protocol_json: str | Path | None,
    checkpoint: str | Path | None,
    checkpoint_sha256: str | None,
    eval_export_model: str | Path | None,
    evaluation_asset_sha256: str,
) -> dict[str, str | None] | None:
    values = (protocol_json, checkpoint, checkpoint_sha256, eval_export_model)
    if not any(value is not None and str(value).strip() for value in values):
        return None
    if not all(value is not None and str(value).strip() for value in (protocol_json, checkpoint, eval_export_model)):
        raise ValueError(
            "release-bound evaluation requires --protocol_json, --checkpoint, and --eval_export_model"
        )
    protocol_path = Path(str(protocol_json)).expanduser().resolve()
    checkpoint_path = Path(str(checkpoint)).expanduser().resolve()
    model_path = Path(str(eval_export_model)).expanduser().resolve()
    protocol_digest = _sha256(protocol_path)
    checkpoint_digest = _sha256(checkpoint_path)
    model_digest = _sha256(model_path)
    if checkpoint_sha256 and str(checkpoint_sha256).lower() != checkpoint_digest:
        raise ValueError("checkpoint_sha256 does not match checkpoint")
    return {
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "eval_export_model_path": str(model_path),
        "eval_export_model_sha256": model_digest,
        "evaluation_asset_sha256": str(evaluation_asset_sha256),
    }


def _candidate_expression(response: str) -> str:
    text = str(response).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return_match = re.search(r"\breturn\s+([^\r\n]+)", text)
    if return_match is not None:
        return return_match.group(1).strip()
    return text.splitlines()[0].strip() if text else ""


def _validated_expression(expression: str) -> Any:
    tree = ast.parse(str(expression), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed AST node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in {"x", "max"}:
            raise ValueError(f"disallowed name: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id != "max":
                raise ValueError("only max(...) calls are allowed")
            if node.keywords:
                raise ValueError("keyword arguments are not allowed")
    return compile(tree, "<restricted-generated-expression>", "eval")


def _expected_code_value(*, index: int, value: int) -> int | bool:
    operation = int(index) // 25
    constant = 2 + int(index) % 25
    if operation == 0:
        return value + constant
    if operation == 1:
        return value * constant
    if operation == 2:
        return value % constant == 0
    if operation == 3:
        return max(value, constant)
    raise ValueError(f"unsupported generated code case index: {index}")


def evaluate_code_case(row: dict[str, Any]) -> dict[str, Any]:
    case_id = str(row.get("id") or "")
    match = _CODE_ID_RE.fullmatch(case_id)
    if match is None:
        raise ValueError(f"unsupported code case id: {case_id}")
    index = int(match.group(1))
    expression = _candidate_expression(str(row.get("response") or ""))
    test_inputs = (-17, -1, 0, 1, 7, 31)
    error = ""
    passed = False
    outputs: list[dict[str, Any]] = []
    try:
        compiled = _validated_expression(expression)
        for value in test_inputs:
            actual = eval(
                compiled,
                {"__builtins__": {}, "max": max},
                {"x": value},
            )
            expected = _expected_code_value(index=index, value=value)
            outputs.append(
                {"input": value, "actual": actual, "expected": expected}
            )
        passed = all(item["actual"] == item["expected"] for item in outputs)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "id": case_id,
        "expression": expression,
        "passed": bool(passed),
        "error": error,
        "tests": outputs,
    }


def evaluate_generation_report(
    *,
    generation_report_path: Path,
    expected_suite_sha256: str,
    protocol_json: str | Path | None = None,
    checkpoint: str | Path | None = None,
    checkpoint_sha256: str | None = None,
    eval_export_model: str | Path | None = None,
) -> dict[str, Any]:
    generation = _load_json(generation_report_path)
    if generation.get("schema") != "sophia_generation_quality_v1":
        raise ValueError("unsupported generation report")
    prompt_format = str(generation.get("prompt_format", "") or "")
    if prompt_format not in {"raw", "chat"}:
        raise ValueError("executable capability evaluation requires raw or chat prompts")
    if generation.get("suite_sha256") != str(expected_suite_sha256):
        raise ValueError("generation report suite sha256 mismatch")
    rows = generation.get("cases")
    if not isinstance(rows, list):
        raise ValueError("generation report has no cases")
    code_rows = [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("capability")) == "code"
    ]
    math_rows = [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("capability")) == "math"
    ]
    if len(code_rows) != 200 or len(math_rows) != 1200:
        raise ValueError(
            f"unexpected executable suite coverage: code={len(code_rows)} math={len(math_rows)}"
        )
    code_results = [evaluate_code_case(row) for row in code_rows]
    math_passed = sum(bool(row.get("answer_match")) for row in math_rows)
    code_passed = sum(bool(row["passed"]) for row in code_results)
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "prompt_format": prompt_format,
        "evaluation_binding": _evaluation_binding(
            protocol_json=protocol_json,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            eval_export_model=eval_export_model,
            evaluation_asset_sha256=str(expected_suite_sha256),
        ),
        "generation_report_path": str(generation_report_path.resolve()),
        "generation_report_sha256": _sha256(generation_report_path),
        "suite_sha256": str(expected_suite_sha256),
        "sandbox": {
            "kind": "ast_whitelist_expression_evaluator",
            "allowed_names": ["x", "max"],
            "imports_attributes_subscripts_and_statements_allowed": False,
        },
        "summary": {
            "math_cases": len(math_rows),
            "math_exact_matches": math_passed,
            "math_exact_match_rate": math_passed / float(len(math_rows)),
            "code_cases": len(code_rows),
            "code_passed": code_passed,
            "code_pass_at_1": code_passed / float(len(code_rows)),
        },
        "code_cases": code_results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely execute the pinned math/code cases from a generation report.")
    parser.add_argument("--generation-report", required=True)
    parser.add_argument("--suite-report", default="configs/eval/generation_quality_report.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol_json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint_sha256")
    parser.add_argument("--eval_export_model")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suite_report = _load_json(Path(args.suite_report))
    report = evaluate_generation_report(
        generation_report_path=Path(args.generation_report).expanduser().resolve(),
        expected_suite_sha256=str(suite_report.get("suite_sha256") or ""),
        protocol_json=args.protocol_json,
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        eval_export_model=args.eval_export_model,
    )
    write_json_atomic(args.output, report, ensure_ascii=False, sort_keys=True, make_parents=True)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
