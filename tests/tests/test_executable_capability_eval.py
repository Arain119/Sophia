from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.tooling.scripts import eval_executable_capabilities as mod


def test_restricted_code_expression_executes_expected_tests() -> None:
    passed = mod.evaluate_code_case(
        {"id": "zh_code_000", "response": "x + 2"}
    )
    rejected = mod.evaluate_code_case(
        {"id": "en_code_000", "response": "__import__('os').system('id')"}
    )

    assert passed["passed"] is True
    assert rejected["passed"] is False
    assert "only max" in rejected["error"] or "disallowed" in rejected["error"]


def test_executable_report_requires_full_pinned_coverage(tmp_path: Path) -> None:
    rows = []
    for language in ("zh", "en"):
        for index in range(100):
            constant = 2 + index % 25
            operation = index // 25
            expression = (
                f"x + {constant}"
                if operation == 0
                else f"x * {constant}"
                if operation == 1
                else f"x % {constant} == 0"
                if operation == 2
                else f"max(x, {constant})"
            )
            rows.append(
                {
                    "id": f"{language}_code_{index:03d}",
                    "capability": "code",
                    "response": expression,
                }
            )
    rows.extend(
        {
            "id": f"math_{index:04d}",
            "capability": "math",
            "answer_match": True,
        }
        for index in range(1200)
    )
    report_path = tmp_path / "generation.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "sophia_generation_quality_v1",
                "prompt_format": "raw",
                "suite_sha256": "a" * 64,
                "cases": rows,
            }
        ),
        encoding="utf-8",
    )

    report = mod.evaluate_generation_report(
        generation_report_path=report_path,
        expected_suite_sha256="a" * 64,
    )

    assert report["summary"]["math_exact_match_rate"] == pytest.approx(1.0)
    assert report["summary"]["code_pass_at_1"] == pytest.approx(1.0)
    assert report["sandbox"]["imports_attributes_subscripts_and_statements_allowed"] is False
    assert report["evaluation_binding"] is None


def test_executable_release_binding_hashes_files(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    checkpoint = tmp_path / "checkpoint.bin"
    model = tmp_path / "model.bin"
    protocol.write_bytes(b"protocol")
    checkpoint.write_bytes(b"checkpoint")
    model.write_bytes(b"model")
    binding = mod._evaluation_binding(
        protocol_json=protocol,
        checkpoint=checkpoint,
        checkpoint_sha256=None,
        eval_export_model=model,
        evaluation_asset_sha256="asset",
    )
    assert binding["protocol_sha256"] == hashlib.sha256(b"protocol").hexdigest()
    assert binding["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert binding["eval_export_model_sha256"] == hashlib.sha256(b"model").hexdigest()
