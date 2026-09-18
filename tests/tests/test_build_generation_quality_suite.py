from __future__ import annotations

import json
from pathlib import Path

from ml.tooling.scripts import build_generation_quality_suite as mod
from ml.tooling.scripts.eval_generation_quality import load_suite


def test_build_cases_are_balanced_unique_and_complete() -> None:
    rows = mod.build_cases()

    assert len(rows) == 2_000
    assert len({row["id"] for row in rows}) == 2_000
    assert sum(row["language"] == "zh" for row in rows) == 1_000
    assert sum(row["language"] == "en" for row in rows) == 1_000
    assert {row["capability"] for row in rows} == {
        "code",
        "fact",
        "logic",
        "math",
        "science",
    }
    assert all(len(str(row["raw_prompt"])) >= 40 for row in rows)
    assert all(len(str(row["chat_prompt"])) >= 40 for row in rows)


def test_build_suite_writes_eval_and_decontamination_artifacts(tmp_path: Path) -> None:
    suite = tmp_path / "suite.jsonl"
    signatures = tmp_path / "signatures.jsonl"
    report_path = tmp_path / "report.json"

    report = mod.build_suite(
        suite_path=suite,
        signatures_path=signatures,
        report_path=report_path,
    )

    assert report["status"] == "complete"
    assert report["case_count"] == 2_000
    assert report["signature_count"] == 4_000
    assert len(load_suite(str(suite))) == 2_000
    signature_rows = [
        json.loads(line) for line in signatures.read_text(encoding="utf-8").splitlines()
    ]
    assert len(signature_rows) == 4_000
    assert all(len(row["prompt"]) >= 24 for row in signature_rows)
    assert json.loads(report_path.read_text(encoding="utf-8"))["suite_sha256"] == report[
        "suite_sha256"
    ]
