from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.tooling.scripts.summarize_pretrain_eval_trajectory import (
    build_trajectory,
)


def _write(path: Path, payload: dict[str, object]) -> str:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_trajectory_preserves_missing_reports_and_utf8(tmp_path: Path) -> None:
    generation = _write(
        tmp_path / "generation.json",
        {
            "schema": "sophia_generation_quality_v1",
            "suite_sha256": "a" * 64,
            "selection": {"evaluated_case_count": 100},
            "summary": {
                "overall": {"answer_accuracy": 0.25, "high_repetition_rate": 0.1},
                "by_capability": {"fact": {"cases": 10, "answer_accuracy": 0.5}},
            },
        },
    )
    perplexity = _write(
        tmp_path / "perplexity.json",
        {
            "schema": "sophia_pretrain_perplexity_eval_v1",
            "datasets": {"中文验证": {"loss": 3.5, "perplexity": 33.1, "manifest_sha1": "b" * 40}},
        },
    )
    report = build_trajectory(
        generation=[f"6000={generation}"],
        perplexity=[f"6000={perplexity}"],
        executable=[],
        chinese=[],
    )
    row = report["steps"][0]
    assert row["generation"]["metrics"]["cases"] == 100
    assert row["perplexity"]["metrics"]["datasets"]["中文验证"]["perplexity"] == 33.1
    assert row["executable"] is None


def test_trajectory_rejects_duplicate_step(tmp_path: Path) -> None:
    path = _write(tmp_path / "r.json", {"schema": "x"})
    with pytest.raises(ValueError, match="duplicate"):
        build_trajectory(
            generation=[f"1={path}", f"1={path}"],
            perplexity=[],
            executable=[],
            chinese=[],
        )
