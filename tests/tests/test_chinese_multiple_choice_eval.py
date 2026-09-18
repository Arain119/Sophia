from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ml.tooling.scripts import eval_chinese_multiple_choice as mod


def _case(case_id: str, *, benchmark: str = "ceval", split: str = "dev"):
    return {
        "id": case_id,
        "benchmark": benchmark,
        "subject": "math",
        "split": split,
        "question": "一加一等于多少？",
        "choices": {"A": "一", "B": "二", "C": "三", "D": "四"},
        "answer": "B",
    }


def test_build_prompt_contains_question_choices_and_answer_cue() -> None:
    prompt = mod.build_prompt(_case("one"))

    assert "一加一等于多少" in prompt
    assert "A. 一" in prompt
    assert prompt.endswith("答案：")


def test_select_cases_is_deterministic_and_split_scoped() -> None:
    cases = [
        *(_case(f"dev-{index}") for index in range(10)),
        _case("test", split="test"),
        _case("cmmlu", benchmark="cmmlu"),
    ]

    first = mod.select_cases(
        cases,
        splits={"dev"},
        max_cases_per_benchmark=3,
        seed=7,
    )
    second = mod.select_cases(
        cases,
        splits={"dev"},
        max_cases_per_benchmark=3,
        seed=7,
    )

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 4
    assert all(row["split"] == "dev" for row in first)


def test_release_binding_hashes_files_and_rejects_partial_group(tmp_path: Path) -> None:
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
    assert binding["evaluation_asset_sha256"] == "asset"
    with pytest.raises(ValueError, match="requires"):
        mod._evaluation_binding(
            protocol_json=protocol,
            checkpoint=None,
            checkpoint_sha256=None,
            eval_export_model=model,
            evaluation_asset_sha256="asset",
        )


def test_aggregate_predictions_reports_accuracy_and_distribution() -> None:
    rows = [
        {
            **_case("one"),
            "prediction": "B",
            "correct": True,
        },
        {
            **_case("two"),
            "prediction": "A",
            "correct": False,
        },
    ]

    summary = mod.aggregate_predictions(rows)

    assert summary["overall"]["accuracy"] == 0.5
    assert summary["overall"]["prediction_distribution"] == {"A": 1, "B": 1}
