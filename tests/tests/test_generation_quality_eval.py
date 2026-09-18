from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.tooling.scripts import eval_generation_quality as mod
from ml.tooling.scripts.eval_generation_quality import (
    aggregate_results,
    analyze_response,
    load_suite,
    repeated_ngram_ratio,
    response_language_match,
    select_cases,
)


def test_analyze_response_detects_match_echo_and_abnormal_path() -> None:
    matched = analyze_response(
        prompt="The capital of France is",
        response="Paris.",
        expected_any=["Paris"],
    )
    echoed = analyze_response(
        prompt="Explain photosynthesis.",
        response="Explain photosynthesis.",
        expected_any=["sunlight"],
    )
    abnormal = analyze_response(
        prompt="test",
        response='<location filename="../src/http/client.cpp"/>',
        expected_any=["ok"],
    )

    assert matched["answer_match"] is True
    assert echoed["prompt_echo"] is True
    assert abnormal["abnormal_markup_or_path"] is True


def test_repeated_ngram_ratio_flags_degenerate_generation() -> None:
    assert repeated_ngram_ratio("a b c d e f g") == pytest.approx(0.0)
    assert repeated_ngram_ratio("foo foo foo foo foo foo foo foo") >= 0.25


def test_response_language_match_handles_bilingual_and_neutral_answers() -> None:
    assert response_language_match(
        response="北京", expected_any=["北京"], language="zh"
    ) is True
    assert response_language_match(
        response="Beijing", expected_any=["北京"], language="zh"
    ) is False
    assert response_language_match(
        response="42", expected_any=["42"], language="zh"
    ) is True
    assert response_language_match(
        response="H2O", expected_any=["H2O"], language="zh"
    ) is True
    assert response_language_match(
        response="答案是北京", expected_any=["Beijing"], language="en"
    ) is False
    assert response_language_match(
        response="return x + 1",
        expected_any=["return x + 1"],
        language="zh",
        mode="exempt",
    ) is None


def test_load_suite_rejects_duplicate_ids(tmp_path) -> None:
    row = {
        "id": "duplicate",
        "language": "en",
        "capability": "fact",
        "raw_prompt": "A",
        "chat_prompt": "B",
        "expected_any": ["C"],
    }
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_suite(str(suite))


def test_select_cases_is_utf8_balanced_and_unique() -> None:
    cases = load_suite("configs/eval/generation_quality.jsonl")
    selected = select_cases(cases, max_cases=100, seed=2026)

    assert len(selected) == 100
    assert len({case.case_id for case in selected}) == 100
    assert all(any("\u4e00" <= char <= "\u9fff" for char in case.prompt("raw"))
               for case in selected if case.language == "zh")
    counts = {}
    for case in selected:
        key = (case.language, case.capability)
        counts[key] = counts.get(key, 0) + 1
    assert set(counts) == {
        (language, capability)
        for language in ("zh", "en")
        for capability in ("math", "logic", "code", "fact", "science")
    }
    assert len(select_cases(cases, max_cases=1000, seed=2026)) == 1000


def test_default_suite_is_the_pinned_two_thousand_case_suite() -> None:
    from ml.tooling.scripts.eval_generation_quality import build_parser

    parsed = build_parser().parse_args(
        ["--export_dir", "model", "--output", "report.json"]
    )

    assert str(parsed.suite).endswith("generation_quality.jsonl")
    assert parsed.kda_backend == "auto"


def test_reference_backend_is_not_a_release_path() -> None:
    from types import SimpleNamespace

    args = SimpleNamespace(
        kda_backend="reference",
        protocol_json="protocol.json",
        checkpoint="checkpoint.pt",
        eval_export_model="model",
    )
    with pytest.raises(ValueError, match="diagnostic-only"):
        mod.run_evaluation(args)


def test_release_binding_hashes_files_and_requires_complete_group(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    checkpoint = tmp_path / "checkpoint.bin"
    model = tmp_path / "eval-model.bin"
    protocol.write_bytes(b"protocol")
    checkpoint.write_bytes(b"checkpoint")
    model.write_bytes(b"model")

    binding = mod._evaluation_binding(
        protocol_json=protocol,
        checkpoint=checkpoint,
        checkpoint_sha256=None,
        eval_export_model=model,
        evaluation_asset_sha256="asset-sha",
    )

    assert binding["protocol_path"] == str(protocol.resolve())
    assert binding["protocol_sha256"] == hashlib.sha256(b"protocol").hexdigest()
    assert binding["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert binding["eval_export_model_sha256"] == hashlib.sha256(b"model").hexdigest()
    assert binding["evaluation_asset_sha256"] == "asset-sha"
    with pytest.raises(ValueError, match="requires"):
        mod._evaluation_binding(
            protocol_json=protocol,
            checkpoint=None,
            checkpoint_sha256=None,
            eval_export_model=model,
            evaluation_asset_sha256="asset-sha",
        )


def test_aggregate_results_groups_language_and_capability() -> None:
    base = {
        "empty_response": False,
        "prompt_echo": False,
        "abnormal_markup_or_path": False,
        "high_repetition": False,
        "first_token_top1_id": 1,
        "eos_token_id": 2,
        "first_token_eos_probability": 0.1,
        "language_match": True,
    }
    rows = [
        {**base, "language": "zh", "capability": "fact", "answer_match": True},
        {**base, "language": "en", "capability": "fact", "answer_match": False},
    ]

    report = aggregate_results(rows)

    assert report["overall"]["answer_accuracy"] == pytest.approx(0.5)
    assert report["by_language"]["zh"]["answer_accuracy"] == pytest.approx(1.0)
    assert report["by_capability"]["fact"]["cases"] == 2
    assert report["overall"]["language_match_rate"] == pytest.approx(1.0)
