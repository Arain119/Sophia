from __future__ import annotations

import json

from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from ml.tooling.scripts.data import audit_tokenizer_quality as mod


def _write_tokenizer(path) -> None:
    path.mkdir()
    tokenizer = Tokenizer(
        models.WordLevel(
            vocab={"<unk>": 0, "hello": 1, "world": 2},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "chat_template.jinja").write_text("", encoding="utf-8")


def test_audit_reports_domain_fertility_and_unknowns(tmp_path) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    _write_tokenizer(tokenizer_dir)
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "schema": "test",
                "domains": {
                    "known": ["hello world"],
                    "unknown": ["missing"],
                },
            }
        ),
        encoding="utf-8",
    )

    report = mod.audit_tokenizer(
        tokenizer_dir=tokenizer_dir,
        suite_path=suite,
        warmup=0,
        iterations=1,
    )

    assert report["schema"] == mod.REPORT_SCHEMA
    assert report["model"]["vocab_size"] == 3
    assert report["domains"]["known"]["tokens"] == 2
    assert report["domains"]["known"]["roundtrip_failures"] == 0
    assert report["domains"]["unknown"]["unk_tokens"] == 1
    assert report["summary"]["unk_tokens"] == 1
    assert report["summary"]["provenance_present"] is False
    assert report["bundle"]["files"]["tokenizer.json"]["present"] is True


def test_suite_rejects_empty_domains(tmp_path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text('{"domains":{"empty":[]}}', encoding="utf-8")

    try:
        mod._load_suite(suite)
    except ValueError as exc:
        assert "must be non-empty" in str(exc)
    else:
        raise AssertionError("empty tokenizer suite domain was accepted")


def test_audit_can_bound_details_without_changing_totals(tmp_path) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    _write_tokenizer(tokenizer_dir)
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps({"domains": {"known": ["hello", "world"]}}),
        encoding="utf-8",
    )

    report = mod.audit_tokenizer(
        tokenizer_dir=tokenizer_dir,
        suite_path=suite,
        warmup=0,
        iterations=1,
        max_details_per_domain=1,
    )

    assert report["domains"]["known"]["samples"] == 2
    assert len(report["domains"]["known"]["details"]) == 1
