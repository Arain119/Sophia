from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ml.data.pretrain_document_admission import (
    load_post_clean_document_admission,
)
from ml.data.pretrain_filters import repeated_sentence_excess


def _policy() -> dict[str, object]:
    return {
        "post_clean_document_admission": {
            "exclude_legacy_exact_overlap": True,
            "exclude_legacy_near_overlap": False,
            "repeated_sentence_rules": {
                "books": {
                    "min_sentence_chars": 20,
                    "repeats": 2,
                    "minimum_repeated_chars": 80,
                    "minimum_repeated_fraction": 0.2,
                }
            },
        }
    }


def test_repeated_sentence_excess_counts_only_extra_occurrences() -> None:
    sentence = "A meaningful repeated sentence with enough explanatory content"
    text = f"{sentence}. {sentence}. A separate conclusion remains unique."

    repeated, fraction = repeated_sentence_excess(text)

    assert repeated == len(sentence)
    assert 0.3 < fraction < 0.5


def test_post_clean_admission_excludes_only_configured_risks() -> None:
    admission = load_post_clean_document_admission(_policy())
    repeated = "This duplicated textbook sentence is intentionally very long " * 3
    repeated_text = f"{repeated}. {repeated}. A short unique conclusion."

    assert (
        admission.exclusion_reason(
            {"source": "other", "legacy_exact_overlap": True, "text": "safe"}
        )
        == "legacy_exact_overlap"
    )
    assert (
        admission.exclusion_reason(
            {"source": "books", "legacy_exact_overlap": False, "text": repeated_text}
        )
        == "severe_internal_sentence_repetition"
    )
    assert (
        admission.exclusion_reason(
            {
                "source": "code",
                "legacy_exact_overlap": False,
                "text": 'password = "oracle12345"; key = "AKIAEXAMPLEKEY123456"',
            }
        )
        == ""
    )


def test_post_clean_admission_rejects_unknown_configuration() -> None:
    with pytest.raises(ValueError, match="unknown post_clean_document_admission"):
        load_post_clean_document_admission(
            {"post_clean_document_admission": {"typo": True}}
        )


def test_post_clean_admission_loads_hash_pinned_exclusion_manifest(
    tmp_path: Path,
) -> None:
    excluded_hash = "12" * 32
    manifest = tmp_path / "artifacts" / "exclusions.tsv"
    manifest.parent.mkdir()
    manifest.write_bytes(
        f"{excluded_hash}\tfinemath_subscription_preview\n".encode("ascii")
    )
    policy_path = tmp_path / "policy.json"
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    policy = {
        "post_clean_document_admission": {
            "document_exclusion_manifest": {
                "path": "artifacts/exclusions.tsv",
                "sha256": manifest_sha256,
                "entries": 1,
                "sources": ["finemath"],
            }
        }
    }

    admission = load_post_clean_document_admission(
        policy,
        policy_path=policy_path,
    )

    assert admission.required_columns == {"document_sha256"}
    assert (
        admission.exclusion_reason(
            {"source": "finemath", "document_sha256": excluded_hash}
        )
        == "finemath_subscription_preview"
    )
    assert (
        admission.exclusion_reason(
            {"source": "finemath", "document_sha256": "34" * 32}
        )
        == ""
    )
    assert (
        admission.exclusion_reason(
            {"source": "other", "document_sha256": excluded_hash}
        )
        == ""
    )
    assert admission.to_report()["document_exclusion_manifest"] == {
        "path": str(manifest.resolve()),
        "sha256": manifest_sha256,
        "entries": 1,
        "sources": ["finemath"],
        "reason_counts": {"finemath_subscription_preview": 1},
    }


def test_post_clean_admission_rejects_manifest_drift(tmp_path: Path) -> None:
    manifest = tmp_path / "exclusions.tsv"
    manifest.write_text(f"{'ff' * 32}\treason\n", encoding="ascii", newline="\n")
    policy = {
        "post_clean_document_admission": {
            "document_exclusion_manifest": {
                "path": str(manifest),
                "sha256": "00" * 32,
                "entries": 1,
                "sources": ["finemath"],
            }
        }
    }

    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_post_clean_document_admission(policy)
