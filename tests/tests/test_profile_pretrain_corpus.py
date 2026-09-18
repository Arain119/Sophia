from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ml.tooling.scripts.data import profile_pretrain_corpus as mod


token_length_bucket = mod.token_length_bucket


def test_token_length_buckets_cover_native_context_boundaries() -> None:
    assert token_length_bucket(4_095) == "tokens_lt_4k"
    assert token_length_bucket(4_096) == "tokens_4k_8k"
    assert token_length_bucket(8_192) == "tokens_8k_16k"
    assert token_length_bucket(16_384) == "tokens_ge_16k"


def test_profile_reports_source_taxonomy_token_supply(
    tmp_path: Path, monkeypatch
) -> None:
    clean_root = tmp_path / "clean"
    parquet_path = clean_root / "train" / "curated_zh" / "part.parquet"
    parquet_path.parent.mkdir(parents=True)
    classical = (
        "天地玄黄，宇宙洪荒。" + "吾之所学，乃圣人之道也，学而时习之，不亦说乎。" * 20
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "text": classical,
                    "source": "curated_zh",
                    "source_revision": "a" * 40,
                    "repo_id": "example/curated_zh",
                    "license": "cc-by-sa-4.0",
                    "domain": "chinese_primary_texts",
                    "language": "zh",
                    "length_bucket": "chars_lt_2k",
                    "mix_bucket": "curated_zh|chars_lt_2k",
                    "document_timestamp": "2026-07-01T00:00:00Z",
                }
            ]
        ),
        parquet_path,
    )
    (clean_root / "cleaning_report.json").write_text(
        json.dumps(
            {
                "schema": "sophia_clean_pretrain_corpus_v1",
                "status": "complete",
                "sources": [
                    {
                        "output_files": [str(parquet_path)],
                        "output_file_inventory": [
                            {
                                "path": str(parquet_path.resolve()),
                                "bytes": parquet_path.stat().st_size,
                                "sha256": hashlib.sha256(
                                    parquet_path.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps({"required_tokenizer_bundle_sha1": "selected-tokenizer"}),
        encoding="utf-8",
    )

    class FakeTokenizer:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, texts, **_kwargs):
            self.calls += 1
            return {"length": [7 for _ in texts]}

    tokenizer = FakeTokenizer()
    working_state = tmp_path / "native-state" / "profile.sqlite3"
    monkeypatch.setattr(
        mod, "compute_tokenizer_bundle_sha1", lambda _path: "selected-tokenizer"
    )
    monkeypatch.setattr(
        mod, "load_local_tokenizer", lambda *_args, **_kwargs: tokenizer
    )

    report = mod.profile_pretrain_corpus(
        clean_root=str(clean_root),
        tokenizer_path=str(tmp_path / "tokenizer"),
        admission_policy=str(policy_path),
        output_path=str(tmp_path / "profile.json"),
        working_state_database=str(working_state),
    )

    assert report["tokens"] == 8
    assert report["tokens_by"]["user_taxonomy"] == {"literature_classical": 8}
    assert report["tokens_by"]["source_user_taxonomy"] == {
        "curated_zh|literature_classical": 8
    }
    assert report["tokens_by"]["document_time_year"] == {"2026": 8}
    assert report["tokens_by"]["source_document_time_year"] == {"curated_zh|2026": 8}
    cleaning_path = clean_root / "cleaning_report.json"
    assert (
        report["cleaning_report_sha256"]
        == hashlib.sha256(cleaning_path.read_bytes()).hexdigest()
    )
    assert (
        report["admission_policy_sha256"]
        == hashlib.sha256(policy_path.read_bytes()).hexdigest()
    )
    assert report["resume_state"] == str(working_state.resolve())
    assert working_state.is_file()

    resumed = mod.profile_pretrain_corpus(
        clean_root=str(clean_root),
        tokenizer_path=str(tmp_path / "tokenizer"),
        admission_policy=str(policy_path),
        output_path=str(tmp_path / "profile.json"),
        working_state_database=str(working_state),
    )
    assert tokenizer.calls == 1
    assert resumed == report

    cleaning = json.loads(cleaning_path.read_text())
    cleaning["lineage_drift"] = True
    cleaning_path.write_text(json.dumps(cleaning), encoding="utf-8")
    with pytest.raises(ValueError, match="resume lineage mismatch"):
        mod.profile_pretrain_corpus(
            clean_root=str(clean_root),
            tokenizer_path=str(tmp_path / "tokenizer"),
            admission_policy=str(policy_path),
            output_path=str(tmp_path / "profile.json"),
            working_state_database=str(working_state),
        )


def test_profile_applies_post_clean_document_admission(
    tmp_path: Path, monkeypatch
) -> None:
    clean_root = tmp_path / "clean"
    parquet_path = clean_root / "train" / "books" / "part.parquet"
    parquet_path.parent.mkdir(parents=True)
    repeated = "A duplicated textbook sentence with substantial useful wording " * 5

    def row(text: str, *, legacy: bool = False) -> dict[str, object]:
        return {
            "text": text,
            "source": "books",
            "source_revision": "a" * 40,
            "repo_id": "example/books",
            "license": "cc-by-4.0",
            "domain": "books",
            "language": "en",
            "length_bucket": "chars_lt_2k",
            "mix_bucket": "books|chars_lt_2k",
            "document_timestamp": "2026-07-01T00:00:00Z",
            "legacy_exact_overlap": legacy,
        }

    pq.write_table(
        pa.Table.from_pylist(
            [
                row(
                    " ".join(
                        f"A clean educational explanation number {index} is unique."
                        for index in range(10)
                    )
                ),
                row(f"{repeated}. {repeated}. A unique conclusion."),
                row(
                    "A legacy document that must not enter the new model. " * 8,
                    legacy=True,
                ),
            ]
        ),
        parquet_path,
    )
    (clean_root / "cleaning_report.json").write_text(
        json.dumps(
            {
                "schema": "sophia_clean_pretrain_corpus_v1",
                "status": "complete",
                "sources": [
                    {
                        "output_files": [str(parquet_path)],
                        "output_file_inventory": [
                            {
                                "path": str(parquet_path.resolve()),
                                "bytes": parquet_path.stat().st_size,
                                "sha256": hashlib.sha256(
                                    parquet_path.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "required_tokenizer_bundle_sha1": "selected-tokenizer",
                "post_clean_document_admission": {
                    "exclude_legacy_exact_overlap": True,
                    "repeated_sentence_rules": {
                        "books": {
                            "min_sentence_chars": 20,
                            "repeats": 2,
                            "minimum_repeated_chars": 80,
                            "minimum_repeated_fraction": 0.2,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeTokenizer:
        def __call__(self, texts, **_kwargs):
            return {"length": [7 for _ in texts]}

    monkeypatch.setattr(
        mod, "compute_tokenizer_bundle_sha1", lambda _path: "selected-tokenizer"
    )
    monkeypatch.setattr(
        mod, "load_local_tokenizer", lambda *_args, **_kwargs: FakeTokenizer()
    )

    report = mod.profile_pretrain_corpus(
        clean_root=str(clean_root),
        tokenizer_path=str(tmp_path / "tokenizer"),
        admission_policy=str(policy_path),
        output_path=str(tmp_path / "profile.json"),
    )

    assert report["documents_scanned"] == 3
    assert report["documents"] == 1
    assert report["documents_excluded"] == 2
    assert report["documents_excluded_by"] == {
        "legacy_exact_overlap": 1,
        "severe_internal_sentence_repetition": 1,
    }
    assert report["tokens"] == 8


def test_parallel_profile_matches_serial_profile(tmp_path: Path) -> None:
    clean_root = tmp_path / "clean"
    parquet_paths = [
        clean_root / "curated_zh" / "train" / f"input-{index:05d}" / "part.parquet"
        for index in range(2)
    ]
    inventory = []
    for index, parquet_path in enumerate(parquet_paths):
        parquet_path.parent.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "text": "这是可复现的中文教育文本。" * (20 + index),
                        "source": "curated_zh",
                        "source_revision": "b" * 40,
                        "repo_id": "example/curated_zh",
                        "license": "cc-by-sa-4.0",
                        "domain": "modern_chinese_education",
                        "language": "zh",
                        "length_bucket": "chars_lt_2k",
                        "mix_bucket": "curated_zh|chars_lt_2k",
                        "document_timestamp": "2026-07-01T00:00:00Z",
                    },
                    {
                        "text": "A deterministic English STEM document. "
                        * (30 + index),
                        "source": "curated_zh",
                        "source_revision": "b" * 40,
                        "repo_id": "example/curated_zh",
                        "license": "cc-by-sa-4.0",
                        "domain": "modern_chinese_education",
                        "language": "en",
                        "length_bucket": "chars_lt_2k",
                        "mix_bucket": "curated_zh|chars_lt_2k",
                        "document_timestamp": "2025-06-01T00:00:00Z",
                    },
                ]
            ),
            parquet_path,
        )
        inventory.append(
            {
                "path": str(parquet_path.resolve()),
                "bytes": parquet_path.stat().st_size,
                "sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
            }
        )
    (clean_root / "cleaning_report.json").write_text(
        json.dumps(
            {
                "schema": "sophia_clean_pretrain_corpus_v1",
                "status": "complete",
                "sources": [
                    {
                        "output_files": [str(path) for path in parquet_paths],
                        "output_file_inventory": inventory,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    tokenizer_path = Path("ml/modeling/text").resolve()
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "required_tokenizer_bundle_sha1": (
                    mod.compute_tokenizer_bundle_sha1(tokenizer_path)
                )
            }
        ),
        encoding="utf-8",
    )

    serial = mod.profile_pretrain_corpus(
        clean_root=str(clean_root),
        tokenizer_path=str(tokenizer_path),
        admission_policy=str(policy_path),
        output_path=str(tmp_path / "serial.json"),
        working_state_database=str(tmp_path / "serial.sqlite3"),
        batch_size=1,
        workers=1,
    )
    parallel = mod.profile_pretrain_corpus(
        clean_root=str(clean_root),
        tokenizer_path=str(tokenizer_path),
        admission_policy=str(policy_path),
        output_path=str(tmp_path / "parallel.json"),
        working_state_database=str(tmp_path / "parallel.sqlite3"),
        batch_size=1,
        workers=2,
    )

    for key in (
        "profiled_files",
        "documents",
        "tokens",
        "utf8_bytes",
        "documents_by",
        "tokens_by",
        "long_context_supply",
    ):
        assert parallel[key] == serial[key]
