from __future__ import annotations

import hashlib
import json
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ml.tooling.scripts.data import build_legacy_content_exclusion as mod
from ml.tooling.scripts.data.corpus_quality import dedup_normalize


def _digest(value: str) -> bytes:
    return hashlib.sha256(dedup_normalize(value).encode("utf-8")).digest()


def test_builds_resumable_exact_and_near_index_with_title_variant(tmp_path) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"title": "标题", "content": "第一篇旧训练文档。"},
                {"title": "", "content": "第二篇旧训练文档。"},
            ]
        ),
        root / "a.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"text": "第三篇旧训练文档。"}]),
        root / "b.parquet",
    )
    database = tmp_path / "index.sqlite3"
    report = tmp_path / "report.json"

    partial = mod.build_legacy_content_exclusion(
        corpus_roots=[str(root)],
        database_path=str(database),
        report_path=str(report),
        max_files=1,
    )
    assert partial["status"] == "in_progress"
    assert partial["processed_files"] == 1

    complete = mod.build_legacy_content_exclusion(
        corpus_roots=[str(root)],
        database_path=str(database),
        report_path=str(report),
    )
    assert complete["status"] == "complete"
    assert complete["processed_files"] == 2
    assert complete["documents"] == 3
    assert complete["indexed_variants"] == 4
    assert complete["unique_hashes"] == 4
    assert complete["indexed_signatures"] == 4
    assert complete["unique_signatures"] == 4
    assert complete["near_duplicate_index"] == {
        "signature": "coordinate MinHash over content-defined normalized text chunks",
        "signature_size": 8,
        "chunk_min_characters": 32,
        "chunk_target_characters": 64,
        "chunk_max_characters": 128,
        "bands": 4,
        "band_size": 2,
        "similarity": "matching MinHash coordinates divided by signature size",
        "minimum_similarity": 0.75,
    }
    assert complete["database_sha256"]
    assert json.loads(report.read_text())["status"] == "complete"

    connection = sqlite3.connect(str(database))
    try:
        hashes = {bytes(row[0]) for row in connection.execute("SELECT sha256 FROM hashes")}
        signatures = {
            bytes(row[0]): bytes(row[1])
            for row in connection.execute("SELECT sha256, signature FROM signatures")
        }
    finally:
        connection.close()
    assert _digest("第一篇旧训练文档。") in hashes
    assert _digest("标题\n\n第一篇旧训练文档。") in hashes
    assert _digest("第三篇旧训练文档。") in hashes
    assert set(signatures) == hashes
    assert all(signatures.values())


def test_resume_rejects_changed_root_manifest(tmp_path) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    pq.write_table(pa.Table.from_pylist([{"text": "旧文档"}]), root / "a.parquet")
    database = tmp_path / "index.sqlite3"
    report = tmp_path / "report.json"
    mod.build_legacy_content_exclusion(
        corpus_roots=[str(root)],
        database_path=str(database),
        report_path=str(report),
    )
    pq.write_table(pa.Table.from_pylist([{"text": "另一篇"}]), root / "b.parquet")

    with pytest.raises(ValueError, match="candidate_manifest_sha256"):
        mod.build_legacy_content_exclusion(
            corpus_roots=[str(root)],
            database_path=str(database),
            report_path=str(report),
        )


def test_rejects_v1_database_before_mutating_it(tmp_path) -> None:
    database = tmp_path / "legacy-v1.sqlite3"
    connection = sqlite3.connect(str(database))
    try:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('index_schema', ?)",
            ("sha256_nfkc_casefold_remove_whitespace_v1",),
        )
        connection.commit()
    finally:
        connection.close()
    before = hashlib.sha256(database.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="incompatible index schema"):
        mod.LegacyHashIndex(database)

    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    connection = sqlite3.connect(str(database))
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()
    assert tables == {"metadata"}


def test_native_working_database_resumes_and_publishes_atomically(tmp_path) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    pq.write_table(
        pa.Table.from_pylist([{"text": "第一篇旧训练文档。"}]),
        root / "a.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"text": "第二篇旧训练文档。"}]),
        root / "b.parquet",
    )
    published = tmp_path / "published" / "index.sqlite3"
    working = tmp_path / "native" / "index.sqlite3"
    report = tmp_path / "report.json"

    partial = mod.build_legacy_content_exclusion(
        corpus_roots=[str(root)],
        database_path=str(published),
        working_database_path=str(working),
        report_path=str(report),
        max_files=1,
    )
    assert partial["status"] == "in_progress"
    assert working.is_file()
    assert not published.exists()

    complete = mod.build_legacy_content_exclusion(
        corpus_roots=[str(root)],
        database_path=str(published),
        working_database_path=str(working),
        report_path=str(report),
    )
    assert complete["status"] == "complete"
    assert complete["database_path"] == str(published.resolve())
    assert complete["working_database_path"] == str(working.resolve())
    assert complete["database_sha256"] == hashlib.sha256(
        published.read_bytes()
    ).hexdigest()
    assert not list(published.parent.glob(".index.sqlite3.publish-*.tmp"))
