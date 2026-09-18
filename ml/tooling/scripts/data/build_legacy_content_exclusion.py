from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import time
from collections.abc import Iterator
from typing import Any

import pyarrow.parquet as pq

from ml.core.common.io import write_json_atomic
from ml.tooling.scripts.data.corpus_quality import (
    dedup_normalize,
    near_duplicate_signature,
)
from ml.tooling.scripts.data.download_hf_source_inventory import sha256_file


REPORT_SCHEMA = "sophia_legacy_content_exclusion_v1"
INDEX_SCHEMA = "sha256_nfkc_casefold_remove_whitespace_plus_cdc_minhash8_v3"
NEAR_SIGNATURE_SIZE = 8
NEAR_BANDS = 4
NEAR_BAND_SIZE = 2
NEAR_SIMILARITY_THRESHOLD = 0.75


def _signature_blob(signature: tuple[int, ...]) -> bytes:
    return struct.pack(f">{len(signature)}Q", *signature) if signature else b""


def _band_keys(signature: tuple[int, ...]) -> tuple[bytes, ...]:
    if len(signature) < NEAR_SIGNATURE_SIZE:
        return ()
    return tuple(
        bytes([band])
        + struct.pack(
            ">2Q",
            *signature[
                band * NEAR_BAND_SIZE : band * NEAR_BAND_SIZE + NEAR_BAND_SIZE
            ],
        )
        for band in range(NEAR_BANDS)
    )


def _canonical_roots(values: list[str]) -> list[Path]:
    roots = sorted({Path(value).expanduser().resolve() for value in values})
    if not roots:
        raise ValueError("at least one legacy corpus root is required")
    missing = [str(path) for path in roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"legacy corpus roots do not exist: {missing}")
    return roots


def _candidate_files(roots: list[Path]) -> list[tuple[Path, Path]]:
    rows = sorted(
        {
            (root, path.resolve())
            for root in roots
            for path in root.rglob("*.parquet")
            if path.is_file()
        },
        key=lambda row: str(row[1]),
    )
    if not rows:
        raise ValueError("legacy corpus roots contain no parquet files")
    return rows


def _manifest_sha256(files: list[tuple[Path, Path]]) -> str:
    digest = hashlib.sha256()
    for root, path in files:
        relative = str(path.relative_to(root)).replace("\\", "/")
        digest.update(str(root).encode("utf-8"))
        digest.update(b"\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


class LegacyHashIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(str(path), timeout=120.0)
        has_metadata = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'metadata'"
        ).fetchone()
        if has_metadata is not None:
            existing = self.connection.execute(
                "SELECT value FROM metadata WHERE key = 'index_schema'"
            ).fetchone()
            if existing is not None and str(existing[0]) != INDEX_SCHEMA:
                self.connection.close()
                raise ValueError(
                    "legacy exclusion database uses an incompatible index schema: "
                    f"stored={existing[0]} current={INDEX_SCHEMA}"
                )
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA cache_size=-524288")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("PRAGMA mmap_size=1073741824")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hashes (
                sha256 BLOB PRIMARY KEY
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS signatures (
                sha256 BLOB PRIMARY KEY,
                signature BLOB NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS bands (
                band_key BLOB NOT NULL,
                sha256 BLOB NOT NULL,
                PRIMARY KEY (band_key, sha256)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS processed_files (
                path TEXT PRIMARY KEY,
                corpus_root TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                documents INTEGER NOT NULL,
                indexed_variants INTEGER NOT NULL,
                unique_hashes_added INTEGER NOT NULL,
                indexed_signatures INTEGER NOT NULL,
                unique_signatures_added INTEGER NOT NULL
            );
            """
        )

    def set_lineage(self, values: dict[str, str]) -> None:
        try:
            for key, value in values.items():
                existing = self.connection.execute(
                    "SELECT value FROM metadata WHERE key = ?", (str(key),)
                ).fetchone()
                if existing is not None and str(existing[0]) != str(value):
                    raise ValueError(
                        f"legacy exclusion resume lineage mismatch for {key}: "
                        f"stored={existing[0]} current={value}"
                    )
                self.connection.execute(
                    "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
                    (str(key), str(value)),
                )
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def processed(self, path: Path) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM processed_files WHERE path = ?", (str(path),)
            ).fetchone()
            is not None
        )

    def add_file(
        self,
        *,
        root: Path,
        path: Path,
        file_sha256: str,
        batches: Iterator[
            tuple[list[tuple[bytes, bytes, tuple[bytes, ...]]], int, int]
        ],
    ) -> tuple[int, int, int, int, int]:
        before = int(self.connection.execute("SELECT count(*) FROM hashes").fetchone()[0])
        signatures_before = int(
            self.connection.execute("SELECT count(*) FROM signatures").fetchone()[0]
        )
        documents = 0
        indexed_variants = 0
        indexed_signatures = 0
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            for batch, batch_documents, batch_variants in batches:
                self.connection.executemany(
                    "INSERT OR IGNORE INTO hashes(sha256) VALUES (?)",
                    ((sqlite3.Binary(value[0]),) for value in batch),
                )
                signature_rows = [value for value in batch if value[1]]
                self.connection.executemany(
                    "INSERT OR IGNORE INTO signatures(sha256, signature) VALUES (?, ?)",
                    (
                        (sqlite3.Binary(sha256), sqlite3.Binary(signature))
                        for sha256, signature, _bands in signature_rows
                    ),
                )
                self.connection.executemany(
                    "INSERT OR IGNORE INTO bands(band_key, sha256) VALUES (?, ?)",
                    (
                        (sqlite3.Binary(band), sqlite3.Binary(sha256))
                        for sha256, _signature, bands in signature_rows
                        for band in bands
                    ),
                )
                documents += int(batch_documents)
                indexed_variants += int(batch_variants)
                indexed_signatures += len(signature_rows)
            after = int(
                self.connection.execute("SELECT count(*) FROM hashes").fetchone()[0]
            )
            added = after - before
            signatures_after = int(
                self.connection.execute("SELECT count(*) FROM signatures").fetchone()[0]
            )
            signatures_added = signatures_after - signatures_before
            self.connection.execute(
                "INSERT INTO processed_files("
                "path, corpus_root, relative_path, bytes, sha256, documents, "
                "indexed_variants, unique_hashes_added, indexed_signatures, "
                "unique_signatures_added) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(path),
                    str(root),
                    str(path.relative_to(root)).replace("\\", "/"),
                    int(path.stat().st_size),
                    str(file_sha256),
                    int(documents),
                    int(indexed_variants),
                    int(added),
                    int(indexed_signatures),
                    int(signatures_added),
                ),
            )
            self.connection.commit()
            return (
                added,
                documents,
                indexed_variants,
                indexed_signatures,
                signatures_added,
            )
        except BaseException:
            self.connection.rollback()
            raise

    def summary(self) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT count(*), coalesce(sum(bytes), 0), "
            "coalesce(sum(documents), 0), coalesce(sum(indexed_variants), 0), "
            "coalesce(sum(unique_hashes_added), 0), "
            "coalesce(sum(indexed_signatures), 0), "
            "coalesce(sum(unique_signatures_added), 0) FROM processed_files"
        ).fetchone()
        roots = [
            {
                "corpus_root": str(root),
                "processed_files": int(files),
                "bytes": int(byte_count),
                "documents": int(documents),
                "indexed_variants": int(variants),
                "unique_hashes_added": int(hashes),
                "indexed_signatures": int(signatures),
                "unique_signatures_added": int(unique_signatures),
            }
            for (
                root,
                files,
                byte_count,
                documents,
                variants,
                hashes,
                signatures,
                unique_signatures,
            ) in self.connection.execute(
                "SELECT corpus_root, count(*), sum(bytes), sum(documents), "
                "sum(indexed_variants), sum(unique_hashes_added), "
                "sum(indexed_signatures), sum(unique_signatures_added) "
                "FROM processed_files GROUP BY corpus_root ORDER BY corpus_root"
            )
        ]
        return {
            "processed_files": int(row[0]),
            "processed_bytes": int(row[1]),
            "documents": int(row[2]),
            "indexed_variants": int(row[3]),
            "unique_hashes": int(
                self.connection.execute("SELECT count(*) FROM hashes").fetchone()[0]
            ),
            "unique_hashes_added": int(row[4]),
            "indexed_signatures": int(row[5]),
            "unique_signatures": int(
                self.connection.execute("SELECT count(*) FROM signatures").fetchone()[0]
            ),
            "unique_signatures_added": int(row[6]),
            "roots": roots,
        }

    def close(self) -> None:
        if self.connection.in_transaction:
            self.connection.rollback()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()


def _working_database_path(
    *,
    database: Path,
    working_database_path: str,
) -> Path:
    value = str(working_database_path or "").strip()
    working = Path(value).expanduser().resolve() if value else database
    working.parent.mkdir(parents=True, exist_ok=True)
    if working != database and not working.exists() and database.exists():
        shutil.copy2(database, working)
    return working


def _publish_database(*, working: Path, database: Path) -> None:
    if working == database:
        return
    temporary = database.with_name(f".{database.name}.publish-{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale database publication file exists: {temporary}")
    try:
        shutil.copy2(working, temporary)
        os.replace(temporary, database)
    finally:
        if temporary.exists():
            temporary.unlink()


def _text_variants(
    batch: Any, *, text_column: str, title_column: str
) -> tuple[list[tuple[bytes, bytes, tuple[bytes, ...]]], int, int]:
    indexed: list[tuple[bytes, bytes, tuple[bytes, ...]]] = []
    documents = 0
    variants = 0
    rows = batch.to_pylist()
    for row in rows:
        text = str(row.get(text_column) or "").strip()
        if not text:
            continue
        documents += 1
        candidates = [text]
        title = str(row.get(title_column) or "").strip() if title_column else ""
        if title and not text.startswith(title):
            candidates.append(f"{title}\n\n{text}")
        seen: set[bytes] = set()
        for candidate in candidates:
            normalized = dedup_normalize(candidate)
            if not normalized:
                continue
            digest = hashlib.sha256(normalized.encode("utf-8")).digest()
            if digest in seen:
                continue
            seen.add(digest)
            signature_values = near_duplicate_signature(
                candidate, size=NEAR_SIGNATURE_SIZE
            )
            indexed.append(
                (
                    digest,
                    _signature_blob(signature_values),
                    _band_keys(signature_values),
                )
            )
            variants += 1
    return indexed, documents, variants


def _index_parquet(
    path: Path,
    *,
    batch_size: int,
) -> Iterator[tuple[list[tuple[bytes, bytes, tuple[bytes, ...]]], int, int]]:
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    text_column = "text" if "text" in names else "content" if "content" in names else ""
    if not text_column:
        raise ValueError(f"legacy parquet has no text/content column: {path}")
    title_column = "title" if "title" in names else ""
    columns = [text_column, *([title_column] if title_column else [])]
    for batch in parquet.iter_batches(batch_size=int(batch_size), columns=columns):
        yield _text_variants(
            batch,
            text_column=text_column,
            title_column=title_column,
        )


def build_legacy_content_exclusion(
    *,
    corpus_roots: list[str],
    database_path: str,
    report_path: str,
    working_database_path: str = "",
    batch_size: int = 4_096,
    max_files: int = 0,
) -> dict[str, Any]:
    roots = _canonical_roots(corpus_roots)
    files = _candidate_files(roots)
    manifest_sha256 = _manifest_sha256(files)
    database = Path(database_path).expanduser().resolve()
    working_database = _working_database_path(
        database=database,
        working_database_path=working_database_path,
    )
    report_file = Path(report_path).expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    index = LegacyHashIndex(working_database)
    processed_now = 0
    try:
        index.set_lineage(
            {
                "schema": REPORT_SCHEMA,
                "index_schema": INDEX_SCHEMA,
                "candidate_manifest_sha256": manifest_sha256,
                "corpus_roots": json.dumps(
                    [str(path) for path in roots], sort_keys=True
                ),
            }
        )
        for file_index, (root, path) in enumerate(files, start=1):
            if index.processed(path):
                continue
            if int(max_files) > 0 and processed_now >= int(max_files):
                break
            file_sha256 = sha256_file(path)
            added, documents, variants, signatures, signatures_added = index.add_file(
                root=root,
                path=path,
                file_sha256=file_sha256,
                batches=_index_parquet(path, batch_size=int(batch_size)),
            )
            processed_now += 1
            if processed_now % 10 == 0 or file_index == len(files):
                print(
                    f"[LEGACY-INDEX] files={index.summary()['processed_files']}/"
                    f"{len(files)} documents={index.summary()['documents']:,} "
                    f"unique_hashes={index.summary()['unique_hashes']:,} "
                    f"unique_signatures={index.summary()['unique_signatures']:,} "
                    f"last_added={added:,}/{signatures_added:,} "
                    f"last_signatures={signatures:,}",
                    flush=True,
                )
        summary = index.summary()
    finally:
        index.close()
    complete = int(summary["processed_files"]) == len(files)
    if complete:
        _publish_database(working=working_database, database=database)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete" if complete else "in_progress",
        "index_schema": INDEX_SCHEMA,
        "corpus_roots": [str(path) for path in roots],
        "candidate_files": len(files),
        "candidate_manifest_sha256": manifest_sha256,
        "database_path": str(database),
        "database_sha256": sha256_file(database) if complete else "",
        "working_database_path": (
            str(working_database) if working_database != database else ""
        ),
        "duration_seconds_this_invocation": float(time.time() - started),
        "near_duplicate_index": {
            "signature": (
                "coordinate MinHash over content-defined normalized text chunks"
            ),
            "signature_size": NEAR_SIGNATURE_SIZE,
            "chunk_min_characters": 32,
            "chunk_target_characters": 64,
            "chunk_max_characters": 128,
            "bands": NEAR_BANDS,
            "band_size": NEAR_BAND_SIZE,
            "similarity": "matching MinHash coordinates divided by signature size",
            "minimum_similarity": NEAR_SIMILARITY_THRESHOLD,
        },
        **summary,
    }
    write_json_atomic(
        report_file,
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a resumable exact and near-content exclusion index from legacy corpora."
    )
    parser.add_argument("--corpus-root", action="append", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument(
        "--working-database",
        default="",
        help=(
            "Optional native-filesystem build path. An existing partial --database "
            "is copied here once, then the complete index is atomically published back."
        ),
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--max-files", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_legacy_content_exclusion(
        corpus_roots=[str(value) for value in args.corpus_root],
        database_path=str(args.database),
        report_path=str(args.report),
        working_database_path=str(args.working_database),
        batch_size=int(args.batch_size),
        max_files=int(args.max_files),
    )
    print(
        f"[DONE] status={report['status']} files={report['processed_files']}/"
        f"{report['candidate_files']} unique_hashes={report['unique_hashes']:,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
