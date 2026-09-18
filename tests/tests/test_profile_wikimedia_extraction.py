from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ml.tooling.scripts.data import profile_wikimedia_extraction as mod
from ml.tooling.scripts.data.extract_wikimedia_dump import REPORT_SCHEMA


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path, *, future: bool = False) -> Path:
    root = tmp_path / "extracted"
    name = "wikimedia_zhwiki_20260701_v1"
    parquet = root / name / "shard-00001" / "data.parquet"
    parquet.parent.mkdir(parents=True)
    rows = [
        {
            "revision_timestamp": (
                "2026-07-02T00:00:00Z" if future else "2026-06-30T00:00:00Z"
            ),
            "extracted_text_sha256": "a" * 64,
            "normalized_text_sha256": "b" * 64,
            "character_count": 100,
            "utf8_bytes": 250,
        },
        {
            "revision_timestamp": "2024-01-01T00:00:00Z",
            "extracted_text_sha256": "c" * 64,
            "normalized_text_sha256": "c" * 64,
            "character_count": 200,
            "utf8_bytes": 500,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    report = tmp_path / "extraction.json"
    report.write_text(
        json.dumps(
            {
                "schema": REPORT_SCHEMA,
                "status": "complete",
                "inventory_sha256": "d" * 64,
                "output_root": str(root),
                "sources": [
                    {
                        "name": name,
                        "wiki": "zhwiki",
                        "snapshot_date": "20260701",
                        "shards": [
                            {
                                "status": "complete",
                                "shard_index": 1,
                                "output_sha256": _sha256(parquet),
                                "stats": {"pages_out": 2},
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return report


def test_profiles_revision_time_and_normalization(tmp_path: Path) -> None:
    report = mod.profile_extraction(
        extraction_report=str(_write_fixture(tmp_path)),
        output_path=str(tmp_path / "profile.json"),
    )

    assert report["status"] == "complete"
    assert report["documents"] == 2
    source = report["sources"][0]
    assert source["revision_year_documents"] == {"2024": 1, "2026": 1}
    assert source["revision_within_365_days_before_snapshot_label_documents"] == 1
    assert source["revision_after_snapshot_label_documents"] == 0
    assert source["normalization_changed_documents"] == 1
    assert source["revision_timestamp_max"] == "2026-06-30T00:00:00Z"


def test_reports_revision_newer_than_snapshot_label(tmp_path: Path) -> None:
    report = mod.profile_extraction(
        extraction_report=str(_write_fixture(tmp_path, future=True)),
        output_path=str(tmp_path / "profile.json"),
    )

    source = report["sources"][0]
    assert source["revision_after_snapshot_label_documents"] == 1
    assert source["revision_timestamp_max"] == "2026-07-02T00:00:00Z"
