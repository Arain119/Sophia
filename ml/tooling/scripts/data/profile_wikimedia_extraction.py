from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ml.core.common.io import write_json_atomic
from ml.tooling.scripts.data.download_hf_source_inventory import sha256_file
from ml.tooling.scripts.data.extract_wikimedia_dump import REPORT_SCHEMA
from ml.tooling.scripts.data.snapshot_wikimedia_dump_inventory import SUPPORTED_WIKIS


PROFILE_SCHEMA = "sophia_wikimedia_extraction_profile_v1"
_PROFILE_COLUMNS = (
    "revision_timestamp",
    "extracted_text_sha256",
    "normalized_text_sha256",
    "character_count",
    "utf8_bytes",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _parse_snapshot_date(value: object) -> date:
    raw = str(value or "")
    try:
        parsed = datetime.strptime(raw, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"invalid Wikimedia snapshot date: {raw!r}") from exc
    return parsed


def _parse_revision_timestamp(value: object) -> tuple[str, date]:
    raw = str(value or "")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid Wikimedia revision timestamp: {raw!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Wikimedia revision timestamp lacks timezone: {raw!r}")
    return raw, parsed.date()


def profile_extraction(*, extraction_report: str, output_path: str) -> dict[str, Any]:
    report_path = Path(extraction_report).expanduser().resolve()
    extraction = _load_json(report_path)
    if (
        str(extraction.get("schema")) != REPORT_SCHEMA
        or str(extraction.get("status")) != "complete"
    ):
        raise ValueError("Wikimedia profiling requires a complete extraction report")
    extraction_root = Path(str(extraction.get("output_root") or "")).resolve()
    source_profiles: list[dict[str, Any]] = []
    overall = Counter()
    for source in extraction.get("sources", []):
        if not isinstance(source, dict):
            raise ValueError("Wikimedia extraction sources must be objects")
        name = str(source.get("name") or "")
        wiki = str(source.get("wiki") or "")
        snapshot_raw = str(source.get("snapshot_date") or "")
        if not name or wiki not in SUPPORTED_WIKIS:
            raise ValueError(f"invalid Wikimedia extraction source: {name!r}")
        snapshot = _parse_snapshot_date(snapshot_raw)
        stats = Counter()
        years_documents = Counter()
        years_characters = Counter()
        years_utf8_bytes = Counter()
        minimum_timestamp = ""
        maximum_timestamp = ""
        shard_evidence: list[dict[str, Any]] = []
        for shard in source.get("shards", []):
            if not isinstance(shard, dict) or str(shard.get("status")) != "complete":
                raise ValueError(f"incomplete Wikimedia extracted shard: {name}")
            shard_index = int(shard.get("shard_index") or 0)
            path = extraction_root / name / f"shard-{shard_index:05d}" / "data.parquet"
            expected_sha256 = str(shard.get("output_sha256") or "")
            if (
                shard_index <= 0
                or not path.is_file()
                or sha256_file(path) != expected_sha256
            ):
                raise ValueError(f"Wikimedia extracted shard checksum mismatch: {path}")
            parquet = pq.ParquetFile(path)
            missing = sorted(set(_PROFILE_COLUMNS) - set(parquet.schema_arrow.names))
            if missing:
                raise ValueError(f"Wikimedia profile input lacks {missing}: {path}")
            shard_documents = 0
            for batch in parquet.iter_batches(columns=list(_PROFILE_COLUMNS), batch_size=8192):
                columns = batch.to_pydict()
                for timestamp, extracted_hash, normalized_hash, characters, utf8_bytes in zip(
                    columns["revision_timestamp"],
                    columns["extracted_text_sha256"],
                    columns["normalized_text_sha256"],
                    columns["character_count"],
                    columns["utf8_bytes"],
                    strict=True,
                ):
                    timestamp_raw, revision_date = _parse_revision_timestamp(timestamp)
                    year = str(revision_date.year)
                    character_count = int(characters or 0)
                    byte_count = int(utf8_bytes or 0)
                    if character_count <= 0 or byte_count <= 0:
                        raise ValueError(f"invalid Wikimedia extracted lengths: {path}")
                    age_days = (snapshot - revision_date).days
                    stats["documents"] += 1
                    stats["characters"] += character_count
                    stats["utf8_bytes"] += byte_count
                    stats["normalization_changed_documents"] += int(
                        str(extracted_hash or "") != str(normalized_hash or "")
                    )
                    stats["revision_after_snapshot_label_documents"] += int(age_days < 0)
                    stats["revision_within_365_days_before_snapshot_label_documents"] += int(
                        0 <= age_days <= 365
                    )
                    stats["revision_within_730_days_before_snapshot_label_documents"] += int(
                        0 <= age_days <= 730
                    )
                    years_documents[year] += 1
                    years_characters[year] += character_count
                    years_utf8_bytes[year] += byte_count
                    minimum_timestamp = (
                        timestamp_raw
                        if not minimum_timestamp or timestamp_raw < minimum_timestamp
                        else minimum_timestamp
                    )
                    maximum_timestamp = max(maximum_timestamp, timestamp_raw)
                    shard_documents += 1
            if shard_documents != int(shard.get("stats", {}).get("pages_out", 0)):
                raise ValueError(
                    f"Wikimedia extracted row count mismatch: {path} "
                    f"profiled={shard_documents} reported={shard.get('stats', {}).get('pages_out')}"
                )
            shard_evidence.append(
                {
                    "shard_index": shard_index,
                    "path": str(path),
                    "sha256": expected_sha256,
                    "documents": shard_documents,
                }
            )
        documents = int(stats["documents"])
        if documents <= 0:
            raise ValueError(f"Wikimedia extraction source has no documents: {name}")
        profile = {
            "name": name,
            "wiki": wiki,
            "snapshot_date": snapshot_raw,
            "documents": documents,
            "characters": int(stats["characters"]),
            "utf8_bytes": int(stats["utf8_bytes"]),
            "revision_timestamp_min": minimum_timestamp,
            "revision_timestamp_max": maximum_timestamp,
            "revision_year_documents": dict(sorted(years_documents.items())),
            "revision_year_characters": dict(sorted(years_characters.items())),
            "revision_year_utf8_bytes": dict(sorted(years_utf8_bytes.items())),
            "revision_after_snapshot_label_documents": int(
                stats["revision_after_snapshot_label_documents"]
            ),
            "revision_after_snapshot_label_fraction": float(
                stats["revision_after_snapshot_label_documents"]
            )
            / documents,
            "revision_within_365_days_before_snapshot_label_documents": int(
                stats["revision_within_365_days_before_snapshot_label_documents"]
            ),
            "revision_within_365_days_before_snapshot_label_fraction": float(
                stats["revision_within_365_days_before_snapshot_label_documents"]
            )
            / documents,
            "revision_within_730_days_before_snapshot_label_documents": int(
                stats["revision_within_730_days_before_snapshot_label_documents"]
            ),
            "revision_within_730_days_before_snapshot_label_fraction": float(
                stats["revision_within_730_days_before_snapshot_label_documents"]
            )
            / documents,
            "normalization_changed_documents": int(
                stats["normalization_changed_documents"]
            ),
            "normalization_changed_document_fraction": float(
                stats["normalization_changed_documents"]
            )
            / documents,
            "shards": shard_evidence,
        }
        source_profiles.append(profile)
        overall.update(
            {
                "documents": documents,
                "characters": int(stats["characters"]),
                "utf8_bytes": int(stats["utf8_bytes"]),
                "normalization_changed_documents": int(
                    stats["normalization_changed_documents"]
                ),
                "revision_after_snapshot_label_documents": int(
                    stats["revision_after_snapshot_label_documents"]
                ),
                "revision_within_365_days_before_snapshot_label_documents": int(
                    stats["revision_within_365_days_before_snapshot_label_documents"]
                ),
                "revision_within_730_days_before_snapshot_label_documents": int(
                    stats["revision_within_730_days_before_snapshot_label_documents"]
                ),
            }
        )
        print(
            f"[WIKIMEDIA-PROFILE] source={name} documents={documents:,} "
            f"latest_revision={maximum_timestamp}",
            flush=True,
        )
    if not source_profiles:
        raise ValueError("Wikimedia extraction report contains no sources")
    result = {
        "schema": PROFILE_SCHEMA,
        "status": "complete",
        "extraction_report": str(report_path),
        "extraction_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "inventory_sha256": str(extraction.get("inventory_sha256") or ""),
        "source_count": len(source_profiles),
        "documents": int(overall["documents"]),
        "characters": int(overall["characters"]),
        "utf8_bytes": int(overall["utf8_bytes"]),
        "sources": source_profiles,
    }
    write_json_atomic(
        Path(output_path).expanduser().resolve(),
        result,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile time coverage and script normalization in Wikimedia extraction."
    )
    parser.add_argument("--extraction-report", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = profile_extraction(
        extraction_report=str(args.extraction_report),
        output_path=str(args.output),
    )
    print(
        f"[DONE] sources={report['source_count']} documents={report['documents']:,} "
        f"output={Path(args.output).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
