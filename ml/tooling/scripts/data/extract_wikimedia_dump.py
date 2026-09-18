from __future__ import annotations

import argparse
import bz2
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any
from urllib.parse import quote
import uuid

from lxml import etree
import mwparserfromhell
from opencc import OpenCC
import pyarrow as pa
import pyarrow.parquet as pq

from ml.core.common.io import write_json_atomic
from ml.tooling.scripts.data.snapshot_wikimedia_dump_inventory import INVENTORY_SCHEMA


REPORT_SCHEMA = "sophia_wikimedia_extraction_v1"
SHARD_REPORT_SCHEMA = "sophia_wikimedia_extracted_shard_v1"
_BATCH_ROWS = 2_048
_OUTPUT_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("wiki", pa.string()),
        ("language", pa.string()),
        ("output_language", pa.string()),
        ("snapshot_date", pa.string()),
        ("page_id", pa.int64()),
        ("revision_id", pa.int64()),
        ("revision_timestamp", pa.string()),
        ("title", pa.string()),
        ("url", pa.string()),
        ("text", pa.string()),
        ("raw_wikitext_sha256", pa.string()),
        ("extracted_text_sha256", pa.string()),
        ("normalized_text_sha256", pa.string()),
        ("script_normalization", pa.string()),
        ("source_article_filename", pa.string()),
        ("source_article_sha1", pa.string()),
        ("source_index_filename", pa.string()),
        ("source_index_sha1", pa.string()),
        ("character_count", pa.int64()),
        ("utf8_bytes", pa.int64()),
    ]
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_hashes(path: Path) -> tuple[str, str, str]:
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1(usedforsecurity=False)
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha256.update(chunk)
            sha1.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), sha1.hexdigest(), md5.hexdigest()


def _verify_input(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing Wikimedia input: {path}")
    actual_bytes = int(path.stat().st_size)
    if actual_bytes != int(row.get("bytes") or 0):
        raise ValueError(f"Wikimedia input size mismatch: {path}")
    sha256, sha1, md5 = _file_hashes(path)
    if sha1 != str(row.get("sha1") or "") or md5 != str(row.get("md5") or ""):
        raise ValueError(f"Wikimedia input checksum mismatch: {path}")
    return {
        "filename": str(row["filename"]),
        "path": str(path.resolve()),
        "bytes": actual_bytes,
        "sha256": sha256,
        "sha1": sha1,
        "md5": md5,
    }


def _clean_extracted_text(wikitext: str) -> str:
    parsed = mwparserfromhell.parse(wikitext)
    text = parsed.strip_code(normalize=True, collapse=True)
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _child_text(element: etree._Element, name: str) -> str:
    return str(element.findtext(f"./{{*}}{name}") or "")


def _canonical_url(canonical_host: str, title: str) -> str:
    host = str(canonical_host).strip().lower()
    if not re.fullmatch(r"[a-z0-9.-]+\.org", host):
        raise ValueError(f"invalid Wikimedia canonical host: {canonical_host!r}")
    path = quote(title.replace(" ", "_"), safe="()_-")
    return f"https://{host}/wiki/{path}"


def _iter_pages(path: Path):
    with bz2.open(path, "rb") as handle:
        context = etree.iterparse(
            handle,
            events=("end",),
            tag="{*}page",
            recover=False,
            huge_tree=True,
        )
        for _event, page in context:
            yield page
            page.clear()
            parent = page.getparent()
            if parent is not None:
                while page.getprevious() is not None:
                    del parent[0]


def _write_batch(writer: pq.ParquetWriter, rows: list[dict[str, object]]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=_OUTPUT_SCHEMA))
        rows.clear()


def _extract_shard(
    *,
    source: dict[str, Any],
    article: dict[str, Any],
    index: dict[str, Any],
    raw_source_root: Path,
    output_source_root: Path,
    inventory_sha256: str,
) -> dict[str, Any]:
    shard_index = int(article["shard_index"])
    if int(index["shard_index"]) != shard_index:
        raise ValueError(f"Wikimedia input pair mismatch: shard={shard_index}")
    final_dir = output_source_root / f"shard-{shard_index:05d}"
    if final_dir.exists():
        shard_report = _load_json(final_dir / "report.json")
        output_file = final_dir / "data.parquet"
        if (
            str(shard_report.get("schema")) != SHARD_REPORT_SCHEMA
            or str(shard_report.get("inventory_sha256")) != inventory_sha256
            or str(shard_report.get("article_sha1")) != str(article["sha1"])
            or str(shard_report.get("index_sha1")) != str(index["sha1"])
            or not output_file.is_file()
            or _file_hashes(output_file)[0]
            != str(shard_report.get("output_sha256") or "")
        ):
            raise ValueError(f"invalid cached Wikimedia extraction: {final_dir}")
        return shard_report

    article_path = raw_source_root / str(article["filename"])
    index_path = raw_source_root / str(index["filename"])
    article_input = _verify_input(article_path, article)
    index_input = _verify_input(index_path, index)
    stage = output_source_root / f".shard-{shard_index:05d}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    output_file = stage / "data.parquet"
    stats: Counter[str] = Counter()
    batch: list[dict[str, object]] = []
    normalization = str(source.get("script_normalization") or "")
    normalizer = OpenCC("t2s") if normalization.startswith("opencc_t2s") else None
    canonical_host = str(source.get("canonical_host") or "")
    writer = pq.ParquetWriter(output_file, _OUTPUT_SCHEMA, compression="zstd")
    try:
        for page in _iter_pages(article_path):
            stats["pages_in"] += 1
            if stats["pages_in"] % 25_000 == 0:
                print(
                    f"[WIKIMEDIA-EXTRACT-PROGRESS] source={source['name']} "
                    f"shard={shard_index} pages={stats['pages_in']:,} "
                    f"kept={stats['pages_out']:,}",
                    flush=True,
                )
            if _child_text(page, "ns") != "0":
                stats["drop_namespace"] += 1
                continue
            if page.find("./{*}redirect") is not None:
                stats["drop_redirect"] += 1
                continue
            revision = page.find("./{*}revision")
            if revision is None:
                stats["drop_missing_revision"] += 1
                continue
            title = _child_text(page, "title").strip()
            page_id_raw = _child_text(page, "id")
            revision_id_raw = _child_text(revision, "id")
            timestamp = _child_text(revision, "timestamp").strip()
            text_element = revision.find("./{*}text")
            wikitext = "" if text_element is None or text_element.text is None else str(text_element.text)
            try:
                page_id = int(page_id_raw)
                revision_id = int(revision_id_raw)
            except ValueError:
                stats["drop_invalid_provenance"] += 1
                continue
            if not (
                int(article["page_id_start"])
                <= page_id
                <= int(article["page_id_end"])
            ):
                raise ValueError(
                    f"page id outside declared Wikimedia shard range: {page_id}"
                )
            if not title or not timestamp or not wikitext.strip():
                stats["drop_empty_source"] += 1
                continue
            try:
                extracted = _clean_extracted_text(wikitext)
            except Exception:
                stats["drop_wikitext_parse_error"] += 1
                continue
            if not extracted:
                stats["drop_empty_extracted"] += 1
                continue
            normalized = normalizer.convert(extracted) if normalizer is not None else extracted
            if not normalized.strip():
                stats["drop_empty_normalized"] += 1
                continue
            batch.append(
                {
                    "source": str(source["name"]),
                    "wiki": str(source["wiki"]),
                    "language": str(source["language"]),
                    "output_language": str(source["output_language"]),
                    "snapshot_date": str(source["snapshot_date"]),
                    "page_id": page_id,
                    "revision_id": revision_id,
                    "revision_timestamp": timestamp,
                    "title": title,
                    "url": _canonical_url(canonical_host, title),
                    "text": normalized,
                    "raw_wikitext_sha256": _sha256_bytes(wikitext),
                    "extracted_text_sha256": _sha256_bytes(extracted),
                    "normalized_text_sha256": _sha256_bytes(normalized),
                    "script_normalization": str(source["script_normalization"]),
                    "source_article_filename": str(article["filename"]),
                    "source_article_sha1": str(article["sha1"]),
                    "source_index_filename": str(index["filename"]),
                    "source_index_sha1": str(index["sha1"]),
                    "character_count": len(normalized),
                    "utf8_bytes": len(normalized.encode("utf-8")),
                }
            )
            stats["pages_out"] += 1
            stats["characters_out"] += len(normalized)
            stats["utf8_bytes_out"] += len(normalized.encode("utf-8"))
            if len(batch) >= _BATCH_ROWS:
                _write_batch(writer, batch)
        _write_batch(writer, batch)
    except BaseException:
        writer.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise
    writer.close()
    output_sha256 = _file_hashes(output_file)[0]
    shard_report = {
        "schema": SHARD_REPORT_SCHEMA,
        "status": "complete",
        "inventory_sha256": inventory_sha256,
        "source": str(source["name"]),
        "wiki": str(source["wiki"]),
        "snapshot_date": str(source["snapshot_date"]),
        "shard_index": shard_index,
        "page_id_start": int(article["page_id_start"]),
        "page_id_end": int(article["page_id_end"]),
        "article_filename": str(article["filename"]),
        "article_sha1": str(article["sha1"]),
        "article_sha256": str(article_input["sha256"]),
        "index_filename": str(index["filename"]),
        "index_sha1": str(index["sha1"]),
        "index_sha256": str(index_input["sha256"]),
        "output_file": "data.parquet",
        "output_sha256": output_sha256,
        "stats": dict(sorted(stats.items())),
    }
    write_json_atomic(stage / "report.json", shard_report, sort_keys=True)
    output_source_root.mkdir(parents=True, exist_ok=True)
    os.replace(stage, final_dir)
    return shard_report


def extract_inventory(
    *,
    inventory_path: str,
    raw_root: str,
    output_root: str,
    report_path: str,
    selected_sources: list[str] | None = None,
    available_only: bool = False,
) -> dict[str, Any]:
    inventory_file = Path(inventory_path).expanduser().resolve()
    inventory = _load_json(inventory_file)
    if (
        str(inventory.get("schema")) != INVENTORY_SCHEMA
        or str(inventory.get("status")) != "complete"
        or not bool(inventory.get("mutable_latest_urls_forbidden"))
    ):
        raise ValueError("unsupported or incomplete Wikimedia dump inventory")
    inventory_sha256 = hashlib.sha256(inventory_file.read_bytes()).hexdigest()
    requested = {
        str(value).strip() for value in (selected_sources or []) if str(value).strip()
    }
    sources = [row for row in inventory.get("sources", []) if isinstance(row, dict)]
    available = {str(row.get("name") or "") for row in sources}
    if requested - available:
        raise ValueError(f"unknown Wikimedia sources requested: {sorted(requested - available)}")
    raw = Path(raw_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    extracted_sources: list[dict[str, Any]] = []
    for source in sources:
        name = str(source.get("name") or "")
        if requested and name not in requested:
            continue
        by_shard: dict[int, dict[str, dict[str, Any]]] = {}
        for row in source.get("files", []):
            if not isinstance(row, dict):
                raise ValueError("Wikimedia inventory file rows must be objects")
            by_shard.setdefault(int(row["shard_index"]), {})[str(row["role"])] = row
        shard_reports: list[dict[str, Any]] = []
        unavailable_shards: list[int] = []
        for shard_index in sorted(by_shard):
            pair = by_shard[shard_index]
            if set(pair) != {"articles", "index"}:
                raise ValueError(f"unpaired Wikimedia extraction input: {name} shard={shard_index}")
            final_dir = output / name / f"shard-{shard_index:05d}"
            article_path = raw / name / str(pair["articles"]["filename"])
            index_path = raw / name / str(pair["index"]["filename"])
            if (
                bool(available_only)
                and not final_dir.exists()
                and (not article_path.is_file() or not index_path.is_file())
            ):
                unavailable_shards.append(shard_index)
                continue
            print(f"[WIKIMEDIA-EXTRACT] source={name} shard={shard_index}", flush=True)
            shard_reports.append(
                _extract_shard(
                    source=source,
                    article=pair["articles"],
                    index=pair["index"],
                    raw_source_root=raw / name,
                    output_source_root=output / name,
                    inventory_sha256=inventory_sha256,
                )
            )
        if unavailable_shards:
            print(
                f"[WIKIMEDIA-EXTRACT-SKIP] source={name} "
                f"unavailable_shards={len(unavailable_shards)} "
                f"first={unavailable_shards[0]}",
                flush=True,
            )
        extracted_sources.append(
            {
                "name": name,
                "wiki": str(source.get("wiki") or ""),
                "language": str(source.get("language") or ""),
                "output_language": str(source.get("output_language") or ""),
                "domain": str(source.get("domain") or ""),
                "canonical_host": str(source.get("canonical_host") or ""),
                "script_normalization": str(
                    source.get("script_normalization") or ""
                ),
                "license": str(source.get("license") or ""),
                "license_review": str(source.get("license_review") or ""),
                "quality_score": str(source.get("quality_score") or ""),
                "min_chars": int(source.get("min_chars") or 0),
                "deduplication_priority": int(
                    source.get("deduplication_priority") or 0
                ),
                "snapshot_date": str(source.get("snapshot_date") or ""),
                "inventory_shard_count": len(by_shard),
                "shard_count": len(shard_reports),
                "unavailable_shard_count": len(unavailable_shards),
                "complete": len(shard_reports) == len(by_shard),
                "pages_in": sum(int(row["stats"].get("pages_in", 0)) for row in shard_reports),
                "pages_out": sum(int(row["stats"].get("pages_out", 0)) for row in shard_reports),
                "characters_out": sum(
                    int(row["stats"].get("characters_out", 0)) for row in shard_reports
                ),
                "shards": shard_reports,
            }
        )
    if not extracted_sources:
        raise ValueError("Wikimedia extraction selected no sources")
    complete = all(bool(row["complete"]) for row in extracted_sources)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete" if complete else "partial",
        "inventory_path": str(inventory_file),
        "inventory_sha256": inventory_sha256,
        "raw_root": str(raw),
        "output_root": str(output),
        "source_count": len(extracted_sources),
        "inventory_shard_count": sum(
            int(row["inventory_shard_count"]) for row in extracted_sources
        ),
        "shard_count": sum(int(row["shard_count"]) for row in extracted_sources),
        "pages_in": sum(int(row["pages_in"]) for row in extracted_sources),
        "pages_out": sum(int(row["pages_out"]) for row in extracted_sources),
        "characters_out": sum(int(row["characters_out"]) for row in extracted_sources),
        "sources": extracted_sources,
    }
    write_json_atomic(
        Path(report_path).expanduser().resolve(),
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract provenance-rich main-namespace text from pinned Wikimedia dumps."
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Extract only complete local shard pairs and publish a partial report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = extract_inventory(
        inventory_path=str(args.inventory),
        raw_root=str(args.raw_root),
        output_root=str(args.output_root),
        report_path=str(args.report),
        selected_sources=list(args.source),
        available_only=bool(args.available_only),
    )
    print(
        f"[DONE] sources={report['source_count']} shards={report['shard_count']} "
        f"pages={report['pages_out']:,} report={Path(args.report).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
