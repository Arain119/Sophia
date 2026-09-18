from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ml.core.common.io import write_json_atomic
from ml.tooling.scripts.data.download_hf_source_inventory import sha256_file
from ml.tooling.scripts.data.extract_wikimedia_dump import REPORT_SCHEMA as WIKIMEDIA_SCHEMA
from ml.tooling.scripts.data.snapshot_wikimedia_dump_inventory import SUPPORTED_WIKIS


INVENTORY_SCHEMA = "sophia_hf_source_inventory_v2"
ACQUISITION_SCHEMA = "sophia_hf_acquisition_report_v1"
DERIVATIVE_SCHEMA = "sophia_chinese_humanities_derivative_v1"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _json_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _verified_hf_sources(
    report_path: Path, *, bundle_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    acquisition = _load_json(report_path)
    if str(acquisition.get("schema")) != ACQUISITION_SCHEMA:
        raise ValueError(f"unsupported HF acquisition report: {report_path}")
    inventory_path = Path(str(acquisition.get("inventory_path") or "")).resolve()
    inventory = _load_json(inventory_path)
    if str(inventory.get("schema")) != INVENTORY_SCHEMA:
        raise ValueError(f"unsupported HF inventory: {inventory_path}")
    inventory_sha256 = _json_sha256(inventory_path)
    if inventory_sha256 != str(acquisition.get("inventory_sha256") or ""):
        raise ValueError(f"HF acquisition inventory hash mismatch: {report_path}")
    acquired = {
        str(source.get("name") or ""): source
        for source in acquisition.get("sources", [])
        if isinstance(source, dict)
    }
    selected = {
        str(source.get("name") or ""): source
        for source in inventory.get("sources", [])
        if isinstance(source, dict)
    }
    if not acquired or set(acquired) != set(selected):
        raise ValueError(f"HF acquisition source mismatch: {report_path}")
    for name, source in acquired.items():
        expected = {
            str(row.get("path") or ""): row
            for row in selected[name].get("files", [])
            if isinstance(row, dict)
        }
        actual = {
            str(row.get("filename") or ""): row
            for row in source.get("files", [])
            if isinstance(row, dict)
        }
        if not expected or set(actual) != set(expected):
            raise ValueError(f"HF acquisition file mismatch: {name}")
        for filename, row in actual.items():
            path = Path(str(row.get("path") or "")).resolve()
            if not path.is_file() or not _is_relative_to(path, bundle_root):
                raise ValueError(f"HF acquisition file is outside bundle root: {path}")
            expected_row = expected[filename]
            if (
                int(path.stat().st_size) != int(expected_row.get("bytes") or 0)
                or str(row.get("sha256") or "")
                != str(expected_row.get("sha256") or "")
                or sha256_file(path) != str(row.get("sha256") or "")
            ):
                raise ValueError(f"HF acquisition file checksum mismatch: {path}")
    return list(selected.values()), list(acquired.values())


def _wikimedia_sources(
    report_path: Path, *, bundle_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report = _load_json(report_path)
    if (
        str(report.get("schema")) != WIKIMEDIA_SCHEMA
        or str(report.get("status")) != "complete"
    ):
        raise ValueError(f"unsupported Wikimedia extraction report: {report_path}")
    extraction_root = Path(str(report.get("output_root") or "")).resolve()
    inventory_sources: list[dict[str, Any]] = []
    acquired_sources: list[dict[str, Any]] = []
    for source in report.get("sources", []):
        if not isinstance(source, dict):
            raise ValueError("Wikimedia extraction sources must be objects")
        name = str(source.get("name") or "")
        wiki = str(source.get("wiki") or "")
        snapshot_date = str(source.get("snapshot_date") or "")
        wiki_metadata = SUPPORTED_WIKIS.get(wiki)
        if not name or wiki_metadata is None or len(snapshot_date) != 8:
            raise ValueError(f"invalid Wikimedia extraction source: {name}")
        inventory_files: list[dict[str, Any]] = []
        acquired_files: list[dict[str, Any]] = []
        for shard in source.get("shards", []):
            if not isinstance(shard, dict) or str(shard.get("status")) != "complete":
                raise ValueError(f"incomplete Wikimedia extracted shard: {name}")
            shard_index = int(shard.get("shard_index") or 0)
            relative = f"shard-{shard_index:05d}/data.parquet"
            path = (extraction_root / name / relative).resolve()
            expected_sha256 = str(shard.get("output_sha256") or "")
            if (
                shard_index <= 0
                or not path.is_file()
                or not _is_relative_to(path, bundle_root)
                or sha256_file(path) != expected_sha256
            ):
                raise ValueError(f"Wikimedia extracted shard checksum mismatch: {path}")
            logical_path = f"{name}/{relative}"
            inventory_files.append(
                {
                    "path": logical_path,
                    "bytes": int(path.stat().st_size),
                    "sha256": expected_sha256,
                }
            )
            acquired_files.append(
                {
                    "filename": logical_path,
                    "path": str(path),
                    "bytes": int(path.stat().st_size),
                    "sha256": expected_sha256,
                }
            )
        if not inventory_files:
            raise ValueError(f"Wikimedia extraction source has no shards: {name}")
        language = str(source.get("language") or wiki_metadata["language"])
        domain = str(source.get("domain") or wiki_metadata["domain"])
        license_name = str(source.get("license") or wiki_metadata["license"])
        license_review = str(
            source.get("license_review") or wiki_metadata["license_review"]
        )
        quality_score = str(
            source.get("quality_score") or wiki_metadata["quality_score"]
        )
        min_chars = int(source.get("min_chars") or wiki_metadata["min_chars"])
        deduplication_priority = int(
            source.get("deduplication_priority")
            if source.get("deduplication_priority") is not None
            else wiki_metadata["deduplication_priority"]
        )
        if (
            language != wiki_metadata["language"]
            or domain != wiki_metadata["domain"]
            or license_name != wiki_metadata["license"]
            or not license_review
            or quality_score != wiki_metadata["quality_score"]
            or min_chars != int(wiki_metadata["min_chars"])
            or deduplication_priority
            != int(wiki_metadata["deduplication_priority"])
        ):
            raise ValueError(f"Wikimedia extraction metadata mismatch: {name}")
        metadata = {
            "name": name,
            "repo_id": f"wikimedia/{wiki}",
            "revision": snapshot_date,
            "license": license_name,
            "license_review": license_review,
            "language": language,
            "domain": domain,
            "record_format": "parquet",
            "text_column": "text",
            "title_column": "title",
            "id_column": "page_id",
            "url_column": "url",
            "timestamp_column": "revision_timestamp",
            "quality_score": quality_score,
            "deduplication_priority": deduplication_priority,
            "deduplication_priority_reason": (
                "Prefer the canonical page with revision, URL, and attribution lineage "
                "over exact or near-duplicate downstream web copies."
            ),
            "min_chars": min_chars,
            "snapshot_date": snapshot_date,
            "upstream_inventory_sha256": str(report.get("inventory_sha256") or ""),
        }
        inventory_sources.append({**metadata, "files": inventory_files})
        acquired_sources.append({**metadata, "files": acquired_files})
    return inventory_sources, acquired_sources


def _derivative_source(
    report_path: Path, *, bundle_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _load_json(report_path)
    if (
        str(report.get("schema")) != DERIVATIVE_SCHEMA
        or str(report.get("status")) != "complete"
    ):
        raise ValueError(f"unsupported humanities derivative report: {report_path}")
    upstream_acquisition_path = Path(
        str(report.get("upstream_acquisition_report") or "")
    ).resolve()
    upstream_inventory_path = Path(
        str(report.get("upstream_inventory") or "")
    ).resolve()
    if (
        _json_sha256(upstream_acquisition_path)
        != str(report.get("upstream_acquisition_sha256") or "")
        or _json_sha256(upstream_inventory_path)
        != str(report.get("upstream_inventory_sha256") or "")
    ):
        raise ValueError(f"humanities derivative upstream hash mismatch: {report_path}")
    upstream_acquisition = _load_json(upstream_acquisition_path)
    if str(upstream_acquisition.get("schema")) != ACQUISITION_SCHEMA:
        raise ValueError("humanities derivative has unsupported upstream acquisition")
    if Path(str(upstream_acquisition.get("inventory_path") or "")).resolve() != (
        upstream_inventory_path
    ) or str(upstream_acquisition.get("inventory_sha256") or "") != str(
        report.get("upstream_inventory_sha256") or ""
    ):
        raise ValueError("humanities derivative upstream inventory path mismatch")
    upstream_source_name = str(
        (report.get("source") or {}).get("upstream_source_name") or ""
    )
    upstream_sources = [
        row
        for row in upstream_acquisition.get("sources", [])
        if isinstance(row, dict)
        and str(row.get("name") or "") == upstream_source_name
    ]
    if len(upstream_sources) != 1:
        raise ValueError("humanities derivative upstream source mismatch")
    upstream_files = {
        str(row.get("filename") or ""): row
        for row in upstream_sources[0].get("files", [])
        if isinstance(row, dict)
    }
    if int(report.get("input_file_count") or 0) != len(upstream_files):
        raise ValueError("humanities derivative input file count mismatch")
    source = report.get("source")
    if not isinstance(source, dict) or not str(source.get("name") or ""):
        raise ValueError("humanities derivative source metadata is invalid")
    fingerprint = str(report.get("filter_fingerprint_sha256") or "")
    if (
        len(fingerprint) != 64
        or str(source.get("derivative_filter_fingerprint_sha256") or "")
        != fingerprint
    ):
        raise ValueError("humanities derivative filter fingerprint mismatch")
    derivative_root = Path(str(report.get("output_root") or "")).resolve()
    inventory_files: list[dict[str, Any]] = []
    acquired_files: list[dict[str, Any]] = []
    seen_outputs: set[str] = set()
    seen_upstreams: set[str] = set()
    required_columns = {
        "text",
        "upstream_file",
        "upstream_row",
        "upstream_file_sha256",
        "upstream_text_sha256",
        "chinese_quality_proxy_score",
        "humanities_taxonomy",
    }
    for row in report.get("files", []):
        if not isinstance(row, dict) or str(row.get("status")) != "complete":
            raise ValueError("humanities derivative contains an incomplete file")
        filename = str(row.get("filename") or "")
        upstream_filename = str(row.get("upstream_filename") or "")
        path = Path(str(row.get("path") or "")).resolve()
        upstream = upstream_files.get(upstream_filename)
        if (
            not filename
            or filename in seen_outputs
            or upstream_filename in seen_upstreams
            or upstream is None
            or not path.is_file()
            or not _is_relative_to(path, derivative_root)
            or not _is_relative_to(path, bundle_root)
            or int(path.stat().st_size) != int(row.get("bytes") or 0)
            or sha256_file(path) != str(row.get("sha256") or "")
            or str(upstream.get("sha256") or "")
            != str(row.get("upstream_sha256") or "")
        ):
            raise ValueError(f"humanities derivative file mismatch: {filename}")
        parquet = pq.ParquetFile(path)
        if (
            int(parquet.metadata.num_rows) != int(row.get("rows") or 0)
            or not required_columns.issubset(parquet.schema_arrow.names)
        ):
            raise ValueError(f"humanities derivative parquet mismatch: {path}")
        seen_outputs.add(filename)
        seen_upstreams.add(upstream_filename)
        inventory_files.append(
            {
                "path": filename,
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
                "rows": int(row["rows"]),
                "upstream_filename": upstream_filename,
                "upstream_sha256": str(row["upstream_sha256"]),
            }
        )
        acquired_files.append(
            {
                "filename": filename,
                "path": str(path),
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
                "rows": int(row["rows"]),
                "upstream_filename": upstream_filename,
                "upstream_sha256": str(row["upstream_sha256"]),
            }
        )
    if not inventory_files or seen_upstreams != set(upstream_files):
        raise ValueError("humanities derivative does not cover every upstream file")
    lineage = {
        "derivative_report": str(report_path),
        "derivative_report_sha256": _json_sha256(report_path),
        "derivative_filter_fingerprint_sha256": str(
            fingerprint
        ),
    }
    inventory_source = {
        **source,
        **lineage,
        "files": inventory_files,
    }
    acquired_source = {
        **source,
        **lineage,
        "files": acquired_files,
    }
    return inventory_source, acquired_source


def build_bundle(
    *,
    hf_acquisition_reports: list[str],
    wikimedia_extraction_reports: list[str],
    output_root: str,
    inventory_output: str,
    acquisition_output: str,
    derivative_reports: list[str] | None = None,
) -> dict[str, Any]:
    derivatives = list(derivative_reports or [])
    if not hf_acquisition_reports and not wikimedia_extraction_reports and not derivatives:
        raise ValueError("pretrain acquisition bundle requires at least one input report")
    bundle_root = Path(output_root).expanduser().resolve()
    inventory_sources: list[dict[str, Any]] = []
    acquired_sources: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []
    for value in hf_acquisition_reports:
        path = Path(value).expanduser().resolve()
        selected, acquired = _verified_hf_sources(path, bundle_root=bundle_root)
        inventory_sources.extend(selected)
        acquired_sources.extend(acquired)
        inputs.append({"kind": "hf_acquisition", "path": str(path), "sha256": _json_sha256(path)})
    for value in wikimedia_extraction_reports:
        path = Path(value).expanduser().resolve()
        selected, acquired = _wikimedia_sources(path, bundle_root=bundle_root)
        inventory_sources.extend(selected)
        acquired_sources.extend(acquired)
        inputs.append(
            {"kind": "wikimedia_extraction", "path": str(path), "sha256": _json_sha256(path)}
        )
    for value in derivatives:
        path = Path(value).expanduser().resolve()
        selected, acquired = _derivative_source(path, bundle_root=bundle_root)
        inventory_sources.append(selected)
        acquired_sources.append(acquired)
        inputs.append(
            {
                "kind": "humanities_derivative",
                "path": str(path),
                "sha256": _json_sha256(path),
            }
        )
    names = [str(source.get("name") or "") for source in inventory_sources]
    if not names or len(names) != len(set(names)):
        raise ValueError("pretrain acquisition bundle source names must be non-empty and unique")
    inventory_path = Path(inventory_output).expanduser().resolve()
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "status": "complete",
        "purpose": "Unified fresh-input inventory for one globally deduplicated pretrain clean run",
        "input_reports": inputs,
        "sources": inventory_sources,
        "selected_file_count": sum(len(source["files"]) for source in inventory_sources),
        "selected_bytes": sum(
            int(row["bytes"])
            for source in inventory_sources
            for row in source["files"]
        ),
    }
    write_json_atomic(inventory_path, inventory, ensure_ascii=False, sort_keys=True, make_parents=True)
    acquisition_path = Path(acquisition_output).expanduser().resolve()
    acquisition = {
        "schema": ACQUISITION_SCHEMA,
        "status": "complete",
        "inventory_path": str(inventory_path),
        "inventory_sha256": _json_sha256(inventory_path),
        "output_root": str(bundle_root),
        "input_reports": inputs,
        "sources": acquired_sources,
        "total_bytes": inventory["selected_bytes"],
    }
    write_json_atomic(
        acquisition_path,
        acquisition,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return acquisition


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one verified acquisition lineage from HF and Wikimedia inputs."
    )
    parser.add_argument("--hf-acquisition", action="append", default=[])
    parser.add_argument("--wikimedia-extraction", action="append", default=[])
    parser.add_argument("--humanities-derivative", action="append", default=[])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--inventory-output", required=True)
    parser.add_argument("--acquisition-output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_bundle(
        hf_acquisition_reports=[str(value) for value in args.hf_acquisition],
        wikimedia_extraction_reports=[str(value) for value in args.wikimedia_extraction],
        output_root=str(args.output_root),
        inventory_output=str(args.inventory_output),
        acquisition_output=str(args.acquisition_output),
        derivative_reports=[str(value) for value in args.humanities_derivative],
    )
    print(
        f"[DONE] sources={len(report['sources'])} bytes={report['total_bytes']:,} "
        f"output={Path(args.acquisition_output).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
