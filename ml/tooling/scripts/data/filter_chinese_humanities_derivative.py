from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ml.core.common.io import write_json_atomic
from ml.tooling.core import pretrain_taxonomy_user as taxonomy_mod
from ml.tooling.scripts.data.clean_pretrain_corpus import (
    _apply_legacy_overlap,
    _load_chinese_quality_proxy,
    _load_legacy_content_exclusion,
    _prepare_clean_candidates,
)
from ml.tooling.scripts.data.download_hf_source_inventory import sha256_file


REPORT_SCHEMA = "sophia_chinese_humanities_derivative_v1"
ACQUISITION_SCHEMA = "sophia_hf_acquisition_report_v1"
INVENTORY_SCHEMA = "sophia_hf_source_inventory_v2"
HUMANITIES_TAXONOMIES = (
    "humanities_history",
    "humanities_philosophy",
    "social_psychology",
    "literature_classic",
    "literature_classical",
)
SOCIAL_PSYCHOLOGY_PRECISION_POLICY = (
    "strong_subject_anchor_in_first_240_chars_and_at_least_two_anchors_v2"
)
_RE_SOCIAL_PSYCHOLOGY_TOPIC = re.compile(
    r"(社会心理学|心理学|认知心理|人格心理|发展心理|教育心理|临床心理|变态心理|"
    r"心理健康|心理咨询|心理治疗|心理测量|心理障碍|心理机制|群体心理|"
    r"社会学|社会结构|社会阶层|社会分层|社会认同|人口学|人类学)|"
    r"\b(psychology|sociology|social psychology|cognitive psychology|"
    r"behavioral science)\b",
    re.IGNORECASE,
)
_RE_SOCIAL_PSYCHOLOGY_STRONG_PREFIX = re.compile(
    r"(社会心理学|心理学|认知心理|人格心理|发展心理|教育心理|临床心理|变态心理|"
    r"心理咨询|心理治疗|心理测量|心理障碍|心理机制|群体心理|"
    r"社会学|社会结构|社会阶层|社会分层|社会认同|人口学|人类学)|"
    r"\b(psychology|sociology|social psychology|cognitive psychology|"
    r"behavioral science)\b",
    re.IGNORECASE,
)
OUTPUT_SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("upstream_file", pa.string()),
        ("upstream_row", pa.int64()),
        ("upstream_file_sha256", pa.string()),
        ("upstream_text_sha256", pa.string()),
        ("upstream_score", pa.float32()),
        ("upstream_source", pa.string()),
        ("chinese_quality_proxy_score", pa.float32()),
        ("humanities_taxonomy", pa.string()),
        ("document_sha256", pa.string()),
        ("legacy_exact_overlap", pa.bool_()),
        ("legacy_near_overlap", pa.bool_()),
    ]
)


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


def _upstream_files(
    *, acquisition_path: Path, source_name: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path, str]:
    acquisition = _load_json(acquisition_path)
    if str(acquisition.get("schema")) != ACQUISITION_SCHEMA:
        raise ValueError("unsupported upstream acquisition report")
    inventory_path = Path(str(acquisition.get("inventory_path") or "")).resolve()
    inventory = _load_json(inventory_path)
    if str(inventory.get("schema")) != INVENTORY_SCHEMA:
        raise ValueError("unsupported upstream source inventory")
    inventory_sha256 = _json_sha256(inventory_path)
    if inventory_sha256 != str(acquisition.get("inventory_sha256") or ""):
        raise ValueError("upstream acquisition inventory hash mismatch")
    selected = [
        row
        for row in inventory.get("sources", [])
        if isinstance(row, dict) and str(row.get("name") or "") == source_name
    ]
    acquired = [
        row
        for row in acquisition.get("sources", [])
        if isinstance(row, dict) and str(row.get("name") or "") == source_name
    ]
    if len(selected) != 1 or len(acquired) != 1:
        raise ValueError(f"upstream source must resolve exactly once: {source_name}")
    expected = {
        str(row.get("path") or ""): row
        for row in selected[0].get("files", [])
        if isinstance(row, dict)
    }
    actual = {
        str(row.get("filename") or ""): row
        for row in acquired[0].get("files", [])
        if isinstance(row, dict)
    }
    if not expected or set(expected) != set(actual):
        raise ValueError("upstream acquisition file list does not match inventory")
    acquisition_root = Path(str(acquisition.get("output_root") or "")).resolve()
    rows: list[dict[str, Any]] = []
    for filename, expected_row in expected.items():
        actual_row = actual[filename]
        path = Path(str(actual_row.get("path") or "")).resolve()
        expected_bytes = int(expected_row.get("bytes") or 0)
        expected_sha256 = str(expected_row.get("sha256") or "")
        if (
            not path.is_file()
            or not _is_relative_to(path, acquisition_root)
            or int(path.stat().st_size) != expected_bytes
            or int(actual_row.get("bytes") or 0) != expected_bytes
            or str(actual_row.get("sha256") or "") != expected_sha256
        ):
            raise ValueError(f"upstream acquisition file mismatch: {filename}")
        rows.append(
            {
                "filename": filename,
                "path": path,
                "bytes": expected_bytes,
                "sha256": expected_sha256,
            }
        )
    return selected[0], acquired[0], rows, inventory_path, inventory_sha256


def _filter_fingerprint(
    *,
    upstream_source_name: str,
    acquisition_sha256: str,
    inventory_sha256: str,
    policy_sha256: str,
    taxonomy_sha256: str,
    derivative_filter_sha256: str,
    quality_proxy: Any,
) -> tuple[dict[str, Any], str]:
    lineage = {
        "schema": REPORT_SCHEMA,
        "upstream_source_name": str(upstream_source_name),
        "upstream_acquisition_sha256": acquisition_sha256,
        "upstream_inventory_sha256": inventory_sha256,
        "admission_policy_sha256": policy_sha256,
        "taxonomy_module_sha256": taxonomy_sha256,
        "derivative_filter_module_sha256": derivative_filter_sha256,
        "quality_proxy_report_sha256": str(quality_proxy.report_sha256),
        "quality_proxy_model_sha256": str(quality_proxy.model_sha256),
        "quality_proxy_labels_sha256": str(quality_proxy.labels_sha256),
        "quality_proxy_minimum_score": float(quality_proxy.minimum_score),
        "humanities_taxonomies": list(HUMANITIES_TAXONOMIES),
        "social_psychology_precision_policy": SOCIAL_PSYCHOLOGY_PRECISION_POLICY,
    }
    encoded = json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode()
    return lineage, hashlib.sha256(encoded).hexdigest()


def _passes_humanities_precision_gate(*, text: str, taxonomy: str) -> bool:
    if taxonomy != "social_psychology":
        return True
    anchors = _RE_SOCIAL_PSYCHOLOGY_TOPIC.findall(text)
    return (
        _RE_SOCIAL_PSYCHOLOGY_STRONG_PREFIX.search(text[:240]) is not None
        and len(anchors) >= 2
    )


def _aggregate_stats(files: list[dict[str, Any]]) -> dict[str, int]:
    stats: Counter[str] = Counter()
    for row in files:
        stats.update(
            {
                str(key): int(value)
                for key, value in row.get("stats", {}).items()
            }
        )
    return dict(sorted(stats.items()))


def _checkpoint_report(
    *,
    report_path: Path,
    status: str,
    upstream_acquisition_path: Path,
    upstream_acquisition_sha256: str,
    upstream_inventory_path: Path,
    upstream_inventory_sha256: str,
    admission_policy_path: Path,
    admission_policy_sha256: str,
    output_root: Path,
    source: dict[str, Any],
    filter_lineage: dict[str, Any],
    filter_fingerprint_sha256: str,
    files: list[dict[str, Any]],
    input_file_count: int,
) -> dict[str, Any]:
    report = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "upstream_acquisition_report": str(upstream_acquisition_path),
        "upstream_acquisition_sha256": upstream_acquisition_sha256,
        "upstream_inventory": str(upstream_inventory_path),
        "upstream_inventory_sha256": upstream_inventory_sha256,
        "admission_policy": str(admission_policy_path),
        "admission_policy_sha256": admission_policy_sha256,
        "output_root": str(output_root),
        "source": source,
        "filter_lineage": filter_lineage,
        "filter_fingerprint_sha256": filter_fingerprint_sha256,
        "input_file_count": int(input_file_count),
        "files": files,
        "stats": _aggregate_stats(files),
    }
    write_json_atomic(
        report_path,
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return report


def _existing_completed_files(
    *, report_path: Path, fingerprint: str
) -> dict[str, dict[str, Any]]:
    if not report_path.is_file():
        return {}
    report = _load_json(report_path)
    if (
        str(report.get("schema")) != REPORT_SCHEMA
        or str(report.get("filter_fingerprint_sha256") or "") != fingerprint
    ):
        raise ValueError("existing derivative report does not match current filter")
    completed: dict[str, dict[str, Any]] = {}
    for row in report.get("files", []):
        if not isinstance(row, dict):
            continue
        path = Path(str(row.get("path") or "")).resolve()
        if (
            str(row.get("status")) == "complete"
            and path.is_file()
            and int(path.stat().st_size) == int(row.get("bytes") or 0)
            and sha256_file(path) == str(row.get("sha256") or "")
        ):
            completed[str(row.get("upstream_filename") or "")] = row
    return completed


def _process_file(
    *,
    upstream: dict[str, Any],
    source: dict[str, Any],
    output_root: Path,
    quality_proxy: Any,
    legacy_exclusion: Any,
    batch_size: int,
) -> dict[str, Any]:
    input_path = Path(upstream["path"])
    if sha256_file(input_path) != str(upstream["sha256"]):
        raise ValueError(f"upstream file checksum mismatch: {input_path}")
    parquet = pq.ParquetFile(input_path)
    available = set(parquet.schema_arrow.names)
    text_column = str(source.get("text_column") or "text")
    if text_column not in available:
        raise ValueError(f"upstream file lacks text column: {input_path}")
    columns = sorted(({text_column, "score", "source"}) & available)
    relative = Path(str(upstream["filename"]))
    output_path = (output_root / "humanities" / relative.name).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".part")
    partial.unlink(missing_ok=True)
    stats: Counter[str] = Counter()
    source_for_cleaning = dict(source)
    source_for_cleaning["id_column"] = "__sophia_upstream_row"
    writer = pq.ParquetWriter(
        partial,
        OUTPUT_SCHEMA,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
    )
    row_offset = 0
    try:
        for batch in parquet.iter_batches(batch_size=int(batch_size), columns=columns):
            records = batch.to_pylist()
            metadata: dict[int, dict[str, Any]] = {}
            for index, record in enumerate(records):
                upstream_row = row_offset + index
                raw_text = str(record.get(text_column) or "")
                record["__sophia_upstream_row"] = str(upstream_row)
                metadata[upstream_row] = {
                    "upstream_text_sha256": hashlib.sha256(
                        raw_text.encode("utf-8")
                    ).hexdigest(),
                    "upstream_score": record.get("score"),
                    "upstream_source": str(record.get("source") or ""),
                }
            candidates = _prepare_clean_candidates(
                tuple(records),
                source=source_for_cleaning,
                stats=stats,
            )
            candidates = _apply_legacy_overlap(
                candidates,
                legacy_exclusion=legacy_exclusion,
                stats=stats,
            )
            texts = [str(candidate["text"]) for candidate in candidates]
            scores = quality_proxy.score(texts) if texts else []
            retained: list[dict[str, Any]] = []
            for candidate, score in zip(candidates, scores, strict=True):
                value = float(score)
                if value < float(quality_proxy.minimum_score):
                    stats["drop_chinese_quality_proxy"] += 1
                    continue
                stats["chinese_quality_proxy_accept"] += 1
                taxonomy = taxonomy_mod.infer_user_taxonomy_label(
                    str(candidate["text"])
                )
                if taxonomy not in HUMANITIES_TAXONOMIES:
                    stats["drop_non_humanities_taxonomy"] += 1
                    continue
                if not _passes_humanities_precision_gate(
                    text=str(candidate["text"]), taxonomy=taxonomy
                ):
                    stats["drop_low_precision_social_psychology"] += 1
                    continue
                upstream_row = int(candidate["document_id"])
                upstream_metadata = metadata[upstream_row]
                upstream_score = upstream_metadata["upstream_score"]
                retained.append(
                    {
                        "text": str(candidate["text"]),
                        "upstream_file": str(upstream["filename"]),
                        "upstream_row": upstream_row,
                        "upstream_file_sha256": str(upstream["sha256"]),
                        "upstream_text_sha256": str(
                            upstream_metadata["upstream_text_sha256"]
                        ),
                        "upstream_score": (
                            float(upstream_score)
                            if upstream_score is not None
                            else None
                        ),
                        "upstream_source": str(
                            upstream_metadata["upstream_source"]
                        ),
                        "chinese_quality_proxy_score": value,
                        "humanities_taxonomy": taxonomy,
                        "document_sha256": str(candidate["document_sha256"]),
                        "legacy_exact_overlap": bool(
                            candidate["legacy_exact_overlap"]
                        ),
                        "legacy_near_overlap": bool(candidate["legacy_near_overlap"]),
                    }
                )
                stats["documents_out"] += 1
                stats[f"documents_out_{taxonomy}"] += 1
            if retained:
                writer.write_table(pa.Table.from_pylist(retained, schema=OUTPUT_SCHEMA))
            row_offset += batch.num_rows
    finally:
        writer.close()
    if row_offset != int(parquet.metadata.num_rows):
        raise RuntimeError(f"upstream row traversal mismatch: {input_path}")
    partial.replace(output_path)
    return {
        "status": "complete",
        "filename": f"humanities/{relative.name}",
        "path": str(output_path),
        "bytes": int(output_path.stat().st_size),
        "sha256": sha256_file(output_path),
        "rows": int(stats["documents_out"]),
        "upstream_filename": str(upstream["filename"]),
        "upstream_bytes": int(upstream["bytes"]),
        "upstream_sha256": str(upstream["sha256"]),
        "upstream_rows": int(parquet.metadata.num_rows),
        "stats": dict(sorted(stats.items())),
    }


_WORKER_SOURCE: dict[str, Any] | None = None
_WORKER_OUTPUT_ROOT: Path | None = None
_WORKER_QUALITY_PROXY: Any = None
_WORKER_LEGACY_EXCLUSION: Any = None
_WORKER_BATCH_SIZE = 0


def _initialize_filter_worker(
    source: dict[str, Any],
    admission_policy: str,
    output_root: str,
    batch_size: int,
) -> None:
    global _WORKER_SOURCE
    global _WORKER_OUTPUT_ROOT
    global _WORKER_QUALITY_PROXY
    global _WORKER_LEGACY_EXCLUSION
    global _WORKER_BATCH_SIZE
    policy = _load_json(Path(admission_policy))
    _WORKER_SOURCE = dict(source)
    _WORKER_OUTPUT_ROOT = Path(output_root)
    _WORKER_QUALITY_PROXY = _load_chinese_quality_proxy(policy=policy)
    _WORKER_LEGACY_EXCLUSION = _load_legacy_content_exclusion(policy=policy)
    _WORKER_BATCH_SIZE = int(batch_size)


def _process_file_worker(upstream: dict[str, Any]) -> dict[str, Any]:
    if (
        _WORKER_SOURCE is None
        or _WORKER_OUTPUT_ROOT is None
        or _WORKER_QUALITY_PROXY is None
        or _WORKER_LEGACY_EXCLUSION is None
        or _WORKER_BATCH_SIZE <= 0
    ):
        raise RuntimeError("humanities filter worker is not initialized")
    return _process_file(
        upstream=upstream,
        source=_WORKER_SOURCE,
        output_root=_WORKER_OUTPUT_ROOT,
        quality_proxy=_WORKER_QUALITY_PROXY,
        legacy_exclusion=_WORKER_LEGACY_EXCLUSION,
        batch_size=_WORKER_BATCH_SIZE,
    )


def filter_humanities_derivative(
    *,
    acquisition_report: str,
    source_name: str,
    admission_policy: str,
    output_root: str,
    report_path: str,
    batch_size: int = 2048,
    workers: int = 1,
) -> dict[str, Any]:
    if not 128 <= int(batch_size) <= 65_536:
        raise ValueError("batch_size must be in [128, 65536]")
    worker_count = int(workers)
    if not 1 <= worker_count <= 8:
        raise ValueError("workers must be in [1, 8]")
    acquisition_path = Path(acquisition_report).expanduser().resolve()
    policy_path = Path(admission_policy).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    output_report = Path(report_path).expanduser().resolve()
    source, _acquired, upstream_files, inventory_path, inventory_sha256 = (
        _upstream_files(acquisition_path=acquisition_path, source_name=source_name)
    )
    policy = _load_json(policy_path)
    quality_proxy = _load_chinese_quality_proxy(policy=policy)
    if not quality_proxy.enabled or float(quality_proxy.minimum_score) != 0.68:
        raise ValueError("humanities derivative requires quality proxy threshold 0.68")
    taxonomy_path = Path(str(taxonomy_mod.__file__)).resolve()
    derivative_filter_path = Path(__file__).resolve()
    acquisition_sha256 = _json_sha256(acquisition_path)
    policy_sha256 = _json_sha256(policy_path)
    filter_lineage, fingerprint = _filter_fingerprint(
        upstream_source_name=source_name,
        acquisition_sha256=acquisition_sha256,
        inventory_sha256=inventory_sha256,
        policy_sha256=policy_sha256,
        taxonomy_sha256=sha256_file(taxonomy_path),
        derivative_filter_sha256=sha256_file(derivative_filter_path),
        quality_proxy=quality_proxy,
    )
    output_source = {
        **{
            key: value
            for key, value in source.items()
            if key
            not in {
                "catalog_files",
                "files",
                "name",
                "selector_evidence",
                "selected_bytes",
                "selected_file_count",
            }
        },
        "name": "opencsg_fineweb_edu_zh_3_4_humanities_v1",
        "upstream_source_name": source_name,
        "domain": "quality_gated_simplified_chinese_humanities_web",
        "quality_score": (
            "upstream_band_3_4_plus_local_quality_proxy_0_68_"
            "and_five_class_humanities_filter"
        ),
        "derivative_schema": REPORT_SCHEMA,
        "derivative_filter_fingerprint_sha256": fingerprint,
        "deduplication_priority": 0,
    }
    completed = _existing_completed_files(
        report_path=output_report,
        fingerprint=fingerprint,
    )
    reports_by_upstream = dict(completed)

    def checkpoint() -> None:
        file_reports = [
            reports_by_upstream[str(upstream["filename"])]
            for upstream in upstream_files
            if str(upstream["filename"]) in reports_by_upstream
        ]
        _checkpoint_report(
            report_path=output_report,
            status="in_progress",
            upstream_acquisition_path=acquisition_path,
            upstream_acquisition_sha256=acquisition_sha256,
            upstream_inventory_path=inventory_path,
            upstream_inventory_sha256=inventory_sha256,
            admission_policy_path=policy_path,
            admission_policy_sha256=policy_sha256,
            output_root=root,
            source=output_source,
            filter_lineage=filter_lineage,
            filter_fingerprint_sha256=fingerprint,
            files=file_reports,
            input_file_count=len(upstream_files),
        )
        print(
            f"[HUMANITIES-FILTER] files={len(file_reports)}/{len(upstream_files)} "
            f"documents={sum(int(item['rows']) for item in file_reports):,}",
            flush=True,
        )

    missing = [
        upstream
        for upstream in upstream_files
        if str(upstream["filename"]) not in reports_by_upstream
    ]
    if worker_count == 1:
        legacy_exclusion = _load_legacy_content_exclusion(policy=policy)
        try:
            for upstream in missing:
                row = _process_file(
                    upstream=upstream,
                    source=source,
                    output_root=root,
                    quality_proxy=quality_proxy,
                    legacy_exclusion=legacy_exclusion,
                    batch_size=int(batch_size),
                )
                reports_by_upstream[str(row["upstream_filename"])] = row
                checkpoint()
        finally:
            legacy_exclusion.close()
    elif missing:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_filter_worker,
            initargs=(source, str(policy_path), str(root), int(batch_size)),
        ) as executor:
            futures = {
                executor.submit(_process_file_worker, upstream): upstream
                for upstream in missing
            }
            for future in as_completed(futures):
                row = future.result()
                reports_by_upstream[str(row["upstream_filename"])] = row
                checkpoint()
    file_reports = [
        reports_by_upstream[str(upstream["filename"])] for upstream in upstream_files
    ]
    return _checkpoint_report(
        report_path=output_report,
        status="complete",
        upstream_acquisition_path=acquisition_path,
        upstream_acquisition_sha256=acquisition_sha256,
        upstream_inventory_path=inventory_path,
        upstream_inventory_sha256=inventory_sha256,
        admission_policy_path=policy_path,
        admission_policy_sha256=policy_sha256,
        output_root=root,
        source=output_source,
        filter_lineage=filter_lineage,
        filter_fingerprint_sha256=fingerprint,
        files=file_reports,
        input_file_count=len(upstream_files),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a provenance-preserving Chinese humanities derivative."
    )
    parser.add_argument("--acquisition-report", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--admission-policy",
        default="configs/data/pretrain_data_admission_policy.json",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = filter_humanities_derivative(
        acquisition_report=str(args.acquisition_report),
        source_name=str(args.source),
        admission_policy=str(args.admission_policy),
        output_root=str(args.output_root),
        report_path=str(args.report),
        batch_size=int(args.batch_size),
        workers=int(args.workers),
    )
    print(
        f"[DONE] documents={report['stats'].get('documents_out', 0):,} "
        f"output={Path(args.report).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
