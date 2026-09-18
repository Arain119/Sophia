from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ml.core.common.io import write_json_atomic
from ml.data.pretrain_document_admission import (
    PostCleanDocumentAdmission,
    load_post_clean_document_admission,
)


REPORT_SCHEMA = "sophia_tokenizer_training_sample_v1"
DEFAULT_SAMPLE_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_TEXT_CHARS = 8192
DEFAULT_MAX_FILES_PER_SOURCE = 512
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("text", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_byte_budgets(
    *, total_bytes: int, source_quotas: Mapping[str, int]
) -> dict[str, int]:
    total_quota = sum(int(value) for value in source_quotas.values())
    if total_bytes <= 0 or total_quota <= 0:
        raise ValueError("sample bytes and source quotas must be positive")
    if any(int(value) <= 0 for value in source_quotas.values()):
        raise ValueError("all source quotas must be positive")
    exact = {
        source: float(total_bytes) * int(quota) / float(total_quota)
        for source, quota in source_quotas.items()
    }
    budgets = {source: int(math.floor(value)) for source, value in exact.items()}
    remainder = int(total_bytes) - sum(budgets.values())
    order = sorted(
        source_quotas,
        key=lambda source: (-(exact[source] - budgets[source]), str(source)),
    )
    for source in order[:remainder]:
        budgets[source] += 1
    if any(value <= 0 for value in budgets.values()):
        raise ValueError("sample budget is too small to cover every source")
    return dict(sorted(budgets.items()))


def _evenly_spaced_files(paths: list[Path], maximum: int) -> list[Path]:
    if maximum <= 0 or len(paths) <= maximum:
        return list(paths)
    if maximum == 1:
        return [paths[len(paths) // 2]]
    last = len(paths) - 1
    indices = [int(round(index * last / float(maximum - 1))) for index in range(maximum)]
    return [paths[index] for index in indices]


def _source_files(clean_root: Path, source: str) -> list[Path]:
    split_root = clean_root / source / "train"
    return sorted(split_root.rglob("*.parquet")) if split_root.is_dir() else []


def _iter_admitted_texts(
    path: Path,
    *,
    expected_source: str,
    admission: PostCleanDocumentAdmission,
    counters: Counter[str],
    batch_size: int,
) -> Iterator[str]:
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    required = {"text", "source"} | admission.required_columns
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"tokenizer sample input is missing columns {missing}: {path}")
    for batch in parquet.iter_batches(batch_size=batch_size, columns=sorted(required)):
        for row in batch.to_pylist():
            counters["documents_scanned"] += 1
            source = str(row.get("source") or "")
            if source != expected_source:
                raise ValueError(
                    f"source column does not match clean layout: expected={expected_source} "
                    f"actual={source} path={path}"
                )
            reason = admission.exclusion_reason(row)
            if reason:
                counters["documents_excluded"] += 1
                counters[f"excluded:{reason}"] += 1
                continue
            text = str(row.get("text") or "").strip()
            if text:
                yield text


def _trim_to_utf8_budget(text: str, maximum: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= maximum:
        return text
    return encoded[:maximum].decode("utf-8", errors="ignore").strip()


def _write_source_sample(
    *,
    clean_root: Path,
    output_root: Path,
    source: str,
    byte_budget: int,
    admission: PostCleanDocumentAdmission,
    max_text_chars: int,
    max_files_per_source: int,
    batch_size: int,
) -> dict[str, Any]:
    available_files = _source_files(clean_root, source)
    if not available_files:
        raise FileNotFoundError(f"no clean train parquet files for source: {source}")
    selected_files = _evenly_spaced_files(available_files, max_files_per_source)
    output_path = output_root / source / "train" / "part-00000.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    buffered_texts: list[str] = []
    buffered_sources: list[str] = []
    bytes_written = 0
    writer = pq.ParquetWriter(output_path, OUTPUT_SCHEMA, compression="zstd")

    def flush() -> None:
        if not buffered_texts:
            return
        writer.write_table(
            pa.Table.from_arrays(
                [pa.array(buffered_texts), pa.array(buffered_sources)],
                schema=OUTPUT_SCHEMA,
            )
        )
        buffered_texts.clear()
        buffered_sources.clear()

    try:
        for file_index, path in enumerate(selected_files):
            files_left = len(selected_files) - file_index
            file_target = bytes_written + int(
                math.ceil((byte_budget - bytes_written) / float(files_left))
            )
            for text in _iter_admitted_texts(
                path,
                expected_source=source,
                admission=admission,
                counters=counters,
                batch_size=batch_size,
            ):
                if max_text_chars > 0 and len(text) > max_text_chars:
                    text = text[:max_text_chars].strip()
                    counters["documents_truncated_by_chars"] += 1
                remaining = byte_budget - bytes_written
                if remaining <= 0:
                    break
                encoded_bytes = len(text.encode("utf-8"))
                if encoded_bytes > remaining:
                    text = _trim_to_utf8_budget(text, remaining)
                    counters["documents_truncated_by_bytes"] += 1
                    encoded_bytes = len(text.encode("utf-8"))
                if not text or encoded_bytes <= 0:
                    continue
                buffered_texts.append(text)
                buffered_sources.append(source)
                bytes_written += encoded_bytes
                counters["documents_written"] += 1
                if len(buffered_texts) >= batch_size:
                    flush()
                if bytes_written >= file_target:
                    break
            if bytes_written >= byte_budget:
                break
        flush()
    finally:
        writer.close()
    if bytes_written != byte_budget:
        raise RuntimeError(
            f"source could not fill tokenizer sample byte budget: source={source} "
            f"expected={byte_budget} actual={bytes_written}"
        )
    exclusions = {
        key.removeprefix("excluded:"): int(value)
        for key, value in sorted(counters.items())
        if key.startswith("excluded:")
    }
    return {
        "byte_budget": int(byte_budget),
        "utf8_bytes": int(bytes_written),
        "documents_written": int(counters["documents_written"]),
        "documents_scanned": int(counters["documents_scanned"]),
        "documents_excluded": int(counters["documents_excluded"]),
        "documents_excluded_by": exclusions,
        "documents_truncated_by_chars": int(counters["documents_truncated_by_chars"]),
        "documents_truncated_by_bytes": int(counters["documents_truncated_by_bytes"]),
        "available_files": len(available_files),
        "selected_files": len(selected_files),
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": _sha256(output_path),
    }


def build_sample(
    *,
    clean_root: Path,
    resolved_mix_plan_path: Path,
    admission_policy_path: Path,
    output_root: Path,
    sample_bytes: int,
    max_text_chars: int,
    max_files_per_source: int,
    batch_size: int,
    workers: int = 1,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    plan = _load_json(resolved_mix_plan_path)
    if str(plan.get("schema")) != "sophia_resolved_pretrain_mix_plan_v1":
        raise ValueError("unsupported resolved pretrain mix plan")
    if str(plan.get("status")) != "ready":
        raise ValueError("tokenizer sampling requires a ready resolved mix plan")
    raw_quotas = plan.get("source_train_quotas")
    if not isinstance(raw_quotas, dict) or not raw_quotas:
        raise ValueError("resolved mix plan has no source_train_quotas")
    source_quotas = {str(key): int(value) for key, value in raw_quotas.items()}
    policy = _load_json(admission_policy_path)
    admission = load_post_clean_document_admission(
        policy,
        policy_path=admission_policy_path,
    )
    budgets = _integer_byte_budgets(
        total_bytes=int(sample_bytes), source_quotas=source_quotas
    )
    output_root.mkdir(parents=True, exist_ok=True)
    source_names = sorted(budgets)
    if int(workers) == 1:
        sources = {
            source: _write_source_sample(
                clean_root=clean_root,
                output_root=output_root,
                source=source,
                byte_budget=budgets[source],
                admission=admission,
                max_text_chars=int(max_text_chars),
                max_files_per_source=int(max_files_per_source),
                batch_size=int(batch_size),
            )
            for source in source_names
        }
    else:
        completed: dict[str, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = {
                executor.submit(
                    _write_source_sample,
                    clean_root=clean_root,
                    output_root=output_root,
                    source=source,
                    byte_budget=budgets[source],
                    admission=admission,
                    max_text_chars=int(max_text_chars),
                    max_files_per_source=int(max_files_per_source),
                    batch_size=int(batch_size),
                ): source
                for source in source_names
            }
            for future in as_completed(futures):
                source = futures[future]
                completed[source] = future.result()
        sources = {source: completed[source] for source in source_names}
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "clean_root": str(clean_root),
        "output_root": str(output_root),
        "sample_utf8_bytes": sum(int(row["utf8_bytes"]) for row in sources.values()),
        "source_byte_budgets": budgets,
        "source_token_quotas": dict(sorted(source_quotas.items())),
        "max_text_chars": int(max_text_chars),
        "max_files_per_source": int(max_files_per_source),
        "batch_size": int(batch_size),
        "workers": int(workers),
        "post_clean_document_admission": admission.to_report(),
        "resolved_mix_plan": {
            "path": str(resolved_mix_plan_path),
            "sha256": _sha256(resolved_mix_plan_path),
        },
        "admission_policy": {
            "path": str(admission_policy_path),
            "sha256": _sha256(admission_policy_path),
        },
        "sources": sources,
    }
    write_json_atomic(str(output_root / "tokenizer_training_sample_report.json"), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an admitted, final-mix-aligned tokenizer training sample."
    )
    parser.add_argument("--clean_root", required=True)
    parser.add_argument("--resolved_mix_plan", required=True)
    parser.add_argument("--admission_policy", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--sample_bytes", type=int, default=DEFAULT_SAMPLE_BYTES)
    parser.add_argument("--max_text_chars", type=int, default=DEFAULT_MAX_TEXT_CHARS)
    parser.add_argument(
        "--max_files_per_source", type=int, default=DEFAULT_MAX_FILES_PER_SOURCE
    )
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.sample_bytes) <= 0:
        raise SystemExit("sample_bytes must be > 0")
    if int(args.max_text_chars) <= 0:
        raise SystemExit("max_text_chars must be > 0")
    if int(args.max_files_per_source) <= 0:
        raise SystemExit("max_files_per_source must be > 0")
    if int(args.batch_size) <= 0:
        raise SystemExit("batch_size must be > 0")
    if int(args.workers) <= 0:
        raise SystemExit("workers must be > 0")
    report = build_sample(
        clean_root=Path(args.clean_root).resolve(),
        resolved_mix_plan_path=Path(args.resolved_mix_plan).resolve(),
        admission_policy_path=Path(args.admission_policy).resolve(),
        output_root=Path(args.output_root).resolve(),
        sample_bytes=int(args.sample_bytes),
        max_text_chars=int(args.max_text_chars),
        max_files_per_source=int(args.max_files_per_source),
        batch_size=int(args.batch_size),
        workers=int(args.workers),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
