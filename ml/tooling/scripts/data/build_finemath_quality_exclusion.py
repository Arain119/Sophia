from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import pyarrow.parquet as pq

from ml.core.common.io import write_json_atomic
from ml.tooling.scripts.data.corpus_quality import dedup_normalize
from ml.tooling.scripts.data.clean_pretrain_corpus import clean_source_document
from ml.tooling.scripts.data.download_hf_source_inventory import sha256_file


REPORT_SCHEMA = "sophia_finemath_quality_exclusion_v1"
TARGET_SOURCES = ("finemath_3plus_v1", "finemath_4plus_v1")
DEFAULT_3PLUS_SCORE_THRESHOLD = 2.75
_REASON_PRIORITY = (
    "finemath_academic_mill",
    "finemath_date_calculator",
    "finemath_number_properties",
    "finemath_subscription_preview",
    "finemath_broken_qa",
    "finemath_score_below_threshold",
)

_ACADEMIC_MILL_RE = re.compile(
    r"(?is)(?:"
    r"get\s+professional\s+assignment\s+help|"
    r"professional\s+assignment\s+help\s+cheaply|"
    r"assignment\s+help\s+service|"
    r"online\s+(?:\w+\s+){0,3}homework\s+help\s+service|"
    r"(?:essay|paper|academic)\s+writing\s+service|"
    r"write\s+my\s+(?:essay|paper)|"
    r"do\s+my\s+(?:homework|assignment)|"
    r"buy\s+(?:an?\s+)?(?:essay|paper)|"
    r"our\s+(?:team\s+of\s+)?professional\s+academic\s+writers|"
    r"hire\s+(?:one\s+of\s+)?(?:our\s+)?(?:expert|academic)\s+writers"
    r")"
)
_DATE_TITLE_RE = re.compile(
    r"(?im)^#{0,3}\s*what\s+(?:is\s+the\s+date|date\s+(?:is|will\s+it\s+be))"
    r"\s+\d+\s+days?\s+(?:from|after)\b"
)
_DATE_BODY_RE = re.compile(
    r"(?i)adding\s+\d+\s+days?\s+from\s+.{3,80}\s+is\s+.{3,80}"
)
_DATE_HAND_RE = re.compile(r"(?i)calculating\s+\d+\s+days?\s+from\s+.{3,80}\s+by\s+hand")
_SUBSCRIPTION_PREVIEW_RE = re.compile(
    r"(?is)(?:"
    r"subscribe\s+to\s+view\s+the\s+full\s+document|"
    r"start\s+(?:your|a)\s+48-hour\s+(?:free|complimentary)\s+trial\s+to\s+unlock\s+this\s+answer|"
    r"become\s+a\s+study\.com\s+member\s+to\s+unlock\s+this\s+answer|"
    r"unlock\s+the\s+solution\s+to\s+this\s+question"
    r")"
)
_EMPTY_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?answer\s*:\s*(?=(?:\n\s*(?:#{1,6}\s*)?question\b)|\Z)"
)


@dataclass(frozen=True)
class _FileTask:
    source: dict[str, Any]
    file_info: dict[str, Any]
    score_threshold: float
    batch_size: int
    verify_sha256: bool
    samples_per_reason: int


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _number_properties_template(text: str) -> bool:
    folded = text.casefold()
    return (
        " in braille:" in folded
        and "qr code bar code" in folded
        and "mathematics of no." in folded
        and "images of the number" in folded
    )


def finemath_template_reason(text: str) -> str:
    folded = text.casefold()
    academic_gate = (
        "service" in folded
        or "academic writers" in folded
        or "write my" in folded
        or "do my" in folded
        or "buy essay" in folded
        or "buy paper" in folded
    ) and (
        "assignment" in folded
        or "homework" in folded
        or "essay" in folded
        or "paper" in folded
        or "academic" in folded
    )
    if academic_gate and _ACADEMIC_MILL_RE.search(text):
        return "finemath_academic_mill"
    if "days" in folded and "what" in folded and _DATE_TITLE_RE.search(text) and (
        _DATE_BODY_RE.search(text) or _DATE_HAND_RE.search(text)
    ):
        return "finemath_date_calculator"
    if "braille" in folded and _number_properties_template(text):
        return "finemath_number_properties"
    if ("subscribe" in folded or "unlock" in folded) and _SUBSCRIPTION_PREVIEW_RE.search(
        text
    ):
        return "finemath_subscription_preview"
    if (
        "question" in folded
        and folded.count("answer:") >= 4
        and len(_EMPTY_ANSWER_RE.findall(text)) >= 4
    ):
        return "finemath_broken_qa"
    return ""


def _cleaned_document(
    row: dict[str, Any],
    *,
    source: dict[str, Any],
) -> tuple[str, str]:
    language_column = str(source.get("language_column") or "")
    allowed_languages = {
        str(value).casefold() for value in source.get("allowed_languages", [])
    }
    if language_column and allowed_languages:
        upstream_language = str(row.get(language_column) or "").casefold()
        if upstream_language not in allowed_languages:
            return "", "upstream_language"
    text_column = str(source.get("text_column") or "text")
    decision = clean_source_document(
        str(row.get(text_column) or ""),
        language=str(source.get("language") or "unknown"),
        min_chars=int(source.get("min_chars") or 200),
        source_name=str(source.get("name") or ""),
        source_path=str(row.get("path") or row.get("max_stars_repo_path") or ""),
        reject_ocr_enumeration=bool(source.get("reject_ocr_enumeration", False)),
        minimum_language_ratio=(
            float(source["minimum_language_ratio"])
            if source.get("minimum_language_ratio") is not None
            else None
        ),
    )
    if decision.reason:
        return "", decision.reason
    text = decision.text
    title_column = str(source.get("title_column") or "")
    title = str(row.get(title_column) or "").strip() if title_column else ""
    if title and not text.startswith(title):
        text = f"{title}\n\n{text}"
    return text, ""


def _sample(text: str, *, document_sha256: str, score: float, url: str) -> dict[str, Any]:
    return {
        "document_sha256": document_sha256,
        "score": score,
        "url": url,
        "text_prefix": " ".join(text[:800].split()),
    }


def _scan_file(task: _FileTask) -> dict[str, Any]:
    source = task.source
    file_info = task.file_info
    path = Path(str(file_info.get("path") or "")).expanduser().resolve()
    expected_bytes = int(file_info.get("bytes") or file_info.get("expected_bytes") or 0)
    expected_sha256 = str(file_info.get("sha256") or file_info.get("expected_sha256") or "")
    if not path.is_file() or (expected_bytes and path.stat().st_size != expected_bytes):
        raise ValueError(f"FineMath input size mismatch: {path}")
    actual_sha256 = sha256_file(path) if task.verify_sha256 else expected_sha256
    if task.verify_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(f"FineMath input sha256 mismatch: {path}")

    counters: Counter[str] = Counter()
    token_counters: Counter[str] = Counter()
    matches: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = {}
    exclusions: dict[str, str] = {}
    source_name = str(source.get("name") or "")
    columns = {str(source.get("text_column") or "text"), "score", "token_count", "url"}
    language_column = str(source.get("language_column") or "")
    title_column = str(source.get("title_column") or "")
    columns.update(value for value in (language_column, title_column) if value)
    parquet = pq.ParquetFile(path)
    missing = sorted(columns - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"FineMath input is missing columns {missing}: {path}")
    for batch in parquet.iter_batches(batch_size=task.batch_size, columns=sorted(columns)):
        for row in batch.to_pylist():
            counters["documents_scanned"] += 1
            upstream_tokens = int(row.get("token_count") or 0)
            token_counters["tokens_scanned"] += upstream_tokens
            raw_text = str(row.get(str(source.get("text_column") or "text")) or "")
            score = float(row.get("score") or 0.0)
            raw_template_reason = finemath_template_reason(raw_text)
            below_threshold = (
                source_name == "finemath_3plus_v1" and score < task.score_threshold
            )
            if not raw_template_reason and not below_threshold:
                counters["documents_retained"] += 1
                token_counters["tokens_retained"] += upstream_tokens
                continue
            counters["documents_candidates_preclean"] += 1
            token_counters["tokens_candidates_preclean"] += upstream_tokens
            text, clean_reason = _cleaned_document(row, source=source)
            if clean_reason:
                counters[f"cleaner_drop:{clean_reason}"] += 1
                token_counters[f"cleaner_drop:{clean_reason}"] += upstream_tokens
                continue
            counters["documents_cleanable"] += 1
            token_counters["tokens_cleanable"] += upstream_tokens
            normalized = dedup_normalize(text)
            if not normalized:
                counters["cleaner_drop:empty_normalized"] += 1
                token_counters["cleaner_drop:empty_normalized"] += upstream_tokens
                continue
            document_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            template_reason = finemath_template_reason(text)
            if template_reason:
                matches[template_reason] += 1
            if below_threshold:
                matches["finemath_score_below_threshold"] += 1
            reason = template_reason or (
                "finemath_score_below_threshold" if below_threshold else ""
            )
            if not reason:
                counters["documents_retained"] += 1
                token_counters["tokens_retained"] += upstream_tokens
                continue
            counters[f"excluded:{reason}"] += 1
            token_counters[f"excluded:{reason}"] += upstream_tokens
            existing = exclusions.get(document_sha256)
            if existing is None or _REASON_PRIORITY.index(reason) < _REASON_PRIORITY.index(existing):
                exclusions[document_sha256] = reason
            reason_samples = samples.setdefault(reason, [])
            if len(reason_samples) < task.samples_per_reason:
                reason_samples.append(
                    _sample(
                        text,
                        document_sha256=document_sha256,
                        score=score,
                        url=str(row.get("url") or ""),
                    )
                )
    return {
        "source": source_name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual_sha256,
        "counters": dict(counters),
        "token_counters": dict(token_counters),
        "rule_matches": dict(matches),
        "samples": samples,
        "exclusions": exclusions,
    }


def _merge_exclusion(
    exclusions: dict[str, str],
    document_sha256: str,
    reason: str,
) -> None:
    existing = exclusions.get(document_sha256)
    if existing is None or _REASON_PRIORITY.index(reason) < _REASON_PRIORITY.index(existing):
        exclusions[document_sha256] = reason


def _write_manifest(path: Path, exclusions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            for document_sha256, reason in sorted(exclusions.items()):
                handle.write(f"{document_sha256}\t{reason}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_finemath_quality_exclusion(
    *,
    acquisition_path: Path,
    manifest_path: Path,
    report_path: Path,
    score_threshold: float = DEFAULT_3PLUS_SCORE_THRESHOLD,
    workers: int = 1,
    batch_size: int = 2_048,
    verify_sha256: bool = True,
    samples_per_reason: int = 8,
    max_files: int = 0,
) -> dict[str, Any]:
    if workers <= 0 or batch_size <= 0 or samples_per_reason < 0:
        raise ValueError("workers and batch size must be positive; samples cannot be negative")
    if not 0.0 < score_threshold <= 4.0:
        raise ValueError("FineMath 3plus score threshold must be in (0, 4]")
    acquisition_path = acquisition_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    acquisition = _load_json(acquisition_path)
    sources = {
        str(source.get("name") or ""): source
        for source in acquisition.get("sources", [])
        if isinstance(source, dict)
        and str(source.get("name") or "") in TARGET_SOURCES
    }
    missing_sources = sorted(set(TARGET_SOURCES) - set(sources))
    if missing_sources:
        raise ValueError(f"acquisition is missing FineMath sources: {missing_sources}")
    tasks = [
        _FileTask(
            source=sources[source_name],
            file_info=file_info,
            score_threshold=float(score_threshold),
            batch_size=int(batch_size),
            verify_sha256=bool(verify_sha256),
            samples_per_reason=int(samples_per_reason),
        )
        for source_name in TARGET_SOURCES
        for file_info in sources[source_name].get("files", [])
        if isinstance(file_info, dict)
    ]
    if max_files > 0:
        tasks = tasks[:max_files]
    if not tasks:
        raise ValueError("acquisition has no selected FineMath files")

    started = time.time()
    if workers == 1:
        file_results = [_scan_file(task) for task in tasks]
    else:
        completed: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_scan_file, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                print(
                    f"[FINEMATH] files={len(completed)}/{len(tasks)} "
                    f"source={result['source']} documents="
                    f"{int(result['counters'].get('documents_scanned', 0)):,}",
                    flush=True,
                )
        file_results = completed

    exclusions: dict[str, str] = {}
    source_counters: dict[str, Counter[str]] = {
        source: Counter() for source in TARGET_SOURCES
    }
    source_token_counters: dict[str, Counter[str]] = {
        source: Counter() for source in TARGET_SOURCES
    }
    source_matches: dict[str, Counter[str]] = {
        source: Counter() for source in TARGET_SOURCES
    }
    samples: dict[str, list[dict[str, Any]]] = {}
    for result in file_results:
        source = str(result["source"])
        source_counters[source].update(result["counters"])
        source_token_counters[source].update(result["token_counters"])
        source_matches[source].update(result["rule_matches"])
        for reason, rows in result["samples"].items():
            retained_samples = samples.setdefault(reason, [])
            for row in rows:
                if len(retained_samples) >= samples_per_reason:
                    break
                retained_samples.append(row)
        for document_sha256, reason in result["exclusions"].items():
            _merge_exclusion(exclusions, document_sha256, reason)
    _write_manifest(manifest_path, exclusions)
    manifest_sha256 = sha256_file(manifest_path)
    manifest_reasons = Counter(exclusions.values())
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete" if max_files <= 0 else "partial",
        "acquisition": {
            "path": str(acquisition_path),
            "sha256": sha256_file(acquisition_path),
        },
        "score_policy": {
            "source": "finemath_3plus_v1",
            "column": "score",
            "minimum_score": float(score_threshold),
        },
        "template_rules": list(_REASON_PRIORITY[:-1]),
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "entries": len(exclusions),
            "reason_counts": dict(sorted(manifest_reasons.items())),
        },
        "sources": {
            source: {
                "counters": dict(sorted(source_counters[source].items())),
                "token_counters": dict(
                    sorted(source_token_counters[source].items())
                ),
                "rule_matches": dict(sorted(source_matches[source].items())),
            }
            for source in TARGET_SOURCES
        },
        "files": [
            {
                key: value
                for key, value in result.items()
                if key not in {"exclusions", "samples"}
            }
            for result in sorted(file_results, key=lambda row: str(row["path"]))
        ],
        "samples": {key: value for key, value in sorted(samples.items())},
        "duration_seconds": float(time.time() - started),
        "workers": int(workers),
        "batch_size": int(batch_size),
        "input_sha256_verified": bool(verify_sha256),
    }
    write_json_atomic(
        str(report_path),
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a hash-pinned FineMath semantic-quality exclusion manifest."
    )
    parser.add_argument("--acquisition", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_3PLUS_SCORE_THRESHOLD,
    )
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--samples-per-reason", type=int, default=8)
    parser.add_argument("--skip-sha256-verification", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_finemath_quality_exclusion(
        acquisition_path=Path(args.acquisition),
        manifest_path=Path(args.manifest),
        report_path=Path(args.report),
        score_threshold=float(args.score_threshold),
        workers=int(args.workers),
        batch_size=int(args.batch_size),
        verify_sha256=not bool(args.skip_sha256_verification),
        samples_per_reason=int(args.samples_per_reason),
        max_files=int(args.max_files),
    )
    manifest = report["manifest"]
    print(
        f"[DONE] status={report['status']} entries={manifest['entries']:,} "
        f"sha256={manifest['sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
