from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from opencc import OpenCC
import pyarrow as pa
import pyarrow.parquet as pq

from ml.core.common.io import write_json_atomic
from ml.data.token_shards.tokenizer_fingerprint import compute_tokenizer_bundle_sha1
from ml.integrations.adapters.hf.tokenizer import load_local_tokenizer
from ml.tooling.core.pretrain_taxonomy_user import infer_user_taxonomy_label
from ml.tooling.scripts.data.clean_pretrain_corpus import (
    _apply_legacy_overlap,
    _band_keys,
    _load_chinese_quality_proxy,
    _load_legacy_content_exclusion,
    _prepare_clean_candidates,
)
from ml.tooling.scripts.data.download_hf_source_inventory import sha256_file


REPORT_SCHEMA = "sophia_modern_chinese_candidate_profile_v1"
INVENTORY_SCHEMA = "sophia_hf_source_inventory_v2"
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _distributed_indexes(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        raise ValueError("distributed sample sizes must be positive")
    selected = min(total, count)
    if selected == 1:
        return [0]
    return sorted({index * (total - 1) // (selected - 1) for index in range(selected)})


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    result = np.quantile(
        np.asarray(values, dtype=np.float64), [0.0, 0.1, 0.5, 0.9, 1.0]
    )
    return {
        key: float(value)
        for key, value in zip(("min", "p10", "p50", "p90", "max"), result, strict=True)
    }


def _changed_positions(original: str, converted: str) -> int:
    return sum(
        left != right for left, right in zip_longest(original, converted, fillvalue="")
    )


def _sample_near_duplicate(
    signature: tuple[int, ...],
    bands: dict[bytes, list[tuple[int, ...]]],
    *,
    threshold: float = 0.75,
) -> bool:
    candidates: set[tuple[int, ...]] = set()
    keys = _band_keys(signature)
    for key in keys:
        candidates.update(bands.get(key, []))
    current = set(signature)
    matched = any(
        len(current & set(candidate)) / float(max(min(len(current), len(candidate)), 1))
        >= threshold
        for candidate in candidates
    )
    for key in keys:
        bands[key].append(signature)
    return matched


def profile_candidate(
    *,
    inventory_path: str,
    raw_root: str,
    source_name: str,
    admission_policy: str,
    tokenizer_path: str,
    output_path: str,
    sample_files: int = 64,
    rows_per_file: int = 64,
    selected_file_indexes: list[int] | None = None,
) -> dict[str, Any]:
    inventory_file = Path(inventory_path).expanduser().resolve()
    inventory = _load_json(inventory_file)
    if str(inventory.get("schema")) != INVENTORY_SCHEMA:
        raise ValueError("unsupported candidate source inventory")
    matches = [
        row
        for row in inventory.get("sources", [])
        if isinstance(row, dict) and str(row.get("name") or "") == str(source_name)
    ]
    if len(matches) != 1:
        raise ValueError(f"candidate source must resolve exactly once: {source_name}")
    source = matches[0]
    files = source.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("candidate inventory source has no files")
    policy_file = Path(admission_policy).expanduser().resolve()
    policy = _load_json(policy_file)
    expected_tokenizer_sha1 = str(policy.get("required_tokenizer_bundle_sha1") or "")
    actual_tokenizer_sha1 = compute_tokenizer_bundle_sha1(tokenizer_path)
    if expected_tokenizer_sha1 and expected_tokenizer_sha1 != actual_tokenizer_sha1:
        raise ValueError("candidate profile tokenizer does not match admission policy")
    quality_proxy = _load_chinese_quality_proxy(policy=policy)
    legacy = _load_legacy_content_exclusion(policy=policy)
    tokenizer = load_local_tokenizer(tokenizer_path, model_max_length=1_000_000_000)
    if selected_file_indexes:
        file_indexes = [int(value) for value in selected_file_indexes]
        if len(file_indexes) != len(set(file_indexes)):
            raise ValueError("explicit candidate file indexes contain duplicates")
        if any(index < 0 or index >= len(files) for index in file_indexes):
            raise ValueError("explicit candidate file index is out of range")
        sampling_method = "explicit_inventory_file_indexes"
    else:
        file_indexes = _distributed_indexes(len(files), int(sample_files))
        sampling_method = "evenly_spaced_file_and_row_indexes"
    root = Path(raw_root).expanduser().resolve() / str(source_name)
    stats: Counter[str] = Counter()
    upstream_scores: list[float] = []
    local_scores: list[float] = []
    raw_sources: Counter[str] = Counter()
    taxonomy_docs: Counter[str] = Counter()
    taxonomy_tokens: Counter[str] = Counter()
    accepted_taxonomy_docs: Counter[str] = Counter()
    accepted_taxonomy_tokens: Counter[str] = Counter()
    sample_files_report: list[dict[str, Any]] = []
    exact_hashes: set[str] = set()
    near_bands: dict[bytes, list[tuple[int, ...]]] = defaultdict(list)
    t2s = OpenCC("t2s")
    s2t = OpenCC("s2t")
    try:
        for ordinal, file_index in enumerate(file_indexes, start=1):
            row = files[file_index]
            if not isinstance(row, dict):
                raise ValueError("candidate inventory file rows must be objects")
            path = root / str(row.get("path") or "")
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing distributed candidate sample file: {path}"
                )
            if int(path.stat().st_size) != int(row.get("bytes") or 0) or sha256_file(
                path
            ) != str(row.get("sha256") or ""):
                raise ValueError(f"candidate sample file checksum mismatch: {path}")
            parquet = pq.ParquetFile(path)
            available = set(parquet.schema_arrow.names)
            required = {str(source.get("text_column") or "text")}
            configured = {
                str(source.get(key) or "").split(".", 1)[0]
                for key in ("id_column", "url_column", "title_column")
                if str(source.get(key) or "")
            }
            missing = (required | configured) - available
            if missing:
                raise ValueError(
                    f"candidate file lacks declared columns {sorted(missing)}: {path}"
                )
            columns = sorted((required | {"score", "source"}) & available)
            table = pq.read_table(path, columns=columns)
            row_indexes = _distributed_indexes(table.num_rows, int(rows_per_file))
            sampled = table.take(pa.array(row_indexes, type=pa.int64())).to_pylist()
            file_stats: Counter[str] = Counter()
            pending = _prepare_clean_candidates(
                tuple(sampled),
                source=source,
                stats=file_stats,
            )
            pending = _apply_legacy_overlap(
                pending,
                legacy_exclusion=legacy,
                stats=file_stats,
            )
            texts = [str(candidate["text"]) for candidate in pending]
            scores = (
                quality_proxy.score(texts)
                if quality_proxy.enabled and texts
                else [None] * len(texts)
            )
            encoded = (
                tokenizer(
                    texts,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_length=True,
                )
                if texts
                else {"length": []}
            )
            lengths = encoded.get("length")
            if lengths is None:
                lengths = [len(values) for values in encoded["input_ids"]]
            for record in sampled:
                if record.get("score") is not None:
                    upstream_scores.append(float(record["score"]))
                raw_sources[str(record.get("source") or "unknown")] += 1
            for candidate, local_score, token_length in zip(
                pending, scores, lengths, strict=True
            ):
                text = str(candidate["text"])
                cjk_chars = len(_CJK_RE.findall(text))
                simplified = t2s.convert(text)
                traditional = s2t.convert(text)
                t2s_changes = _changed_positions(text, simplified)
                s2t_changes = _changed_positions(text, traditional)
                stats["cjk_characters"] += cjk_chars
                stats["t2s_changed_positions"] += t2s_changes
                stats["s2t_changed_positions"] += s2t_changes
                stats["t2s_changed_documents"] += int(t2s_changes > 0)
                stats["s2t_changed_documents"] += int(s2t_changes > 0)
                document_hash = str(candidate["document_sha256"])
                if document_hash in exact_hashes:
                    stats["sample_exact_duplicates"] += 1
                else:
                    exact_hashes.add(document_hash)
                    if _sample_near_duplicate(
                        tuple(candidate["near_signature_values"]), near_bands
                    ):
                        stats["sample_near_duplicates"] += 1
                if local_score is not None:
                    value = float(local_score)
                    local_scores.append(value)
                    quality_accepted = value >= quality_proxy.minimum_score
                    stats["local_quality_proxy_accept"] += int(quality_accepted)
                    stats["local_quality_proxy_reject"] += int(not quality_accepted)
                else:
                    quality_accepted = True
                taxonomy = infer_user_taxonomy_label(text)
                tokens = int(token_length) + 1
                taxonomy_docs[taxonomy] += 1
                taxonomy_tokens[taxonomy] += tokens
                if quality_accepted:
                    accepted_taxonomy_docs[taxonomy] += 1
                    accepted_taxonomy_tokens[taxonomy] += tokens
                stats["clean_documents_profiled"] += 1
                stats["profiled_tokens_including_eos"] += tokens
                stats["profiled_characters"] += len(text)
            stats.update(file_stats)
            sample_files_report.append(
                {
                    "inventory_index": file_index,
                    "path": str(row["path"]),
                    "sha256": str(row["sha256"]),
                    "rows_total": int(table.num_rows),
                    "rows_sampled": len(sampled),
                    "rows_clean": len(pending),
                }
            )
            print(
                f"[CHINESE-PROFILE] files={ordinal}/{len(file_indexes)} "
                f"documents={stats['clean_documents_profiled']:,}",
                flush=True,
            )
    finally:
        legacy.close()
    clean_docs = int(stats["clean_documents_profiled"])
    cjk_chars = int(stats["cjk_characters"])
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "inventory_path": str(inventory_file),
        "inventory_sha256": hashlib.sha256(inventory_file.read_bytes()).hexdigest(),
        "source": str(source_name),
        "repo_id": str(source.get("repo_id") or ""),
        "revision": str(source.get("revision") or ""),
        "document_time_coverage": str(
            source.get("document_time_coverage") or "unknown_not_declared"
        ),
        "raw_root": str(Path(raw_root).expanduser().resolve()),
        "admission_policy": str(policy_file),
        "admission_policy_sha256": hashlib.sha256(policy_file.read_bytes()).hexdigest(),
        "tokenizer_path": str(Path(tokenizer_path).expanduser().resolve()),
        "tokenizer_bundle_sha1": actual_tokenizer_sha1,
        "sampling": {
            "method": sampling_method,
            "inventory_files": len(files),
            "sampled_files": len(file_indexes),
            "rows_per_file": int(rows_per_file),
            "files": sample_files_report,
        },
        "stats": dict(sorted(stats.items())),
        "upstream_score_quantiles": _quantiles(upstream_scores),
        "local_quality_proxy_score_quantiles": _quantiles(local_scores),
        "local_quality_proxy_threshold": (
            quality_proxy.minimum_score if quality_proxy.enabled else None
        ),
        "raw_source_documents": dict(sorted(raw_sources.items())),
        "script_profile": {
            "cjk_characters": cjk_chars,
            "t2s_changed_positions": int(stats["t2s_changed_positions"]),
            "t2s_changed_positions_per_cjk": (
                float(stats["t2s_changed_positions"]) / max(cjk_chars, 1)
            ),
            "t2s_changed_document_fraction": (
                float(stats["t2s_changed_documents"]) / max(clean_docs, 1)
            ),
            "s2t_changed_positions": int(stats["s2t_changed_positions"]),
            "s2t_changed_positions_per_cjk": (
                float(stats["s2t_changed_positions"]) / max(cjk_chars, 1)
            ),
        },
        "taxonomy_documents": dict(sorted(taxonomy_docs.items())),
        "taxonomy_tokens": dict(sorted(taxonomy_tokens.items())),
        "quality_accepted_taxonomy_documents": dict(
            sorted(accepted_taxonomy_docs.items())
        ),
        "quality_accepted_taxonomy_tokens": dict(
            sorted(accepted_taxonomy_tokens.items())
        ),
    }
    write_json_atomic(
        Path(output_path).expanduser().resolve(),
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile a pinned modern-Chinese candidate with distributed samples."
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--admission-policy",
        default="configs/data/pretrain_data_admission_policy.json",
    )
    parser.add_argument("--tokenizer", default="ml/modeling/text")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-files", type=int, default=64)
    parser.add_argument("--file-index", action="append", type=int, default=[])
    parser.add_argument("--rows-per-file", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = profile_candidate(
        inventory_path=str(args.inventory),
        raw_root=str(args.raw_root),
        source_name=str(args.source),
        admission_policy=str(args.admission_policy),
        tokenizer_path=str(args.tokenizer),
        output_path=str(args.output),
        sample_files=int(args.sample_files),
        rows_per_file=int(args.rows_per_file),
        selected_file_indexes=[int(value) for value in args.file_index] or None,
    )
    print(
        f"[DONE] documents={report['stats']['clean_documents_profiled']:,} "
        f"output={Path(args.output).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
