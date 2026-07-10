from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml.errors import SophiaUsageError
from ml.modeling.text.conversation import (
    render_conversation_segments,
)
from ml.integrations.adapters.hf.tokenizer import load_local_tokenizer
from ml.training.posttrain.data import ChatExample, load_supervised_examples

SPLITS: tuple[str, ...] = ("train", "val", "test")
DATASETS: tuple[str, ...] = ("sft",)
_BINS: tuple[tuple[str, int], ...] = (
    ("<=512", 512),
    ("513-1k", 1024),
    ("1k-2k", 2048),
    ("2k-4k", 4096),
    (">4k", 10**18),
)
_THRESHOLDS: tuple[int, ...] = (4096,)


@dataclass(frozen=True)
class LengthRow:
    length: int
    family: str
    lang: str
    bucket: str
    split: str
    row_index: int
    source_bucket: str
    hash_hint: str

def _bucket(length: int) -> str:
    for label, upper in _BINS:
        if int(length) <= int(upper):
            return label
    return ">4k"


def _percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * float(q)))))
    return int(sorted_values[idx])

def _rendered_len(
    *,
    tokenizer: Any,
    example: ChatExample,
    add_generation_prompt: bool,
    mode: str,
) -> int:
    segments = render_conversation_segments(
        list(example.messages),
        add_generation_prompt=bool(add_generation_prompt),
    )
    if str(mode) == "chars":
        return int(sum(len(str(segment.text)) for segment in segments))
    total = 0
    for segment in segments:
        token_ids = tokenizer.encode(str(segment.text), add_special_tokens=False)
        total += int(len(token_ids))
    return int(total)


def _profile_examples(
    *,
    tokenizer: Any,
    examples: list[ChatExample],
    add_generation_prompt: bool,
    mode: str,
    split: str,
    sample_rows_per_split: int,
    top_examples: int,
) -> dict[str, Any]:
    lengths: list[int] = []
    buckets: Counter[str] = Counter()
    capability_buckets: dict[str, Counter[str]] = {}
    language_buckets: dict[str, Counter[str]] = {}
    longest: list[LengthRow] = []
    started = time.time()
    limit = int(sample_rows_per_split)
    if limit > 0:
        examples = examples[:limit]
    for row_index, example in enumerate(examples):
        length = _rendered_len(
            tokenizer=tokenizer,
            example=example,
            add_generation_prompt=bool(add_generation_prompt),
            mode=str(mode),
        )
        lengths.append(int(length))
        bucket = _bucket(int(length))
        buckets[bucket] += 1

        metadata = dict(example.metadata or {})
        family = str(metadata.get("capability_family") or metadata.get("family") or "unknown")
        lang = str(metadata.get("lang") or "unknown")
        source_bucket = str(metadata.get("bucket") or metadata.get("source_bucket") or "unknown")
        hash_hint = str(
            metadata.get("prompt_hash")
            or metadata.get("conversation_hash")
            or metadata.get("row_hash")
            or ""
        )
        capability_buckets.setdefault(family, Counter())[bucket] += 1
        language_buckets.setdefault(lang, Counter())[bucket] += 1
        if int(top_examples) > 0:
            longest.append(
                LengthRow(
                    length=int(length),
                    family=family,
                    lang=lang,
                    bucket=bucket,
                    split=str(split),
                    row_index=int(row_index),
                    source_bucket=source_bucket,
                    hash_hint=hash_hint[:32],
                )
            )

    lengths.sort()
    longest.sort(key=lambda row: row.length, reverse=True)
    longest = longest[: max(int(top_examples), 0)]
    total = len(lengths)
    return {
        "rows": int(total),
        "min": int(lengths[0]) if lengths else 0,
        "max": int(lengths[-1]) if lengths else 0,
        "avg": round(sum(lengths) / float(max(total, 1)), 2),
        "p50": _percentile(lengths, 0.50),
        "p90": _percentile(lengths, 0.90),
        "p95": _percentile(lengths, 0.95),
        "p99": _percentile(lengths, 0.99),
        "buckets": {
            label: {
                "rows": int(buckets[label]),
                "pct": round((float(int(buckets[label])) * 100.0) / float(max(int(total), 1)), 4),
            }
            for label, _upper in _BINS
        },
        "thresholds": {
            f">={threshold}": {
                "rows": int(sum(1 for value in lengths if int(value) >= int(threshold))),
                "pct": round(
                    (
                        float(sum(1 for value in lengths if int(value) >= int(threshold)))
                        * 100.0
                    )
                    / float(max(int(total), 1)),
                    4,
                ),
            }
            for threshold in _THRESHOLDS
        },
        "capability_buckets": {
            family: {
                label: {
                    "rows": int(counter[label]),
                    "pct": round(
                        (float(int(counter[label])) * 100.0)
                        / float(max(sum(counter.values()), 1)),
                        4,
                    ),
                }
                for label, _upper in _BINS
            }
            for family, counter in sorted(capability_buckets.items())
        },
        "language_buckets": {
            lang: {
                label: {
                    "rows": int(counter[label]),
                    "pct": round(
                        (float(int(counter[label])) * 100.0)
                        / float(max(sum(counter.values()), 1)),
                        4,
                    ),
                }
                for label, _upper in _BINS
            }
            for lang, counter in sorted(language_buckets.items())
        },
        "longest_examples": [asdict(row) for row in longest],
        "seconds": round(time.time() - started, 2),
    }


def build_posttrain_length_profile(
    *,
    dataset_root: Path,
    tokenizer_path: Path,
    max_model_length: int,
    mode: str,
    requested_datasets: tuple[str, ...],
    requested_splits: tuple[str, ...],
    sample_rows_per_split: int,
    top_examples: int,
) -> dict[str, Any]:
    tokenizer = None
    if str(mode) == "tokens":
        tokenizer = load_local_tokenizer(
            str(tokenizer_path),
            model_max_length=int(max_model_length),
        )
    bad_datasets = sorted(set(requested_datasets) - set(DATASETS))
    bad_splits = sorted(set(requested_splits) - set(SPLITS))
    if bad_datasets:
        raise SophiaUsageError(f"[ERR] unsupported datasets: {bad_datasets}")
    if bad_splits:
        raise SophiaUsageError(f"[ERR] unsupported splits: {bad_splits}")

    report: dict[str, Any] = {
        "kind": "posttrain_length_profile",
        "dataset_root": str(dataset_root),
        "tokenizer_path": str(tokenizer_path),
        "max_model_length": int(max_model_length),
        "mode": str(mode),
        "sample_rows_per_split": int(sample_rows_per_split),
        "render_style": "posttrain_chat",
        "datasets": {},
    }
    for dataset_name in requested_datasets:
        report["datasets"][dataset_name] = {}
        for split in requested_splits:
            path = dataset_root / dataset_name / f"{split}.jsonl"
            examples = load_supervised_examples(str(path))
            report["datasets"][dataset_name][split] = _profile_examples(
                tokenizer=tokenizer,
                examples=examples,
                add_generation_prompt=False,
                mode=str(mode),
                split=str(split),
                sample_rows_per_split=int(sample_rows_per_split),
                top_examples=int(top_examples),
            )
    return report


def write_posttrain_length_profile(
    *,
    output_path: Path,
    report: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
