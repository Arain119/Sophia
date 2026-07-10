#!/usr/bin/env python
"""
Build a balanced pretraining token-shards dataset directly from parquet corpora.

Why:
  - Avoids writing massive intermediate JSONL.
  - Enforces token-level quotas (tokens including EOS insertions), matching
    `ml.training.pretrain.shard_builder` output semantics.

This script is designed around a **quota plan**:
  - Each bucket has a target token budget.
  - We stream parquet rows, bucket them, and only tokenize+write rows that
    contribute to buckets with remaining budget.

By default this uses the repo's preset backed by the local parquet pool under
`dataset/pretrain_total/train/` (books/wiki/news/discourse, code/math, translation).

Example:
  python -m ml.tooling.cli pretrain build-mix-shards ^
    --preset pretrain ^
    --tokenizer_path ml/modeling/text ^
    --out_dir dataset/pretrain_total/pretrain ^
    --out_dtype int32 ^
    --shard_size_tokens 20000000 ^
    --max_doc_tokens 4096 ^
    --min_chunk_tokens 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast

from ml.errors import SophiaUsageError
from ml.data.token_shards.shard_manifest import TokenShard, TokenShardManifest, save_manifest
from ml.data.token_shards.tokenizer_fingerprint import compute_tokenizer_bundle_sha1
from ml.tooling.core.parquet_inventory import list_glob_prefer_plain
from ml.data.pretrain_filters import (
    has_repeated_sentences,
    is_poison_repetitive_text,
    normalize_text,
)
from ml.tooling.core.pretrain_mix_presets import (
    PretrainMixPreset,
    SourcePreset,
    get_preset,
)
from ml.tooling.core.pretrain_taxonomy_user import (
    LABELS,
    infer_user_taxonomy_label,
)
from ml.tooling.core.text_hygiene import (
    has_human_bot_tags,
    has_placeholders,
    mentions_ai,
)

# Reuse the exact encoding + sharding logic from `ml.training.pretrain.shard_builder`
# to guarantee tokenization parity with training.
from ml.training.pretrain.shard_builder import (  # noqa: E402
    ShardWriter,
    _encode_batch_flat,
    _iter_text_chunks,
    _load_tokenizer,
    _manifest_dtype_name,
    _resolve_out_dtype,
    _token_id_upper_bound,
)


def _safe_report_tag(tag: str) -> str:
    t = str(tag or "").strip()
    if not t:
        return ""
    # Avoid path traversal / weird filesystem chars.
    t = re.sub(r"[^A-Za-z0-9_.-]+", "_", t)
    t = t.strip("._-")
    return t[:64]


def _merge_counts(*dicts: object) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            key = str(k)
            try:
                n = int(v or 0)
            except Exception:
                continue
            if n <= 0:
                continue
            out[key] = int(out.get(key, 0)) + int(n)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _load_json(path: str) -> dict | None:
    try:
        p = str(path)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_json(path: str, obj: object) -> None:
    # Atomic write: resume support relies on checkpoint files not being torn.
    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(p))


def _audit_merge_primary_backfill(*, primary: dict, backfill: dict) -> dict:
    kept_tokens = _merge_counts(
        primary.get("kept_tokens_eos"), backfill.get("kept_tokens_eos")
    )
    kept_docs = _merge_counts(primary.get("kept_docs"), backfill.get("kept_docs"))

    # Prefer the original plan from primary (backfill mode replaces quotas with a single bucket).
    targets = primary.get("targets_tokens_eos")
    if not isinstance(targets, dict) or not targets:
        targets = backfill.get("targets_tokens_eos")
    targets = targets if isinstance(targets, dict) else {}
    targets_tokens = {
        str(k): int(v or 0) for k, v in targets.items() if int(v or 0) > 0
    }

    manifest = (
        backfill.get("manifest") if isinstance(backfill.get("manifest"), dict) else {}
    )
    total_tokens = int(manifest.get("total_tokens") or 0)
    if total_tokens <= 0:
        total_tokens = int(sum(int(v) for v in kept_tokens.values()))

    kept_pct = {
        k: (float(v) / float(max(total_tokens, 1))) for k, v in kept_tokens.items()
    }

    return {
        "preset": primary.get("preset") or backfill.get("preset"),
        "tokenizer_path": backfill.get("tokenizer_path")
        or primary.get("tokenizer_path"),
        "out_dir": backfill.get("out_dir") or primary.get("out_dir"),
        "docs_in": int(primary.get("docs_in") or 0) + int(backfill.get("docs_in") or 0),
        "docs_kept": int(primary.get("docs_kept") or 0)
        + int(backfill.get("docs_kept") or 0),
        "tokens_total": int(total_tokens),
        "targets_tokens_eos": dict(
            sorted(targets_tokens.items(), key=lambda kv: kv[0])
        ),
        "kept_tokens_eos": kept_tokens,
        "kept_tokens_pct": dict(sorted(kept_pct.items(), key=lambda kv: kv[0])),
        "kept_docs": kept_docs,
        "manifest": manifest,
        "audit": {
            "merged_from": [
                "mix_report.primary.json",
                "mix_report.json",
            ],
            "note": "Final kept_tokens_eos/kept_docs are computed by summing primary + backfill reports.",
        },
    }


def _list_glob(pattern: str) -> list[str]:
    return list_glob_prefer_plain(str(pattern))


def _safe_float(x: object) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _looks_like_mojibake(text: str, *, max_frac: float = 0.02) -> bool:
    """
    Drop samples with a high fraction of the Unicode replacement char (�).

    This catches common encoding-corrupted text which can be prevalent in some corpora.
    """

    s = str(text or "")
    if not s:
        return False
    bad = s.count("\ufffd")
    if bad <= 0:
        return False
    return (bad / float(max(len(s), 1))) >= float(max_frac)


def _max_line_len(text: str) -> int:
    max_len = 0
    cur = 0
    for ch in text:
        if ch == "\n" or ch == "\r":
            if cur > max_len:
                max_len = cur
            cur = 0
            continue
        cur += 1
    return max(max_len, cur)


def _looks_like_generated_code_blob(text: str) -> str:
    """
    Conservatively reject code rows that are mostly generated payload data.

    The goal is not to remove long-tail code. We only catch high-confidence
    cases observed in the pool audit: compiled QuickJS arrays, minified JS/WASM
    bundles, embedded base64 assets/archives, and huge numeric data literals.
    """

    s = str(text or "")
    if not s:
        return ""

    if "File generated automatically by the QuickJS compiler" in s:
        return "code_generated_quickjs_blob"

    n_chars = len(s)
    if n_chars < 20_000:
        return ""

    max_line = _max_line_len(s)
    head = s[:200_000]

    if "base64.b64decode(" in head and (max_line >= 8_000 or n_chars >= 100_000):
        return "code_embedded_base64_blob"
    if "sourceMappingURL=data:" in head:
        return "code_embedded_sourcemap_blob"
    if "data:font/" in head and "base64," in head and max_line >= 20_000:
        return "code_embedded_base64_asset_blob"

    if max_line >= 50_000:
        stripped_head = head.lstrip()
        if (
            "webpackJsonp" in head
            or "__webpack_require__" in head
            or "Minified React error" in head
            or stripped_head.startswith(("var react_", "(window.webpackJsonp"))
        ):
            return "code_minified_js_bundle"
        if "_scriptDir" in head and ("WebAssembly" in head or "wasm" in head):
            return "code_emscripten_wasm_bundle"
        if "tagInfoJSON" in head or "TRANSACTION_HISTORY_EXTENSION" in head:
            return "code_huge_data_literal"

    if n_chars >= 100_000 and (
        "raw_data[]" in head
        or "ModelData[]" in head
        or "static const unsigned short raw_data" in head
    ):
        if s.count("0x") > 1_000 or s.count(",") > 5_000:
            return "code_numeric_data_blob"

    return ""


@dataclass
class _CleanCfg:
    min_chars: int = 200
    drop_human_bot_tags: bool = True
    drop_ai_mentions: bool = True
    drop_placeholders: bool = True
    drop_poison: bool = True
    drop_repeated_sentences: bool = True
    drop_mojibake: bool = True
    drop_code_blobs: bool = True
    mojibake_max_frac: float = 0.02


@dataclass
class _Stats:
    docs_in: int = 0
    docs_kept: int = 0
    tokens_kept: int = 0
    by_bucket_docs: dict[str, int] | None = None
    by_bucket_tokens: dict[str, int] | None = None
    dropped: dict[str, int] | None = None

    def add_drop(self, reason: str) -> None:
        if self.dropped is None:
            self.dropped = {}
        self.dropped[reason] = int(self.dropped.get(reason, 0)) + 1

    def add_kept(self, *, bucket: str, docs: int, tokens: int) -> None:
        if self.by_bucket_docs is None:
            self.by_bucket_docs = {}
        if self.by_bucket_tokens is None:
            self.by_bucket_tokens = {}
        self.by_bucket_docs[bucket] = int(self.by_bucket_docs.get(bucket, 0)) + int(
            docs
        )
        self.by_bucket_tokens[bucket] = int(self.by_bucket_tokens.get(bucket, 0)) + int(
            tokens
        )
        self.docs_kept += int(docs)
        self.tokens_kept += int(tokens)


def _infer_bucket_for_source(
    *,
    src: SourcePreset,
    text: str,
    bucket_value: str | None,
) -> str:
    if src.bucket_mode == "column":
        b = str(bucket_value or "").strip() or "UNKNOWN"
        return f"{src.name}:{b}"

    # user_taxonomy
    lab = infer_user_taxonomy_label(text)
    if lab not in LABELS:
        lab = "other"
    return f"{src.name}:{lab}"


def _maybe_clean_text(
    *,
    text: str,
    clean: _CleanCfg,
    src_name: str | None = None,
) -> tuple[str, str]:
    """
    Returns: (clean_text, drop_reason). drop_reason=="" means keep.
    """

    raw = str(text or "")
    src_key = str(src_name or "").strip().lower()
    is_code = src_key == "code" or src_key.startswith("code_")

    # Check mojibake on the raw text *before* normalization.
    # Note: `normalize_text()` intentionally removes U+FFFD to salvage mildly-corrupted docs,
    # so mojibake detection must run before that step to be effective.
    if clean.drop_mojibake and _looks_like_mojibake(
        raw, max_frac=float(clean.mojibake_max_frac)
    ):
        return "", "mojibake"
    if is_code and clean.drop_code_blobs:
        code_blob_reason = _looks_like_generated_code_blob(raw)
        if code_blob_reason:
            return "", code_blob_reason

    s = normalize_text(raw)
    if not s:
        return "", "empty"
    if int(clean.min_chars) > 0 and len(s) < int(clean.min_chars):
        return "", "too_short"

    if clean.drop_human_bot_tags and has_human_bot_tags(s):
        return "", "human_bot_tags"
    if clean.drop_ai_mentions and mentions_ai(s):
        return "", "ai_mention"
    # Code corpora frequently contain "TODO" and other benign placeholder-like tokens.
    # Sentence-based cleaning also tends to false-positive on code (e.g., semicolon-heavy lines),
    # which can starve the code quota. Keep code cleaning conservative here.
    if (not is_code) and clean.drop_placeholders and has_placeholders(s):
        return "", "placeholder"

    if clean.drop_poison and is_poison_repetitive_text(s):
        return "", "poison_repetitive"
    if (not is_code) and clean.drop_repeated_sentences and has_repeated_sentences(s):
        return "", "repeated_sentences"

    return s, ""


def _plan_string(*, quotas: dict[str, int]) -> str:
    lines: list[str] = []
    total = int(sum(int(v) for v in quotas.values()))
    lines.append(f"[PLAN] target tokens_eos: {total:,}")
    for k, v in sorted(quotas.items(), key=lambda kv: int(kv[1]), reverse=True):
        pct = (float(v) / float(max(total, 1))) * 100.0
        lines.append(f"  - {k:40s} {int(v):,} ({pct:5.2f}%)")
    return "\n".join(lines)


def _all_quotas_met(*, kept: dict[str, int], quotas: dict[str, int]) -> bool:
    for k, q in quotas.items():
        if int(q) <= 0:
            continue
        if int(kept.get(k, 0)) < int(q):
            return False
    return True


def _read_parquet_rows(
    *,
    fp: str,
    cols: list[str],
) -> _ParquetBatch | None:
    # Read the single file directly. ``pq.read_table`` routes through the dataset
    # API, whose hive-partition inference turns path segments like ``source=Python``
    # into a partition field that collides with the file's own ``source`` column.
    try:
        return _ParquetBatch.from_table(pq.ParquetFile(fp).read(columns=cols))
    except Exception:
        return None


@dataclass(frozen=True)
class _ParquetColumn:
    values: list[object]

    def to_list(self) -> list[object]:
        return list(self.values)


@dataclass(frozen=True)
class _ParquetBatch:
    table: pa.Table

    @classmethod
    def from_table(cls, table: pa.Table) -> _ParquetBatch:
        return cls(table=table)

    @property
    def height(self) -> int:
        return int(self.table.num_rows)

    @property
    def columns(self) -> list[str]:
        return [str(name) for name in self.table.column_names]

    def get_column(self, name: str) -> _ParquetColumn:
        return _ParquetColumn(values=self.table[str(name)].to_pylist())


def build_mix_token_shards(
    *,
    preset: PretrainMixPreset,
    tokenizer_path: str,
    out_dir: str,
    shard_size_tokens: int,
    out_dtype: str,
    batch_texts: int,
    chunk_rows: int,
    max_doc_chars: int,
    min_chunk_chars: int,
    max_doc_tokens: int,
    min_chunk_tokens: int,
    seed: int,
    max_passes: int,
    progress_every_docs: int,
    clean: _CleanCfg,
    total_tokens_eos_override: int = 0,
    source_allowlist: tuple[str, ...] = (),
    source_allowlist_fill_budget: bool = True,
    resume: bool = False,
    resume_backfill_bucket: str = "",
    report_tag: str = "",
) -> None:
    ctx = _BuildCtx(
        preset=preset,
        tokenizer_path=str(tokenizer_path),
        out_dir=str(out_dir),
        shard_size_tokens=int(shard_size_tokens),
        out_dtype=str(out_dtype),
        batch_texts=int(batch_texts),
        chunk_rows=int(chunk_rows),
        max_doc_chars=int(max_doc_chars),
        min_chunk_chars=int(min_chunk_chars),
        max_doc_tokens=int(max_doc_tokens),
        min_chunk_tokens=int(min_chunk_tokens),
        seed=int(seed),
        max_passes=int(max_passes),
        progress_every_docs=int(progress_every_docs),
        clean=clean,
        total_tokens_eos_override=int(total_tokens_eos_override),
        source_allowlist=tuple(
            str(s).strip() for s in (source_allowlist or ()) if str(s).strip()
        ),
        source_allowlist_fill_budget=bool(source_allowlist_fill_budget),
        resume=bool(resume),
        resume_backfill_bucket=str(resume_backfill_bucket or "").strip(),
        report_tag=_safe_report_tag(str(report_tag or "")),
    )
    _build_mix_token_shards(ctx)


@dataclass
class _BuildCtx:
    preset: PretrainMixPreset
    tokenizer_path: str
    out_dir: str
    shard_size_tokens: int
    out_dtype: str
    batch_texts: int
    chunk_rows: int
    max_doc_chars: int
    min_chunk_chars: int
    max_doc_tokens: int
    min_chunk_tokens: int
    seed: int
    max_passes: int
    progress_every_docs: int
    clean: _CleanCfg
    total_tokens_eos_override: int = 0
    resume: bool = False
    resume_backfill_bucket: str = ""
    report_tag: str = ""
    source_allowlist: tuple[str, ...] = ()
    source_allowlist_fill_budget: bool = True

    tok: PreTrainedTokenizerFast | None = None
    eos_id: int = 0
    dtype: np.dtype | None = None
    writer: ShardWriter | None = None
    quotas: dict[str, int] | None = None
    rng: random.Random | None = None

    kept_tokens: dict[str, int] | None = None
    kept_docs: dict[str, int] | None = None
    buffers: dict[str, list[str]] | None = None
    overall: _Stats | None = None
    resume_existing_tokens: int = 0
    resume_existing_shards: int = 0
    resume_deleted_tmp: int = 0


def _init_build_ctx(ctx: _BuildCtx) -> None:
    ctx.out_dir = os.path.abspath(str(ctx.out_dir))
    os.makedirs(ctx.out_dir, exist_ok=True)

    ctx.tok = _load_tokenizer(ctx.tokenizer_path)
    vocab_size = int(_token_id_upper_bound(ctx.tok) or 0)
    eos_id = getattr(ctx.tok, "eos_token_id", None)
    if not isinstance(eos_id, int):
        raise SophiaUsageError("[ERR] tokenizer has no eos_token_id")
    ctx.eos_id = int(eos_id)
    vocab_size = max(int(vocab_size), int(ctx.eos_id) + 1)

    dtype = _resolve_out_dtype(out_dtype=str(ctx.out_dtype), vocab_size=vocab_size)
    ctx.dtype = np.dtype(dtype)
    ctx.writer = ShardWriter(
        ctx.out_dir,
        shard_size_tokens=max(int(ctx.shard_size_tokens), 1024),
        dtype=np.dtype(dtype),
    )

    ctx.quotas = ctx.preset.resolve_quotas(
        total_tokens_eos=int(ctx.total_tokens_eos_override)
        if int(ctx.total_tokens_eos_override) > 0
        else None
    )
    if ctx.source_allowlist:
        allow = {str(s).strip() for s in ctx.source_allowlist if str(s).strip()}
        available = {str(s.name) for s in ctx.preset.sources}
        unknown = sorted(allow - available)
        if unknown:
            raise SophiaUsageError(
                "[ERR] unknown --source_allowlist entries:\n"
                + "\n".join(f"- {x}" for x in unknown)
                + "\nAvailable sources:\n"
                + "\n".join(f"- {x}" for x in sorted(available))
            )

        if ctx.quotas is None:
            raise RuntimeError("ctx not initialized")
        target_total = int(sum(int(v) for v in ctx.quotas.values()))
        filtered = {
            str(k): int(v)
            for k, v in ctx.quotas.items()
            if str(k).split(":", 1)[0] in allow and int(v) > 0
        }
        if not filtered:
            raise SophiaUsageError(
                "[ERR] --source_allowlist produced an empty quota plan"
            )

        if bool(ctx.source_allowlist_fill_budget):
            ctx.quotas = _allocate_proportional(total=int(target_total), weights=filtered)
        else:
            ctx.quotas = filtered
    ctx.rng = random.Random(int(ctx.seed))
    ctx.kept_tokens = {}
    ctx.kept_docs = {}
    ctx.overall = _Stats()
    ctx.buffers = {k: [] for k, v in ctx.quotas.items() if int(v) > 0}

    if bool(ctx.resume):
        _resume_existing_shards_and_adjust_quotas(ctx)


def _allocate_proportional(*, total: int, weights: dict[str, int]) -> dict[str, int]:
    total = int(total)
    if total <= 0:
        return {k: 0 for k in weights}
    items = [(str(k), int(v)) for k, v in weights.items() if int(v) > 0]
    if not items:
        return {k: 0 for k in weights}
    wsum = int(sum(v for _, v in items))
    if wsum <= 0:
        return {k: 0 for k in weights}
    if total == wsum:
        return {k: int(v) for k, v in items}

    base: dict[str, int] = {}
    fracs: list[tuple[float, int, str]] = []
    for k, w in items:
        exact = (float(total) * float(w)) / float(wsum)
        b = int(math.floor(exact))
        base[k] = b
        fracs.append((float(exact - float(b)), int(w), str(k)))
    used = int(sum(base.values()))
    rem = int(total - used)
    if rem > 0:
        fracs.sort(key=lambda t: (float(t[0]), int(t[1]), str(t[2])), reverse=True)
        for i in range(rem):
            k = str(fracs[i % len(fracs)][2])
            base[k] = int(base.get(k, 0)) + 1
    # Fill any missing keys with 0.
    for k in list(weights.keys()):
        if str(k) not in base:
            base[str(k)] = 0
    return base


def _resume_existing_shards_and_adjust_quotas(ctx: _BuildCtx) -> None:  # noqa: C901
    if ctx.writer is None or ctx.dtype is None or ctx.quotas is None:
        raise RuntimeError("ctx not initialized")

    out_dir = str(ctx.out_dir)
    dtype = np.dtype(ctx.dtype)
    itemsize = int(dtype.itemsize)
    if itemsize <= 0:
        raise RuntimeError("invalid dtype itemsize")

    # Clean up abandoned temp files from a previous interrupted run.
    deleted_tmp = 0
    tmp_re = re.compile(r"^shard_\d{5}\.bin\.tmp\.")
    try:
        for name in os.listdir(out_dir):
            if tmp_re.match(str(name)):
                try:
                    os.remove(os.path.join(out_dir, str(name)))
                    deleted_tmp += 1
                except OSError:
                    continue
    except FileNotFoundError:
        pass

    # Discover existing finalized shards.
    shard_re = re.compile(r"^shard_(\d{5})\.bin$")
    found: list[tuple[int, str, int]] = []
    try:
        for name in os.listdir(out_dir):
            m = shard_re.match(str(name))
            if not m:
                continue
            idx = int(m.group(1))
            path = os.path.join(out_dir, str(name))
            try:
                size = int(os.path.getsize(path))
            except OSError:
                continue
            if size <= 0:
                continue
            if size % itemsize != 0:
                raise SophiaUsageError(
                    f"[ERR] shard size not divisible by dtype ({dtype}) itemsize={itemsize}: {path} bytes={size}"
                )
            tokens = int(size // itemsize)
            found.append((idx, str(name), tokens))
    except FileNotFoundError:
        found = []

    if not found:
        # Nothing to resume.
        ctx.resume_existing_tokens = 0
        ctx.resume_existing_shards = 0
        ctx.resume_deleted_tmp = int(deleted_tmp)
        if deleted_tmp:
            print(
                f"[RESUME] deleted {int(deleted_tmp)} abandoned tmp shard files",
                flush=True,
            )
        return

    found.sort(key=lambda t: int(t[0]))
    existing_shards = [
        TokenShard(path=str(name), tokens=int(tok)) for _, name, tok in found
    ]
    existing_tokens = int(sum(int(tok) for _, _, tok in found))
    next_idx = int(max(i for i, _, _ in found)) + 1

    # Extend manifest shards and advance shard index to avoid overwriting.
    ctx.writer.shards.extend(existing_shards)
    ctx.writer._shard_idx = int(next_idx)  # noqa: SLF001 - intentional resume hook

    ctx.resume_existing_tokens = int(existing_tokens)
    ctx.resume_existing_shards = int(len(existing_shards))
    ctx.resume_deleted_tmp = int(deleted_tmp)

    print(
        f"[RESUME] found {len(existing_shards)} existing shards ({existing_tokens:,} tokens); "
        f"next_shard_idx={next_idx}",
        flush=True,
    )
    if deleted_tmp:
        print(
            f"[RESUME] deleted {int(deleted_tmp)} abandoned tmp shard files", flush=True
        )

    if str(ctx.resume_backfill_bucket or "").strip():
        backfill = str(ctx.resume_backfill_bucket).strip()
        if ctx.quotas is None:
            raise RuntimeError("ctx not initialized")
        target_total = int(sum(int(v) for v in ctx.quotas.values()))
        remaining = int(target_total - existing_tokens)
        if remaining <= 0:
            raise SophiaUsageError(
                f"[ERR] resume backfill: no remaining tokens to allocate (existing={existing_tokens:,} target={target_total:,})"
            )
        ctx.quotas = {backfill: int(remaining)}
        ctx.buffers = {backfill: []}
        print(
            f"[RESUME] backfill mode: remaining {remaining:,} tokens_eos -> {backfill}",
            flush=True,
        )
        return

    # Prefer resuming from a previous build's report when available. This preserves the intended
    # bucket-level remaining quotas even after multiple resume runs.
    report_path = os.path.join(out_dir, "mix_report.json")
    try:
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as f:
                rep = json.load(f)
        else:
            rep = None
    except Exception:
        rep = None

    if isinstance(rep, dict):
        # Resume from report if possible, but compute remaining quotas against *current* targets
        # (allows changing --total_tokens_eos while still using per-bucket kept counts).
        kept = rep.get("kept_tokens_eos")
        if isinstance(kept, dict) and kept and ctx.quotas is not None:
            # Seed per-bucket counters so progress + final report represent the full build.
            # This is critical for long-running builds that are resumed multiple times.
            try:
                ctx.kept_tokens = {str(k): int(v or 0) for k, v in kept.items()}
            except Exception:
                ctx.kept_tokens = {}
            kept_docs_rep = rep.get("kept_docs")
            if isinstance(kept_docs_rep, dict) and kept_docs_rep:
                try:
                    ctx.kept_docs = {
                        str(k): int(v or 0) for k, v in kept_docs_rep.items()
                    }
                except Exception:
                    ctx.kept_docs = {}
            target_quotas = {
                str(k): int(v) for k, v in ctx.quotas.items() if int(v) > 0
            }
            target_total = int(sum(int(v) for v in target_quotas.values()))
            if target_total <= 0:
                raise SophiaUsageError("[ERR] invalid quota plan (total<=0)")
            if existing_tokens >= target_total:
                raise SophiaUsageError(
                    f"[ERR] resume shards already cover target budget: existing={existing_tokens:,} target={target_total:,}. "
                    "Use a larger --total_tokens_eos or a new --out_dir."
                )

            remaining: dict[str, int] = {}
            kept_total = 0
            for k, q in target_quotas.items():
                kv = int(kept.get(k, 0) or 0)
                kept_total += max(0, kv)
                remaining[k] = max(0, int(q) - max(0, kv))

            desired_remaining = int(target_total - existing_tokens)
            rem_sum = int(sum(int(v) for v in remaining.values()))
            drift = int(desired_remaining - rem_sum)
            if drift != 0 and remaining:
                if drift > 0:
                    # Add any missing budget to the largest remaining bucket.
                    max_key = max(remaining.items(), key=lambda kv: int(kv[1]))[0]
                    remaining[max_key] = max(
                        0, int(remaining.get(max_key, 0)) + int(drift)
                    )
                else:
                    # Remove extra budget from largest remaining buckets.
                    to_remove = int(-drift)
                    for k, _ in sorted(
                        remaining.items(), key=lambda kv: int(kv[1]), reverse=True
                    ):
                        if to_remove <= 0:
                            break
                        cur = int(remaining.get(k, 0))
                        if cur <= 0:
                            continue
                        take = min(int(to_remove), int(cur))
                        remaining[k] = int(cur - take)
                        to_remove -= int(take)
                    if to_remove > 0:
                        print(
                            f"[WARN] resume drift leftover after quota adjustment: {int(to_remove):,} tokens",
                            flush=True,
                        )

            ctx.quotas = remaining
            # Ensure we keep buffers for the remaining buckets, but keep the cumulative
            # kept_* dicts (they include previous runs).
            ctx.buffers = {k: [] for k, v in ctx.quotas.items() if int(v) > 0}
            print(
                f"[RESUME] derived remaining quotas from mix_report.json: {desired_remaining:,} tokens_eos "
                f"(kept_total_report={int(kept_total):,}, existing={existing_tokens:,}, target={target_total:,})",
                flush=True,
            )
            return

    # Adjust quotas to fill up to the *target* token budget, assuming existing shards
    # belong to earlier sources in preset order (the builder processes sources sequentially).
    target_quotas = {str(k): int(v) for k, v in ctx.quotas.items()}
    target_total = int(sum(int(v) for v in target_quotas.values()))
    if target_total <= 0:
        raise SophiaUsageError("[ERR] invalid quota plan (total<=0)")
    if existing_tokens >= target_total:
        raise SophiaUsageError(
            f"[ERR] resume shards already cover target budget: existing={existing_tokens:,} target={target_total:,}. "
            "Use a larger --total_tokens_eos or a new --out_dir."
        )

    remaining_quotas = dict(target_quotas)
    remaining_to_allocate = int(existing_tokens)

    for src in ctx.preset.sources:
        if remaining_to_allocate <= 0:
            break
        prefix = f"{str(src.name)}:"
        src_keys = [k for k in target_quotas.keys() if str(k).startswith(prefix)]
        if not src_keys:
            continue
        src_weights = {
            k: int(target_quotas[k]) for k in src_keys if int(target_quotas[k]) > 0
        }
        src_total = int(sum(src_weights.values()))
        if src_total <= 0:
            continue
        take = min(int(remaining_to_allocate), int(src_total))
        if take <= 0:
            continue
        alloc = _allocate_proportional(total=int(take), weights=src_weights)
        for k, sub in alloc.items():
            sub = int(sub)
            if sub <= 0:
                continue
            remaining_quotas[k] = max(0, int(remaining_quotas.get(k, 0)) - sub)
        remaining_to_allocate -= int(take)

    if remaining_to_allocate > 0:
        # More existing tokens than we can attribute; drop them into the largest remaining bucket drift.
        print(
            f"[WARN] resume could not attribute {remaining_to_allocate:,} existing tokens to any source; "
            "quota adjustment may be slightly off.",
            flush=True,
        )

    desired_remaining = int(target_total - existing_tokens)
    actual_remaining = int(sum(int(v) for v in remaining_quotas.values()))
    drift = int(desired_remaining - actual_remaining)
    if drift != 0 and remaining_quotas:
        max_key = max(remaining_quotas.items(), key=lambda kv: int(kv[1]))[0]
        remaining_quotas[max_key] = max(0, int(remaining_quotas[max_key]) + int(drift))

    ctx.quotas = remaining_quotas
    # Rebuild buffers to match the adjusted quota set.
    ctx.buffers = {k: [] for k, v in ctx.quotas.items() if int(v) > 0}


def _iter_parquet_batches(
    *,
    fp: str,
    cols: list[str],
    chunk_rows: int,
) -> Iterator[_ParquetBatch]:
    """
    Read parquet in one or more batches.

    - For small files, read the full table (fast path).
    - For large files, stream record batches through ``pyarrow.parquet.ParquetFile``
      to avoid high peak memory and extra optional dependencies.
    """

    try:
        fsize = int(os.path.getsize(fp))
    except Exception:
        fsize = 0

    # Heuristic: large parquet files can be >1GB in this pool. Avoid full materialization.
    # 256 MiB is a pragmatic cutoff.
    if fsize > 256 * 1024 * 1024 and int(chunk_rows) > 0:
        try:
            pf = pq.ParquetFile(fp)
        except Exception:
            return
        step = max(int(chunk_rows), 1)
        for rb in pf.iter_batches(batch_size=int(step), columns=cols):
            try:
                batch = _ParquetBatch.from_table(pa.Table.from_batches([rb]))
            except Exception:
                break
            if int(batch.height) > 0:
                yield batch
        return

    df = _read_parquet_rows(fp=fp, cols=cols)
    if df is None or int(df.height) <= 0:
        return
    yield df


def _flush_bucket(ctx: _BuildCtx, bucket: str) -> None:
    if ctx.buffers is None or ctx.kept_tokens is None or ctx.kept_docs is None:
        raise RuntimeError("ctx not initialized")
    if (
        ctx.overall is None
        or ctx.writer is None
        or ctx.tok is None
        or ctx.dtype is None
    ):
        raise RuntimeError("ctx not initialized")

    texts = ctx.buffers.get(bucket)
    if not texts:
        return

    arr = _encode_batch_flat(
        ctx.tok,
        list(texts),
        eos_id=int(ctx.eos_id),
        dtype=np.dtype(ctx.dtype),
        max_doc_tokens=int(ctx.max_doc_tokens),
        min_chunk_tokens=int(ctx.min_chunk_tokens),
    )
    ctx.buffers[bucket] = []
    if arr.size <= 0:
        return

    ctx.writer.write(arr)
    n_tok = int(arr.size)
    ctx.kept_tokens[bucket] = int(ctx.kept_tokens.get(bucket, 0)) + n_tok
    ctx.kept_docs[bucket] = int(ctx.kept_docs.get(bucket, 0)) + int(len(texts))
    ctx.overall.add_kept(bucket=bucket, docs=int(len(texts)), tokens=n_tok)


def _maybe_log_progress(ctx: _BuildCtx, *, force: bool = False) -> None:
    if ctx.overall is None or ctx.quotas is None or ctx.writer is None:
        raise RuntimeError("ctx not initialized")

    if int(ctx.progress_every_docs) <= 0 and not force:
        return
    if not force and int(ctx.overall.docs_kept) % int(ctx.progress_every_docs) != 0:
        return
    total_target = int(sum(int(v) for v in ctx.quotas.values()))
    pct = (float(ctx.overall.tokens_kept) / float(max(total_target, 1))) * 100.0
    print(
        f"[PROG] kept_docs={ctx.overall.docs_kept:,} kept_tokens={ctx.overall.tokens_kept:,} "
        f"({pct:.2f}%) shards={len(ctx.writer.shards)}",
        flush=True,
    )
    # Write a lightweight checkpoint for robust resume across interruptions/timeouts.
    # Resume logic only requires `kept_tokens_eos` (and optionally `kept_docs`) to
    # compute accurate remaining quotas.
    try:
        kept_tokens = ctx.kept_tokens or {}
        kept_docs = ctx.kept_docs or {}
        ckpt = {
            "kind": "mix_report_checkpoint",
            "incomplete": True,
            "preset": ctx.preset.name,
            "tokenizer_path": os.path.abspath(str(ctx.tokenizer_path)),
            "out_dir": os.path.abspath(str(ctx.out_dir)),
            "targets_tokens_eos": {k: int(v) for k, v in (ctx.quotas or {}).items()},
            "kept_tokens_eos": {k: int(v) for k, v in sorted(kept_tokens.items())},
            "kept_docs": {k: int(v) for k, v in sorted(kept_docs.items())},
            "resume": {
                "enabled": bool(ctx.resume),
                "existing_shards": int(ctx.resume_existing_shards),
                "existing_tokens": int(ctx.resume_existing_tokens),
                "deleted_tmp": int(ctx.resume_deleted_tmp),
            },
            "manifest": {
                "dtype": str(ctx.dtype),
                "eos_token_id": int(ctx.eos_id),
                "shards": int(len(ctx.writer.shards)),
                "total_tokens": int(sum(int(s.tokens) for s in (ctx.writer.shards or []))),
            },
        }
        _write_json(os.path.join(str(ctx.out_dir), "mix_report.json"), ckpt)
    except Exception:
        # Best-effort: training correctness does not depend on checkpoints,
        # but resume fidelity does, so we avoid failing the build on checkpoint issues.
        pass


def _should_keep_bucket(ctx: _BuildCtx, bucket: str) -> bool:
    if ctx.quotas is None or ctx.kept_tokens is None:
        raise RuntimeError("ctx not initialized")
    q = int(ctx.quotas.get(bucket, 0))
    if q <= 0:
        return False
    return int(ctx.kept_tokens.get(bucket, 0)) < q


def _append_text_to_bucket(ctx: _BuildCtx, *, bucket: str, text: str) -> None:
    if ctx.buffers is None:
        raise RuntimeError("ctx not initialized")
    ctx.buffers[bucket].append(text)
    if len(ctx.buffers[bucket]) >= int(ctx.batch_texts):
        _flush_bucket(ctx, bucket)
        _maybe_log_progress(ctx)


def _process_text_for_bucket(ctx: _BuildCtx, *, bucket: str, cleaned: str) -> None:
    if not _should_keep_bucket(ctx, bucket):
        if ctx.overall is not None:
            ctx.overall.add_drop("quota_met")
        return

    if int(ctx.max_doc_chars) > 0:
        emitted = False
        for chunk in _iter_text_chunks(
            cleaned,
            max_chars=int(ctx.max_doc_chars),
            min_chars=int(ctx.min_chunk_chars),
        ):
            emitted = True
            _append_text_to_bucket(ctx, bucket=bucket, text=chunk)
            if not _should_keep_bucket(ctx, bucket):
                break
        if not emitted and ctx.overall is not None:
            ctx.overall.add_drop("empty_after_split")
        return

    _append_text_to_bucket(ctx, bucket=bucket, text=cleaned)


def _process_row(
    ctx: _BuildCtx,
    *,
    src: SourcePreset,
    raw: object,
    ttl: object,
    bucket_value: object,
    score_value: object,
) -> None:
    if ctx.overall is None or ctx.quotas is None:
        raise RuntimeError("ctx not initialized")

    ctx.overall.docs_in += 1

    score = _safe_float(score_value)
    if (
        src.min_score is not None
        and score is not None
        and float(score) < float(src.min_score)
    ):
        ctx.overall.add_drop("score_low")
        return
    if (
        src.max_score is not None
        and score is not None
        and float(score) > float(src.max_score)
    ):
        ctx.overall.add_drop("score_high")
        return

    body = str(raw or "")
    if src.title_col:
        title_s = normalize_text(str(ttl or ""))
        if title_s:
            body = f"{title_s}{src.title_sep}{body}" if body else title_s

    cleaned, drop_reason = _maybe_clean_text(
        text=body, clean=ctx.clean, src_name=str(src.name)
    )
    if drop_reason:
        ctx.overall.add_drop(drop_reason)
        return

    bucket = _infer_bucket_for_source(
        src=src,
        text=cleaned,
        bucket_value=str(bucket_value) if bucket_value is not None else None,
    )
    if int(ctx.quotas.get(bucket, 0)) <= 0:
        ctx.overall.add_drop("bucket_off")
        return

    _process_text_for_bucket(ctx, bucket=bucket, cleaned=cleaned)


def _source_required_columns(src: SourcePreset) -> list[str]:
    cols = [src.text_col]
    if src.title_col:
        cols.append(src.title_col)
    if src.bucket_mode == "column" and src.bucket_col:
        cols.append(src.bucket_col)
    if src.score_col:
        cols.append(src.score_col)
    return list(dict.fromkeys(cols))


def _iter_source_files(ctx: _BuildCtx, src: SourcePreset) -> list[str]:
    if ctx.rng is None:
        raise RuntimeError("ctx not initialized")
    files = _list_glob(src.parquet_glob)
    if files:
        ctx.rng.shuffle(files)
    return files


def _source_has_remaining_quota(ctx: _BuildCtx, src: SourcePreset) -> bool:
    if ctx.quotas is None:
        return False
    kept = ctx.kept_tokens or {}
    prefix = f"{str(src.name)}:"
    for k, q in ctx.quotas.items():
        if not str(k).startswith(prefix):
            continue
        if int(kept.get(k, 0)) < int(q):
            return True
    return False


def _process_source(ctx: _BuildCtx, src: SourcePreset) -> None:
    if ctx.quotas is None:
        raise RuntimeError("ctx not initialized")
    if _all_quotas_met(kept=(ctx.kept_tokens or {}), quotas=ctx.quotas):
        return
    if not _source_has_remaining_quota(ctx, src):
        return

    files = _iter_source_files(ctx, src)
    if not files:
        print(
            f"[WARN] no files for source={src.name!r} glob={src.parquet_glob!r}",
            flush=True,
        )
        return

    cols = _source_required_columns(src)
    for fp in files:
        if _all_quotas_met(kept=(ctx.kept_tokens or {}), quotas=ctx.quotas):
            break
        if not _source_has_remaining_quota(ctx, src):
            break
        for df in _iter_parquet_batches(
            fp=fp, cols=cols, chunk_rows=int(ctx.chunk_rows)
        ):
            if _all_quotas_met(kept=(ctx.kept_tokens or {}), quotas=ctx.quotas):
                break
            if not _source_has_remaining_quota(ctx, src):
                break

            texts = df.get_column(src.text_col).to_list()
            titles = (
                df.get_column(src.title_col).to_list()
                if src.title_col and src.title_col in df.columns
                else None
            )
            buckets = (
                df.get_column(src.bucket_col).to_list()
                if src.bucket_mode == "column"
                and src.bucket_col
                and src.bucket_col in df.columns
                else None
            )
            scores = (
                df.get_column(src.score_col).to_list()
                if src.score_col and src.score_col in df.columns
                else None
            )

            if titles is None:
                titles = [None] * len(texts)
            if buckets is None:
                buckets = [None] * len(texts)
            if scores is None:
                scores = [None] * len(texts)

            for raw, ttl, bval, sc in zip(texts, titles, buckets, scores, strict=False):
                if _all_quotas_met(kept=(ctx.kept_tokens or {}), quotas=ctx.quotas):
                    break
                _process_row(
                    ctx,
                    src=src,
                    raw=raw,
                    ttl=ttl,
                    bucket_value=bval,
                    score_value=sc,
                )


def _flush_underfilled_buckets(ctx: _BuildCtx) -> None:
    if ctx.quotas is None or ctx.kept_tokens is None:
        raise RuntimeError("ctx not initialized")
    for bucket, q in ctx.quotas.items():
        if int(q) <= 0:
            continue
        if int(ctx.kept_tokens.get(bucket, 0)) >= int(q):
            continue
        _flush_bucket(ctx, bucket)
        _maybe_log_progress(ctx)


def _finalize_build(ctx: _BuildCtx, *, elapsed_sec: float) -> None:
    if (
        ctx.writer is None
        or ctx.dtype is None
        or ctx.quotas is None
        or ctx.overall is None
    ):
        raise RuntimeError("ctx not initialized")

    ctx.writer.close()
    tok_sha1 = compute_tokenizer_bundle_sha1(os.path.abspath(str(ctx.tokenizer_path)))
    manifest = TokenShardManifest(
        dtype=_manifest_dtype_name(np.dtype(ctx.dtype)),
        shards=tuple(ctx.writer.shards),
        eos_token_id=int(ctx.eos_id),
        tokenizer_sha1=tok_sha1,
    )
    save_manifest(os.path.join(ctx.out_dir, "manifest.json"), manifest)

    kept_tokens = ctx.kept_tokens or {}
    kept_docs = ctx.kept_docs or {}
    report = {
        "preset": ctx.preset.name,
        "tokenizer_path": os.path.abspath(str(ctx.tokenizer_path)),
        "out_dir": os.path.abspath(str(ctx.out_dir)),
        "elapsed_sec": float(elapsed_sec),
        "rate_tokens_per_sec": float(ctx.overall.tokens_kept)
        / float(max(elapsed_sec, 1e-6)),
        "rate_docs_per_sec": float(ctx.overall.docs_kept)
        / float(max(elapsed_sec, 1e-6)),
        "docs_in": int(ctx.overall.docs_in),
        "docs_kept": int(ctx.overall.docs_kept),
        "tokens_kept": int(ctx.overall.tokens_kept),
        "targets_tokens_eos": {k: int(v) for k, v in ctx.quotas.items()},
        "kept_tokens_eos": {k: int(v) for k, v in sorted(kept_tokens.items())},
        "kept_docs": {k: int(v) for k, v in sorted(kept_docs.items())},
        "dropped": dict(
            sorted((ctx.overall.dropped or {}).items(), key=lambda kv: kv[0])
        ),
        "resume": {
            "enabled": bool(ctx.resume),
            "existing_shards": int(ctx.resume_existing_shards),
            "existing_tokens": int(ctx.resume_existing_tokens),
            "deleted_tmp": int(ctx.resume_deleted_tmp),
        },
        "manifest": {
            "dtype": manifest.dtype,
            "eos_token_id": manifest.eos_token_id,
            "tokenizer_sha1": manifest.tokenizer_sha1,
            "total_tokens": int(manifest.total_tokens),
            "shards": int(len(manifest.shards)),
            "new_tokens": int(ctx.overall.tokens_kept),
            "existing_tokens": int(ctx.resume_existing_tokens),
        },
        "build_args": {
            "shard_size_tokens": int(ctx.shard_size_tokens),
            "out_dtype": str(ctx.out_dtype),
            "batch_texts": int(ctx.batch_texts),
            "chunk_rows": int(ctx.chunk_rows),
            "max_doc_chars": int(ctx.max_doc_chars),
            "min_chunk_chars": int(ctx.min_chunk_chars),
            "max_doc_tokens": int(ctx.max_doc_tokens),
            "min_chunk_tokens": int(ctx.min_chunk_tokens),
            "seed": int(ctx.seed),
            "max_passes": int(ctx.max_passes),
            "progress_every_docs": int(ctx.progress_every_docs),
            "clean": ctx.clean.__dict__,
        },
    }

    report_path = os.path.join(ctx.out_dir, "mix_report.json")
    _write_json(report_path, report)

    if str(ctx.report_tag or "").strip():
        tag_path = os.path.join(ctx.out_dir, f"mix_report.{ctx.report_tag}.json")
        _write_json(tag_path, report)
        print(f"[AUDIT] wrote tagged report: {os.path.basename(tag_path)}", flush=True)

    # If this is a backfill-resume run and a primary report is present, write a merged final report.
    if str(ctx.resume_backfill_bucket or "").strip():
        primary_path = os.path.join(ctx.out_dir, "mix_report.primary.json")
        primary = _load_json(primary_path)
        if isinstance(primary, dict) and primary:
            final = _audit_merge_primary_backfill(primary=primary, backfill=report)
            final_path = os.path.join(ctx.out_dir, "mix_report.final.json")
            _write_json(final_path, final)
            print(
                f"[AUDIT] wrote merged final report: {os.path.basename(final_path)}",
                flush=True,
            )

    _maybe_log_progress(ctx, force=True)
    # `force=True` writes a checkpoint snapshot; restore the final audit report
    # afterward so the stable `mix_report.json` is not left in checkpoint form.
    _write_json(report_path, report)
    print(f"[OK] wrote shards to: {ctx.out_dir}", flush=True)

    if not _all_quotas_met(kept=kept_tokens, quotas=ctx.quotas):
        missing = {
            k: int(v) - int(kept_tokens.get(k, 0))
            for k, v in ctx.quotas.items()
            if int(v) > 0
        }
        missing = {k: v for k, v in missing.items() if int(v) > 0}
        if missing:
            print("[WARN] unmet quotas (tokens_eos):", flush=True)
            for k, v in sorted(
                missing.items(), key=lambda kv: int(kv[1]), reverse=True
            ):
                print(f"  - {k}: {int(v):,}", flush=True)


def _build_mix_token_shards(ctx: _BuildCtx) -> None:
    _init_build_ctx(ctx)
    if ctx.quotas is None:
        raise RuntimeError("ctx not initialized")
    print(_plan_string(quotas=ctx.quotas), flush=True)

    t0 = time.perf_counter()
    passes = max(int(ctx.max_passes), 1)
    for p in range(passes):
        if _all_quotas_met(kept=(ctx.kept_tokens or {}), quotas=ctx.quotas):
            break
        print(f"[PASS] {p + 1}/{passes}", flush=True)
        sources = tuple(ctx.preset.sources)
        if ctx.source_allowlist:
            allow = {
                str(s).strip() for s in ctx.source_allowlist if str(s).strip()
            }
            sources = tuple(s for s in sources if str(s.name) in allow)
        for src in sources:
            if _all_quotas_met(kept=(ctx.kept_tokens or {}), quotas=ctx.quotas):
                break
            _process_source(ctx, src)
        _flush_underfilled_buckets(ctx)

    # Final best-effort flush.
    _flush_underfilled_buckets(ctx)
    dt = max(1e-6, time.perf_counter() - t0)
    _finalize_build(ctx, elapsed_sec=float(dt))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build pretrain mix token-shards from parquet"
    )
    p.add_argument("--preset", type=str, default="pretrain")
    p.add_argument(
        "--tokenizer_path",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "modeling" / "text"),
        help="Tokenizer directory (expects tokenizer.json). Defaults to the bundled runtime tokenizer.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("dataset", "pretrain", "pretrain"),
    )
    p.add_argument(
        "--total_tokens_eos",
        type=int,
        default=0,
        help="Override preset total token budget (0=use preset)",
    )
    p.add_argument(
        "--resume",
        type=int,
        default=0,
        choices=[0, 1],
        help="Append to existing shard_*.bin in out_dir and adjust quotas to reach the target token budget",
    )
    p.add_argument(
        "--resume_backfill_bucket",
        type=str,
        default="",
        help="When resuming, ignore remaining per-bucket quotas and backfill the remaining token budget using this single bucket (e.g. zh_books:UNKNOWN)",
    )
    p.add_argument(
        "--report_tag",
        type=str,
        default="",
        help="Also write mix_report.<tag>.json in out_dir (recommended: primary/backfill)",
    )

    p.add_argument("--shard_size_tokens", type=int, default=20_000_000)
    p.add_argument(
        "--out_dtype",
        type=str,
        default="int32",
        choices=["int32"],
    )
    p.add_argument("--batch_texts", type=int, default=512)
    p.add_argument(
        "--chunk_rows",
        type=int,
        default=50_000,
        help="Rows per parquet slice for large files",
    )

    p.add_argument("--max_doc_chars", type=int, default=0)
    p.add_argument("--min_chunk_chars", type=int, default=200)
    p.add_argument("--max_doc_tokens", type=int, default=4096)
    p.add_argument("--min_chunk_tokens", type=int, default=1)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_passes",
        type=int,
        default=1,
        help="Repeat input scans to upsample scarce buckets",
    )
    p.add_argument("--progress_every_docs", type=int, default=200_000)
    p.add_argument(
        "--source_allowlist",
        type=str,
        nargs="*",
        default=None,
        help="Optional: build shards only from these sources (e.g. code math zh_wiki).",
    )
    p.add_argument(
        "--source_allowlist_fill_budget",
        type=int,
        default=1,
        choices=[0, 1],
        help="When --source_allowlist is set: renormalize quotas to fill the full token budget (1 recommended).",
    )

    # Cleaning knobs (high-value defaults; can be relaxed for raw builds).
    p.add_argument("--min_chars", type=int, default=200)
    p.add_argument("--drop_human_bot_tags", type=int, default=1, choices=[0, 1])
    p.add_argument("--drop_ai_mentions", type=int, default=1, choices=[0, 1])
    p.add_argument("--drop_placeholders", type=int, default=1, choices=[0, 1])
    p.add_argument("--drop_poison", type=int, default=1, choices=[0, 1])
    p.add_argument("--drop_repeated_sentences", type=int, default=1, choices=[0, 1])
    p.add_argument("--drop_mojibake", type=int, default=1, choices=[0, 1])
    p.add_argument(
        "--drop_code_blobs",
        type=int,
        default=1,
        choices=[0, 1],
        help="Drop high-confidence generated/bundled code data blobs while building token shards.",
    )
    p.add_argument("--mojibake_max_frac", type=float, default=0.02)

    return p.parse_args()


@dataclass(frozen=True)
class MixTokenShardBuildConfig:
    preset_name: str
    tokenizer_path: str
    out_dir: str
    shard_size_tokens: int
    out_dtype: str
    batch_texts: int
    chunk_rows: int
    max_doc_chars: int
    min_chunk_chars: int
    max_doc_tokens: int
    min_chunk_tokens: int
    seed: int
    max_passes: int
    progress_every_docs: int
    total_tokens_eos_override: int
    source_allowlist: tuple[str, ...]
    source_allowlist_fill_budget: bool
    resume: bool
    resume_backfill_bucket: str
    report_tag: str
    min_chars: int
    drop_human_bot_tags: bool
    drop_ai_mentions: bool
    drop_placeholders: bool
    drop_poison: bool
    drop_repeated_sentences: bool
    drop_mojibake: bool
    drop_code_blobs: bool
    mojibake_max_frac: float

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> MixTokenShardBuildConfig:
        return cls(
            preset_name=str(args.preset),
            tokenizer_path=str(args.tokenizer_path),
            out_dir=str(args.out_dir),
            shard_size_tokens=int(args.shard_size_tokens),
            out_dtype=str(args.out_dtype),
            batch_texts=int(args.batch_texts),
            chunk_rows=int(args.chunk_rows),
            max_doc_chars=int(args.max_doc_chars),
            min_chunk_chars=int(args.min_chunk_chars),
            max_doc_tokens=int(args.max_doc_tokens),
            min_chunk_tokens=int(args.min_chunk_tokens),
            seed=int(args.seed),
            max_passes=int(args.max_passes),
            progress_every_docs=int(args.progress_every_docs),
            total_tokens_eos_override=int(args.total_tokens_eos),
            source_allowlist=tuple(
                str(source).strip()
                for source in (args.source_allowlist or [])
                if str(source).strip()
            ),
            source_allowlist_fill_budget=bool(
                int(args.source_allowlist_fill_budget) == 1
            ),
            resume=bool(int(args.resume) == 1),
            resume_backfill_bucket=str(args.resume_backfill_bucket or "").strip(),
            report_tag=str(args.report_tag or "").strip(),
            min_chars=int(args.min_chars),
            drop_human_bot_tags=bool(int(args.drop_human_bot_tags) == 1),
            drop_ai_mentions=bool(int(args.drop_ai_mentions) == 1),
            drop_placeholders=bool(int(args.drop_placeholders) == 1),
            drop_poison=bool(int(args.drop_poison) == 1),
            drop_repeated_sentences=bool(int(args.drop_repeated_sentences) == 1),
            drop_mojibake=bool(int(args.drop_mojibake) == 1),
            drop_code_blobs=bool(int(args.drop_code_blobs) == 1),
            mojibake_max_frac=float(args.mojibake_max_frac),
        )

    @property
    def preset(self) -> PretrainMixPreset:
        return get_preset(self.preset_name)

    @property
    def clean_config(self) -> _CleanCfg:
        return _CleanCfg(
            min_chars=int(self.min_chars),
            drop_human_bot_tags=bool(self.drop_human_bot_tags),
            drop_ai_mentions=bool(self.drop_ai_mentions),
            drop_placeholders=bool(self.drop_placeholders),
            drop_poison=bool(self.drop_poison),
            drop_repeated_sentences=bool(self.drop_repeated_sentences),
            drop_mojibake=bool(self.drop_mojibake),
            drop_code_blobs=bool(self.drop_code_blobs),
            mojibake_max_frac=float(self.mojibake_max_frac),
        )

    def run(self) -> None:
        build_mix_token_shards(
            preset=self.preset,
            tokenizer_path=str(self.tokenizer_path),
            out_dir=str(self.out_dir),
            shard_size_tokens=int(self.shard_size_tokens),
            out_dtype=str(self.out_dtype),
            batch_texts=int(self.batch_texts),
            chunk_rows=int(self.chunk_rows),
            max_doc_chars=int(self.max_doc_chars),
            min_chunk_chars=int(self.min_chunk_chars),
            max_doc_tokens=int(self.max_doc_tokens),
            min_chunk_tokens=int(self.min_chunk_tokens),
            seed=int(self.seed),
            max_passes=int(self.max_passes),
            progress_every_docs=int(self.progress_every_docs),
            clean=self.clean_config,
            total_tokens_eos_override=int(self.total_tokens_eos_override),
            source_allowlist=tuple(self.source_allowlist),
            source_allowlist_fill_budget=bool(self.source_allowlist_fill_budget),
            resume=bool(self.resume),
            resume_backfill_bucket=str(self.resume_backfill_bucket),
            report_tag=str(self.report_tag),
        )


def main() -> None:
    config = MixTokenShardBuildConfig.from_namespace(_parse_args())
    config.run()


if __name__ == "__main__":  # pragma: no cover
    main()
