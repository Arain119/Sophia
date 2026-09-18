#!/usr/bin/env python
"""
Build a FULL pretrain token-shards dataset from the local parquet pool.

Unlike `ml.tooling.pipelines.pretrain_mix_token_shards` (which enforces
per-bucket *quotas* and therefore downsamples abundant buckets), this walks every
parquet file under `dataset/pretrain/<split>/` and tokenizes every document once.
There is no token budget and no quota throttling — the output contains all data
that survives quality cleaning.

Cleaning parity:
  - Reuses the exact `_maybe_clean_text` / `_CleanCfg` logic from the mix pipeline.
  - Reuses the exact tokenization + shard-writing logic from `shard_builder`
    (so output is byte-for-byte consistent with the training data path).

Quality gates kept (these are curation, not balancing):
  - score >= --min_score for any source that carries a `score` column (math, code).
  - mojibake / poison / repeated-sentence / placeholder / ai-mention / code-blob drops.

Speed:
  - Tokenization is multi-threaded via the Rust tokenizer. Export
    RAYON_NUM_THREADS / TOKENIZERS_PARALLELISM *before* launching to override the
    single-thread defaults set by `shard_builder`'s env hardening, e.g.:
      RAYON_NUM_THREADS=12 TOKENIZERS_PARALLELISM=true python -m ... build_full_token_shards

Example:
  RAYON_NUM_THREADS=12 TOKENIZERS_PARALLELISM=true \
  python -m ml.tooling.scripts.data.build_full_token_shards \
    --split train \
    --tokenizer_path ml/modeling/text \
    --out_dir /tmp/pretrain_build/train \
    --shard_size_tokens 50000000 --out_dtype int32 --max_doc_tokens 4096
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from ml.errors import SophiaUsageError
import pyarrow.parquet as pq

from ml.data.token_shards.shard_manifest import (
    TokenShardManifest,
    save_manifest,
)
from ml.data.token_shards.tokenizer_fingerprint import (
    compute_tokenizer_bundle_sha1,
)
from ml.tooling.core.parquet_inventory import list_tree_parquets_prefer_plain
from ml.data.pretrain_filters import normalize_text
from ml.tooling.pipelines.pretrain_mix_token_shards import (
    _CleanCfg,
    _maybe_clean_text,
)
from ml.training.pretrain.shard_builder import (
    ShardWriter,
    _encode_batch_flat,
    _manifest_dtype_name,
    _resolve_out_dtype,
    _load_tokenizer,
    _token_id_upper_bound,
)

# Column preference: tokenize the first text-bearing column that exists.
_TEXT_COL_CANDIDATES = ("text", "content")
_TITLE_COL = "title"
_SCORE_COL = "score"


def _pick_columns(schema_names: list[str]) -> tuple[str | None, str | None, str | None]:
    names = set(schema_names)
    text_col = next((c for c in _TEXT_COL_CANDIDATES if c in names), None)
    title_col = _TITLE_COL if _TITLE_COL in names else None
    score_col = _SCORE_COL if _SCORE_COL in names else None
    return text_col, title_col, score_col


def _is_code_path(path: str) -> bool:
    p = str(path).replace("\\", "/").lower()
    return "/pretrain/" in p and "/code/" in p


def _is_math_path(path: str) -> bool:
    p = str(path).replace("\\", "/").lower()
    return "/pretrain/" in p and "/math/" in p


def _safe_float(x: object) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _iter_clean_texts(
    *,
    files: list[str],
    clean: _CleanCfg,
    min_score: float,
    min_score_math: float,
    chunk_rows: int,
    stats: dict[str, int],
    drops: dict[str, int],
) -> Iterator[str]:
    """Yield cleaned document strings from every parquet file."""
    for fp in files:
        try:
            pf = pq.ParquetFile(fp)
        except Exception:
            drops["unreadable_file"] = drops.get("unreadable_file", 0) + 1
            continue
        text_col, title_col, score_col = _pick_columns(list(pf.schema_arrow.names))
        if text_col is None:
            drops["no_text_column"] = drops.get("no_text_column", 0) + 1
            continue
        is_code = _is_code_path(fp)
        src_name = "code" if is_code else os.path.basename(os.path.dirname(fp))
        # Math carries binary curation flags (1.0 = kept upstream), not the
        # graded 0-10 quality scores the code pool uses.
        file_min_score = float(min_score_math) if _is_math_path(fp) else float(min_score)
        cols = [text_col]
        if title_col:
            cols.append(title_col)
        if score_col:
            cols.append(score_col)

        for batch in pf.iter_batches(batch_size=int(chunk_rows), columns=cols):
            texts = batch.column(text_col).to_pylist()
            titles = batch.column(title_col).to_pylist() if title_col else None
            scores = batch.column(score_col).to_pylist() if score_col else None
            n = len(texts)
            for i in range(n):
                stats["docs_in"] += 1
                if scores is not None:
                    sv = _safe_float(scores[i])
                    if sv is not None and sv < float(file_min_score):
                        drops["score_low"] = drops.get("score_low", 0) + 1
                        continue
                body = str(texts[i] or "")
                if title_col:
                    ttl = normalize_text(str((titles[i] if titles else "") or ""))
                    if ttl:
                        body = f"{ttl}\n\n{body}" if body else ttl
                cleaned, reason = _maybe_clean_text(
                    text=body, clean=clean, src_name=src_name
                )
                if reason:
                    drops[reason] = drops.get(reason, 0) + 1
                    continue
                stats["docs_kept"] += 1
                yield cleaned


def _write_json_atomic(path: str, obj: object) -> None:
    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(p))


def build_full(
    *,
    split: str,
    tokenizer_path: str,
    out_dir: str,
    shard_size_tokens: int,
    out_dtype: str,
    batch_texts: int,
    chunk_rows: int,
    max_doc_tokens: int,
    min_chunk_tokens: int,
    min_score: float,
    min_score_math: float,
    clean: _CleanCfg,
    progress_every_docs: int,
    overwrite: bool,
    part_index: int = 0,
    part_count: int = 1,
) -> None:
    root = os.path.join("dataset", "pretrain", str(split))
    if not os.path.isdir(root):
        raise SophiaUsageError(f"[ERR] split pool not found: {root}")

    files = [str(p) for p in list_tree_parquets_prefer_plain(root)]
    if not files:
        raise SophiaUsageError(f"[ERR] no parquet files under: {root}")
    if part_count > 1:
        files = files[part_index::part_count]
        if not files:
            raise SophiaUsageError(
                f"[ERR] partition {part_index}/{part_count} selected no files"
            )
    print(
        f"[SCAN] split={split} files={len(files)} "
        f"part={part_index}/{part_count}",
        flush=True,
    )

    out_dir = os.path.abspath(str(out_dir))
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not overwrite:
            raise SophiaUsageError(
                f"[ERR] out_dir is non-empty: {out_dir} (use --overwrite 1)"
            )
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    tok = _load_tokenizer(str(tokenizer_path))
    eos_id = getattr(tok, "eos_token_id", None)
    if not isinstance(eos_id, int):
        raise SophiaUsageError("[ERR] tokenizer has no eos_token_id")
    vocab_size = max(int(_token_id_upper_bound(tok) or 0), int(eos_id) + 1)
    dtype = _resolve_out_dtype(out_dtype=str(out_dtype), vocab_size=vocab_size)

    writer = ShardWriter(
        out_dir, shard_size_tokens=max(int(shard_size_tokens), 1024), dtype=dtype
    )

    stats = {"docs_in": 0, "docs_kept": 0}
    drops: dict[str, int] = {}
    total_tokens = 0
    last_log = 0
    t0 = time.perf_counter()

    def flush(buf: list[str]) -> int:
        if not buf:
            return 0
        arr = _encode_batch_flat(
            tok,
            buf,
            eos_id=int(eos_id),
            dtype=dtype,
            max_doc_tokens=int(max_doc_tokens),
            min_chunk_tokens=int(min_chunk_tokens),
        )
        if arr.size <= 0:
            return 0
        writer.write(arr)
        return int(arr.size)

    try:
        buf: list[str] = []
        for text in _iter_clean_texts(
            files=files,
            clean=clean,
            min_score=float(min_score),
            min_score_math=float(min_score_math),
            chunk_rows=int(chunk_rows),
            stats=stats,
            drops=drops,
        ):
            buf.append(text)
            if len(buf) >= int(batch_texts):
                total_tokens += flush(buf)
                buf = []
                if (
                    progress_every_docs > 0
                    and stats["docs_kept"] - last_log >= progress_every_docs
                ):
                    last_log = stats["docs_kept"]
                    el = max(1e-6, time.perf_counter() - t0)
                    print(
                        f"[TOK] in={stats['docs_in']:,} kept={stats['docs_kept']:,} "
                        f"tokens={total_tokens:,} {int(total_tokens / el):,} tok/s "
                        f"shards={len(writer.shards)}",
                        flush=True,
                    )
        total_tokens += flush(buf)
        writer.close()

        tok_sha1 = compute_tokenizer_bundle_sha1(os.path.abspath(str(tokenizer_path)))
        manifest = TokenShardManifest(
            dtype=_manifest_dtype_name(dtype),
            shards=tuple(writer.shards),
            eos_token_id=int(eos_id),
            tokenizer_sha1=tok_sha1,
        )
        save_manifest(os.path.join(out_dir, "manifest.json"), manifest)
    except BaseException:
        writer.close()
        raise

    el = max(1e-6, time.perf_counter() - t0)
    report = {
        "mode": "full_no_quota",
        "split": str(split),
        "out_dir": out_dir,
        "files": len(files),
        "docs_in": int(stats["docs_in"]),
        "docs_kept": int(stats["docs_kept"]),
        "tokens_total": int(total_tokens),
        "dtype": manifest.dtype,
        "eos_token_id": int(eos_id),
        "tokenizer_sha1": tok_sha1,
        "min_score": float(min_score),
        "min_score_math": float(min_score_math),
        "elapsed_sec": float(el),
        "rate_tokens_per_sec": float(total_tokens / el),
        "shards": len(writer.shards),
        "dropped": dict(sorted(drops.items(), key=lambda kv: kv[0])),
    }
    _write_json_atomic(os.path.join(out_dir, "build_report.json"), report)
    print(
        f"[OK] {split}: {len(writer.shards)} shards | docs_in={stats['docs_in']:,} "
        f"docs_kept={stats['docs_kept']:,} tokens={total_tokens:,} | "
        f"{int(total_tokens / el):,} tok/s | dropped={report['dropped']}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    p.add_argument("--tokenizer_path", type=str, default="ml/modeling/text")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--shard_size_tokens", type=int, default=50_000_000)
    p.add_argument(
        "--out_dtype",
        type=str,
        default="int32",
        choices=["auto", "uint16", "uint32", "int32"],
    )
    p.add_argument("--batch_texts", type=int, default=1024)
    p.add_argument("--chunk_rows", type=int, default=20_000)
    p.add_argument("--max_doc_tokens", type=int, default=4096)
    p.add_argument("--min_chunk_tokens", type=int, default=1)
    p.add_argument("--min_score", type=float, default=4.5)
    p.add_argument("--min_score_math", type=float, default=0.5)
    p.add_argument("--progress_every_docs", type=int, default=500_000)
    p.add_argument("--overwrite", type=int, default=0, choices=[0, 1])
    # Deterministic file partitioning for parallel workers: each worker takes
    # files[part_index::part_count] and writes an independent shard set/manifest.
    p.add_argument("--part_index", type=int, default=0)
    p.add_argument("--part_count", type=int, default=1)
    p.add_argument("--min_chars", type=int, default=200)
    p.add_argument("--drop_human_bot_tags", type=int, default=1, choices=[0, 1])
    p.add_argument("--drop_ai_mentions", type=int, default=1, choices=[0, 1])
    p.add_argument("--drop_placeholders", type=int, default=1, choices=[0, 1])
    p.add_argument("--drop_poison", type=int, default=1, choices=[0, 1])
    p.add_argument("--drop_repeated_sentences", type=int, default=1, choices=[0, 1])
    p.add_argument("--drop_mojibake", type=int, default=1, choices=[0, 1])
    p.add_argument("--drop_code_blobs", type=int, default=1, choices=[0, 1])
    p.add_argument("--mojibake_max_frac", type=float, default=0.02)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    clean = _CleanCfg(
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
    build_full(
        split=str(args.split),
        tokenizer_path=str(args.tokenizer_path),
        out_dir=str(args.out_dir),
        shard_size_tokens=int(args.shard_size_tokens),
        out_dtype=str(args.out_dtype),
        batch_texts=int(args.batch_texts),
        chunk_rows=int(args.chunk_rows),
        max_doc_tokens=int(args.max_doc_tokens),
        min_chunk_tokens=int(args.min_chunk_tokens),
        min_score=float(args.min_score),
        min_score_math=float(args.min_score_math),
        clean=clean,
        progress_every_docs=int(args.progress_every_docs),
        overwrite=bool(int(args.overwrite) == 1),
        part_index=int(args.part_index),
        part_count=int(args.part_count),
    )


if __name__ == "__main__":
    main()
