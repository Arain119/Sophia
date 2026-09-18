#!/usr/bin/env python
"""
Build offline pre-tokenized shards for high-throughput pretraining.

Output format:
- out_dir/shard_00000.bin, shard_00001.bin, ...
- out_dir/manifest.json

Each shard is a flat stream of token ids with EOS inserted between documents.

Example (from local JSONL):
  python -m ml.training.pretrain.shard_builder ^
    --tokenizer_path ml/modeling/text ^
    --jsonl dataset/pretrain/train-00000-of-00117-*.jsonl ^
    --out_dir /tmp/pretrain_build ^
    --out_dtype int32 ^
    --shard_size_tokens 20000000

Tip:
  - JSONL reading is strict by default: any invalid JSON line aborts the build.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from collections.abc import Iterator

from ml.core.common.mapping import object_mapping
from ml.data.token_shards.shard_manifest import (
    TokenShardManifest,
    save_manifest,
)
from ml.data.token_shards.tokenizer_fingerprint import (
    compute_tokenizer_bundle_sha1,
)
from ml.errors import SophiaUsageError
from ml.data.jsonl_stream import JsonlFieldStats

from ml.training.pretrain.shard_builder_encode import (
    _encode_batch_flat,
    encode_batches,
)
from ml.training.pretrain.shard_builder_output import (
    OUTPUT_DIR_MARKER,
    _prepare_staging_out_dir,
    _promote_staging_out_dir,
    _promotion_backup_dir,
    _write_output_dir_marker,
    _normalize_out_dir,
)
from ml.training.pretrain.shard_builder_text import (
    JsonlTextStream,
    _iter_text_chunks,
)
from ml.training.pretrain.shard_builder_tokenizer import (
    _load_tokenizer,
    _manifest_dtype_name,
    _resolve_out_dtype,
    _token_id_upper_bound,
)
from ml.training.pretrain.shard_builder_writer import ShardWriter


@dataclass(frozen=True)
class ShardBuilderConfig:
    jsonl: tuple[str, ...]
    out_dir: str
    overwrite_output_dir: bool
    tokenizer_path: str
    out_dtype: str
    shard_size_tokens: int
    batch_texts: int
    num_proc: int
    max_docs: int
    max_doc_chars: int
    min_chunk_chars: int
    max_doc_tokens: int
    min_chunk_tokens: int
    progress_every_docs: int

    @classmethod
    def from_args(cls, args: object) -> ShardBuilderConfig:
        payload = object_mapping(args)
        return cls(
            jsonl=tuple(str(path) for path in list(payload.get("jsonl", ()) or ())),
            out_dir=str(payload["out_dir"]),
            overwrite_output_dir=bool(int(payload.get("overwrite_output_dir", 0))),
            tokenizer_path=str(payload["tokenizer_path"]),
            out_dtype=str(payload.get("out_dtype", "int32")),
            shard_size_tokens=int(payload.get("shard_size_tokens", 20_000_000)),
            batch_texts=int(payload.get("batch_texts", 512)),
            num_proc=int(payload.get("num_proc", 1)),
            max_docs=int(payload.get("max_docs", 0)),
            max_doc_chars=int(payload.get("max_doc_chars", 0)),
            min_chunk_chars=int(payload.get("min_chunk_chars", 0)),
            max_doc_tokens=int(payload.get("max_doc_tokens", 0)),
            min_chunk_tokens=int(payload.get("min_chunk_tokens", 1)),
            progress_every_docs=int(payload.get("progress_every_docs", 0)),
        )


def _batch_iter(
    *,
    texts: Iterator[str],
    batch_texts: int,
) -> Iterator[list[str]]:
    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) >= int(batch_texts):
            yield batch
            batch = []
    if batch:
        yield batch


def _print_source_summary(*, stream: JsonlTextStream) -> None:
    stats = stream.stats
    if stats.lines <= 0:
        return
    if not (
        stats.decode_errors > 0
        or stats.non_object > 0
        or stats.missing_field > 0
        or stats.empty_after_strip > 0
    ):
        return
    print(
        f"[SRC] jsonl_lines={stats.lines} decoded={stats.decoded} decode_errors={stats.decode_errors} "
        f"non_object={stats.non_object} missing_field={stats.missing_field} empty_after_strip={stats.empty_after_strip} "
        f"yielded={stats.yielded} emitted_docs={int(stream.emitted_docs)}"
    )


def run(config: ShardBuilderConfig) -> None:
    if not config.jsonl:
        raise SophiaUsageError(
            "Provide at least one source: --jsonl <file/dir> (repeatable)."
        )

    out_dir = _normalize_out_dir(config.out_dir)
    overwrite_output_dir = bool(config.overwrite_output_dir)
    staging_out_dir = _prepare_staging_out_dir(
        out_dir=out_dir,
        overwrite_output_dir=overwrite_output_dir,
    )

    tok_path = os.path.abspath(str(config.tokenizer_path))
    tok = _load_tokenizer(tok_path)
    vocab_size = int(_token_id_upper_bound(tok) or 0)
    eos_id = getattr(tok, "eos_token_id", None)
    if eos_id is None:
        raise SophiaUsageError("Tokenizer has no eos_token_id; set eos_token first.")
    eos_id = int(eos_id)
    vocab_size = max(int(vocab_size), int(eos_id) + 1)

    dtype = _resolve_out_dtype(out_dtype=str(config.out_dtype), vocab_size=vocab_size)
    stream = JsonlTextStream(config=config, stats=JsonlFieldStats())
    writer = ShardWriter(
        staging_out_dir,
        shard_size_tokens=max(int(config.shard_size_tokens), 1024),
        dtype=dtype,
    )
    try:
        total_tokens, total_docs = encode_batches(
            batches=_batch_iter(
                texts=stream.iter_texts(),
                batch_texts=max(int(config.batch_texts), 1),
            ),
            writer=writer,
            tokenizer=tok,
            tokenizer_path=str(tok_path),
            eos_id=int(eos_id),
            dtype=dtype,
            max_doc_tokens=max(int(config.max_doc_tokens), 0),
            min_chunk_tokens=max(int(config.min_chunk_tokens), 1),
            num_proc=max(int(config.num_proc), 1),
            max_docs=max(int(config.max_docs), 0),
            progress_every_docs=int(config.progress_every_docs),
        )

        writer.close()
        manifest = TokenShardManifest(
            dtype=_manifest_dtype_name(dtype),
            shards=tuple(writer.shards),
            eos_token_id=eos_id,
            tokenizer_sha1=compute_tokenizer_bundle_sha1(tok_path),
        )
        save_manifest(os.path.join(staging_out_dir, "manifest.json"), manifest)
        _write_output_dir_marker(staging_out_dir)
        _promote_staging_out_dir(
            staging_out_dir=staging_out_dir,
            final_out_dir=out_dir,
            overwrite_output_dir=overwrite_output_dir,
        )
    except BaseException:
        writer.close()
        shutil.rmtree(staging_out_dir, ignore_errors=True)
        raise

    print(f"[OK] wrote {len(writer.shards)} shards to: {out_dir}")
    print(
        f"total_docs~={total_docs} | total_tokens~={total_tokens} | dtype={manifest.dtype} | eos_token_id={eos_id}"
    )
    _print_source_summary(stream=stream)


def main() -> None:
    from ml.cli.shard_builder import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()


__all__ = [
    "OUTPUT_DIR_MARKER",
    "ShardBuilderConfig",
    "ShardWriter",
    "_encode_batch_flat",
    "_iter_text_chunks",
    "_load_tokenizer",
    "_manifest_dtype_name",
    "_normalize_out_dir",
    "_prepare_staging_out_dir",
    "_print_source_summary",
    "_promote_staging_out_dir",
    "_promotion_backup_dir",
    "_resolve_out_dtype",
    "_token_id_upper_bound",
    "_write_output_dir_marker",
    "_batch_iter",
    "encode_batches",
    "main",
    "run",
]
