from __future__ import annotations

import sys
import time
from collections.abc import Iterator, Sequence

import numpy as np

from ml.training.pretrain.shard_builder_tokenizer import _load_tokenizer
from ml.training.pretrain.shard_builder_writer import ShardWriter


def _tokenize_texts(tokenizer: object, texts: Sequence[str]) -> list[list[int]]:
    ids_list: list[list[int]] = []
    encode_batch = getattr(tokenizer, "encode_batch", None)
    if callable(encode_batch):
        encs = encode_batch(list(texts))
        ids_list = [list(getattr(item, "ids", ()) or ()) for item in encs]
    else:
        enc = tokenizer(
            list(texts),
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        maybe_ids = enc.get("input_ids") or []
        if isinstance(maybe_ids, list):
            ids_list = [list(ids or ()) for ids in maybe_ids]
    return ids_list


def _count_segments(*, token_count: int, max_doc_tokens: int, min_chunk_tokens: int) -> int:
    if int(max_doc_tokens) <= 0:
        return 1
    segments = int((int(token_count) + int(max_doc_tokens) - 1) // int(max_doc_tokens))
    if int(min_chunk_tokens) > 1 and int(segments) > 1:
        remainder = int(token_count) % int(max_doc_tokens)
        if 0 < int(remainder) < int(min_chunk_tokens):
            segments -= 1
    return int(segments)


def _write_doc_chunks(
    *,
    flat: np.ndarray,
    pos: int,
    ids: list[int],
    eos_id: int,
    max_doc_tokens: int,
    min_chunk_tokens: int,
) -> int:
    if int(max_doc_tokens) <= 0:
        token_count = int(len(ids))
        flat[pos : pos + token_count] = ids
        pos += token_count
        flat[pos] = int(eos_id)
        return pos + 1

    start = 0
    token_count = int(len(ids))
    merge_tail = False
    last_start = -1
    if int(min_chunk_tokens) > 1 and token_count > int(max_doc_tokens):
        remainder = int(token_count % int(max_doc_tokens))
        if 0 < remainder < int(min_chunk_tokens):
            merge_tail = True
            last_start = int(token_count - (int(max_doc_tokens) + remainder))
    while start < token_count:
        if merge_tail and start == last_start:
            end = token_count
        else:
            end = min(start + int(max_doc_tokens), token_count)
        chunk = ids[start:end]
        if not chunk:
            break
        chunk_len = int(len(chunk))
        flat[pos : pos + chunk_len] = chunk
        pos += chunk_len
        flat[pos] = int(eos_id)
        pos += 1
        start = end
    return pos


def _encode_batch_flat(
    tokenizer: object,
    texts: Sequence[str],
    *,
    eos_id: int,
    dtype: np.dtype,
    max_doc_tokens: int,
    min_chunk_tokens: int,
) -> np.ndarray:
    if not texts:
        return np.empty((0,), dtype=dtype)

    ids_list = _tokenize_texts(tokenizer, texts)
    if not ids_list:
        return np.empty((0,), dtype=dtype)

    max_doc_tokens = int(max_doc_tokens)
    min_chunk_tokens = int(min_chunk_tokens)

    total = 0
    per_doc: list[list[int]] = []
    for ids in ids_list:
        if not ids:
            per_doc.append([])
            continue
        per_doc.append(ids)
        total += int(len(ids)) + _count_segments(
            token_count=len(ids),
            max_doc_tokens=max_doc_tokens,
            min_chunk_tokens=min_chunk_tokens,
        )
    if total <= 0:
        return np.empty((0,), dtype=dtype)

    flat = np.empty((total,), dtype=dtype)
    pos = 0
    for ids in per_doc:
        if not ids:
            continue
        pos = _write_doc_chunks(
            flat=flat,
            pos=pos,
            ids=ids,
            eos_id=int(eos_id),
            max_doc_tokens=max_doc_tokens,
            min_chunk_tokens=min_chunk_tokens,
        )
    if pos != total:
        flat = flat[:pos]
    return flat


_WORKER_TOK = None
_WORKER_EOS = None
_WORKER_DTYPE = None
_WORKER_MAX_DOC_TOKENS = 0
_WORKER_MIN_CHUNK_TOKENS = 1


def _worker_init(
    tokenizer_path: str,
    eos_id: int,
    dtype_str: str,
    max_doc_tokens: int,
    min_chunk_tokens: int,
) -> None:
    global _WORKER_TOK, _WORKER_EOS, _WORKER_DTYPE
    global _WORKER_MAX_DOC_TOKENS, _WORKER_MIN_CHUNK_TOKENS
    _WORKER_TOK = _load_tokenizer(tokenizer_path)
    _WORKER_EOS = int(eos_id)
    _WORKER_DTYPE = np.dtype(dtype_str)
    _WORKER_MAX_DOC_TOKENS = int(max_doc_tokens)
    _WORKER_MIN_CHUNK_TOKENS = int(min_chunk_tokens)


def _worker_encode(texts: list[str]) -> np.ndarray:
    if _WORKER_TOK is None:
        raise RuntimeError("Worker tokenizer not initialized")
    return _encode_batch_flat(
        _WORKER_TOK,
        texts,
        eos_id=int(_WORKER_EOS),
        dtype=np.dtype(_WORKER_DTYPE),
        max_doc_tokens=int(_WORKER_MAX_DOC_TOKENS),
        min_chunk_tokens=int(_WORKER_MIN_CHUNK_TOKENS),
    )


def _worker_encode_with_count(texts: list[str]) -> tuple[np.ndarray, int]:
    return _worker_encode(texts), int(len(texts))


def _limited_batches(
    src: Iterator[list[str]],
    *,
    max_docs: int,
) -> Iterator[list[str]]:
    limit = max(int(max_docs), 0)
    if limit <= 0:
        for texts in src:
            if texts:
                yield texts
        return

    emitted_docs = 0
    for texts in src:
        if not texts:
            continue
        if emitted_docs >= limit:
            break
        remaining = limit - emitted_docs
        if remaining <= 0:
            break
        chunk = list(texts[:remaining]) if len(texts) > remaining else texts
        if not chunk:
            continue
        emitted_docs += len(chunk)
        yield chunk


def encode_batches(
    *,
    batches: Iterator[list[str]],
    writer: ShardWriter,
    tokenizer: object,
    tokenizer_path: str,
    eos_id: int,
    dtype: np.dtype,
    max_doc_tokens: int,
    min_chunk_tokens: int,
    num_proc: int,
    max_docs: int,
    progress_every_docs: int,
) -> tuple[int, int]:
    total_tokens = 0
    total_docs = 0
    last_log_docs = 0
    t0 = time.time()
    progress_every = max(int(progress_every_docs), 0)

    batches = _limited_batches(batches, max_docs=max_docs)

    def maybe_log_progress() -> None:
        nonlocal last_log_docs
        if progress_every <= 0:
            return
        if total_docs - last_log_docs < progress_every:
            return
        last_log_docs = total_docs
        elapsed = max(1e-6, time.time() - t0)
        toks_per_s = int(total_tokens / elapsed)
        docs_per_s = int(total_docs / elapsed)
        print(
            f"[TOK] docs={total_docs} tokens={total_tokens} "
            f"rate={docs_per_s} docs/s {toks_per_s} tok/s shards={len(writer.shards)}",
            flush=True,
        )

    num_proc = max(int(num_proc), 1)
    if num_proc > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn") if sys.platform.startswith("win") else mp.get_context()
        with ctx.Pool(
            processes=num_proc,
            initializer=_worker_init,
            initargs=(
                str(tokenizer_path),
                int(eos_id),
                str(dtype.str),
                int(max_doc_tokens),
                int(min_chunk_tokens),
            ),
        ) as pool:
            for arr, n_docs in pool.imap(_worker_encode_with_count, batches, chunksize=1):
                if arr.size <= 0:
                    continue
                writer.write(arr)
                total_tokens += int(arr.size)
                total_docs += int(n_docs)
                maybe_log_progress()
                if max_docs > 0 and total_docs >= max_docs:
                    break
        return int(total_tokens), int(total_docs)

    for texts in batches:
        if not texts:
            continue
        arr = _encode_batch_flat(
            tokenizer,
            texts,
            eos_id=int(eos_id),
            dtype=dtype,
            max_doc_tokens=int(max_doc_tokens),
            min_chunk_tokens=int(min_chunk_tokens),
        )
        if arr.size <= 0:
            continue
        writer.write(arr)
        total_tokens += int(arr.size)
        total_docs += len(texts)
        maybe_log_progress()
        if max_docs > 0 and total_docs >= max_docs:
            break
    return int(total_tokens), int(total_docs)


__all__ = [
    "_encode_batch_flat",
    "_worker_encode",
    "_worker_encode_with_count",
    "_worker_init",
    "encode_batches",
]
