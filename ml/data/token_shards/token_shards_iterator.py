from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from .token_shards_support import (
    shape_changed_resume_incompatibility,
    torch_from_numpy_readonly,
)

if TYPE_CHECKING:
    from .token_shards_dataset import TokenStreamDataset


TOKEN_STREAM_STATE_VERSION = 1
TOKEN_STREAM_CURSOR_SEMANTICS = "next_unread_token_offset"


@dataclass(frozen=True)
class TokenWindowBatchSpec:
    shard_idx: int
    shard_path: str
    dtype: np.dtype
    start: int
    seq_len: int
    batch_size: int
    block_tokens: int


@dataclass(frozen=True)
class TokenStreamResumeState:
    version: int
    cursor_semantics: str
    seq_len: int
    batch_size: int
    block_tokens: int
    worker_id: int
    num_workers: int
    epoch: int
    order: tuple[int, ...]
    order_pos: int
    current_pos: int | None
    consumed_batches: int
    rng_state: object

    @classmethod
    def from_payload(cls, state: Mapping[str, object]) -> TokenStreamResumeState:
        version = int(state.get("version", 0) or 0)
        if version != TOKEN_STREAM_STATE_VERSION:
            raise ValueError(f"unsupported token-stream resume state version: {version}")
        cursor_semantics = str(
            state.get("cursor_semantics", TOKEN_STREAM_CURSOR_SEMANTICS) or ""
        ).strip()
        if cursor_semantics and cursor_semantics != TOKEN_STREAM_CURSOR_SEMANTICS:
            raise ValueError(
                "unsupported token-stream cursor semantics: "
                f"{cursor_semantics!r}"
            )
        order_raw = state.get("order")
        if not isinstance(order_raw, list):
            raise ValueError("token-stream resume state is missing `order`")
        rng_state = state.get("rng_state")
        if rng_state is None:
            raise ValueError("token-stream resume state is missing `rng_state`")
        current_pos_raw = state.get("current_pos")
        current_pos = None if current_pos_raw is None else int(current_pos_raw)
        return cls(
            version=version,
            cursor_semantics=cursor_semantics or TOKEN_STREAM_CURSOR_SEMANTICS,
            seq_len=int(state.get("seq_len", 0) or 0),
            batch_size=int(state.get("batch_size", 0) or 0),
            block_tokens=int(state.get("block_tokens", 0) or 0),
            worker_id=int(state.get("worker_id", 0) or 0),
            num_workers=int(state.get("num_workers", 1) or 1),
            epoch=int(state.get("epoch", 0) or 0),
            order=tuple(int(x) for x in order_raw),
            order_pos=int(state.get("order_pos", 0) or 0),
            current_pos=current_pos,
            consumed_batches=int(state.get("consumed_batches", 0) or 0),
            rng_state=rng_state,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "version": int(self.version),
            "cursor_semantics": str(self.cursor_semantics),
            "seq_len": int(self.seq_len),
            "batch_size": int(self.batch_size),
            "block_tokens": int(self.block_tokens),
            "worker_id": int(self.worker_id),
            "num_workers": int(self.num_workers),
            "epoch": int(self.epoch),
            "order": [int(x) for x in self.order],
            "order_pos": int(self.order_pos),
            "current_pos": None if self.current_pos is None else int(self.current_pos),
            "consumed_batches": int(self.consumed_batches),
            "rng_state": self.rng_state,
        }

    def validate_worker_topology(
        self,
        *,
        worker_id: int,
        num_workers: int,
    ) -> None:
        if int(self.worker_id) != int(worker_id) or int(self.num_workers) != int(num_workers):
            raise ValueError(
                "token-stream resume state worker topology mismatch: "
                f"expected worker_id={int(worker_id)}, num_workers={int(num_workers)}; "
                f"got worker_id={int(self.worker_id)}, num_workers={int(self.num_workers)}"
            )


def resolve_worker_shards(
    *,
    manifest_dir: str,
    shards: tuple[object, ...] | list[object],
    worker_id: int,
    num_workers: int,
) -> tuple[list[str], list[int]]:
    if int(worker_id) < len(shards):
        shard_indices = list(range(int(worker_id), len(shards), max(int(num_workers), 1)))
    else:
        shard_indices = []
    if not shard_indices:
        shard_indices = [int(worker_id) % len(shards)]

    shard_paths = [
        os.path.join(str(manifest_dir), shards[i].path) for i in shard_indices
    ]
    shard_tokens = [int(shards[i].tokens) for i in shard_indices]
    return shard_paths, shard_tokens


def validate_worker_window_capacity(
    *,
    manifest_path: str,
    worker_id: int,
    num_workers: int,
    block_tokens: int,
    shard_tokens: list[int],
) -> None:
    max_shard_tokens = max(shard_tokens, default=0)
    if int(max_shard_tokens) < int(block_tokens):
        raise ValueError(
            "Token shard worker assignment cannot yield a window: "
            f"manifest={manifest_path!r} worker_id={int(worker_id)} "
            f"num_workers={int(num_workers)} block_tokens={int(block_tokens)} "
            f"max_assigned_shard_tokens={int(max_shard_tokens)}"
        )


def materialize_window_batch(
    *,
    memmap: np.memmap,
    start: int,
    seq_len: int,
    batch_size: int,
    block_tokens: int,
) -> dict[str, torch.Tensor]:
    if int(batch_size) <= 0:
        chunk = np.array(memmap[int(start) : int(start) + int(seq_len)], copy=True)
        tensor = torch_from_numpy_readonly(chunk)
        return {
            "input_ids": tensor,
            "labels": tensor,
            "labels_is_input_ids": True,
        }

    flat = np.array(
        memmap[int(start) : int(start) + int(block_tokens)],
        copy=True,
    )
    arr = flat.reshape(int(batch_size), int(seq_len))
    tensor = torch_from_numpy_readonly(arr)
    return {
        "input_ids": tensor,
        "labels": tensor,
        "labels_is_input_ids": True,
    }


def materialize_window_batch_spec(
    spec: TokenWindowBatchSpec,
) -> dict[str, torch.Tensor]:
    memmap = np.memmap(
        str(spec.shard_path),
        mode="r",
        dtype=spec.dtype,
    )
    return materialize_window_batch(
        memmap=memmap,
        start=int(spec.start),
        seq_len=int(spec.seq_len),
        batch_size=int(spec.batch_size),
        block_tokens=int(spec.block_tokens),
    )


class ShardMemmapPool:
    def __init__(
        self,
        *,
        shard_paths: Sequence[str],
        shard_tokens: Sequence[int],
        block_tokens: int,
        dtype: np.dtype,
        preload_enabled: bool,
        preload_bytes: int,
    ) -> None:
        self._shard_paths = [str(path) for path in shard_paths]
        self._shard_tokens = [int(tokens) for tokens in shard_tokens]
        self._block_tokens = int(block_tokens)
        self._dtype = dtype
        self._preload_enabled = bool(preload_enabled)
        self._preload_bytes = int(preload_bytes)
        self._active_shard_idx: int | None = None
        self._active_mm: np.memmap | None = None
        self._preload_shard_idx: int | None = None
        self._preload_mm: np.memmap | None = None

    def _close_memmap(self, mm: np.memmap | None) -> None:
        handle = getattr(mm, "_mmap", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def _close_active(self) -> None:
        mm = self._active_mm
        self._active_mm = None
        self._active_shard_idx = None
        self._close_memmap(mm)

    def _close_preload(self) -> None:
        mm = self._preload_mm
        self._preload_mm = None
        self._preload_shard_idx = None
        self._close_memmap(mm)

    def _next_preload_candidate(
        self,
        *,
        order: Sequence[int],
        order_pos: int,
    ) -> int | None:
        if not self._preload_enabled or not order:
            return None
        for pos in range(int(order_pos) + 1, len(order)):
            shard_idx = int(order[pos])
            if int(self._shard_tokens[shard_idx]) >= int(self._block_tokens):
                return shard_idx
        return None

    def _preload_shard(self, shard_idx: int) -> np.memmap:
        path = self._shard_paths[int(shard_idx)]
        if self._preload_bytes > 0:
            with open(path, "rb") as handle:
                _ = handle.read(int(self._preload_bytes))
        return np.memmap(path, mode="r", dtype=self._dtype)

    def open_shard(
        self,
        shard_idx: int,
        *,
        order: Sequence[int],
        order_pos: int,
    ) -> np.memmap:
        shard_idx = int(shard_idx)
        if self._active_shard_idx == shard_idx and self._active_mm is not None:
            return self._active_mm

        self._close_active()
        if self._preload_shard_idx == shard_idx and self._preload_mm is not None:
            self._active_shard_idx = shard_idx
            self._active_mm = self._preload_mm
            self._preload_mm = None
            self._preload_shard_idx = None
        else:
            self._close_preload()
            self._active_mm = self._preload_shard(int(shard_idx))
            self._active_shard_idx = shard_idx

        preload_idx = self._next_preload_candidate(
            order=order,
            order_pos=order_pos,
        )
        if preload_idx is None:
            self._close_preload()
        elif int(preload_idx) != int(self._preload_shard_idx or -1):
            self._close_preload()
            self._preload_mm = self._preload_shard(int(preload_idx))
            self._preload_shard_idx = int(preload_idx)
        return self._active_mm

    def close(self) -> None:
        self._close_active()
        self._close_preload()


def resolve_saved_shape(
    *,
    resume_state: TokenStreamResumeState,
    current_seq_len: int,
    current_batch_size: int,
    current_block_tokens: int,
) -> tuple[int, int, int]:
    saved_seq_len = (
        int(current_seq_len)
        if int(resume_state.seq_len) <= 0
        else int(resume_state.seq_len)
    )
    saved_batch_size = (
        int(current_batch_size)
        if int(resume_state.batch_size) <= 0
        else int(resume_state.batch_size)
    )
    saved_block_tokens = (
        int(current_block_tokens)
        if int(resume_state.block_tokens) <= 0
        else int(resume_state.block_tokens)
    )
    return int(saved_seq_len), int(saved_batch_size), int(saved_block_tokens)


def validate_shape_changed_resume_state(
    *,
    shard_tokens: list[int],
    order: list[int],
    order_pos: int,
    current_pos: int | None,
    current_block_tokens: int,
    current_seq_len: int,
    current_batch_size: int,
    saved_seq_len: int,
    saved_batch_size: int,
    saved_block_tokens: int,
) -> None:
    if (
        int(saved_seq_len) == int(current_seq_len)
        and int(saved_batch_size) == int(current_batch_size)
        and int(saved_block_tokens) == int(current_block_tokens)
    ):
        return
    incompatibility = shape_changed_resume_incompatibility(
        shard_tokens=shard_tokens,
        order=order,
        order_pos=int(order_pos),
        current_pos=None if current_pos is None else int(current_pos),
        block_tokens=int(current_block_tokens),
    )
    if incompatibility is not None:
        raise ValueError(incompatibility)


def parse_resume_state(state: Mapping[str, object]) -> TokenStreamResumeState:
    if not isinstance(state, Mapping):
        raise TypeError("token-stream resume state must be a dict")
    return TokenStreamResumeState.from_payload(state)


class TokenStreamIterator:
    def __init__(
        self,
        dataset: TokenStreamDataset,
        *,
        worker_id: int,
        num_workers: int,
    ) -> None:
        self._dataset = dataset
        self._worker_id = int(worker_id)
        self._num_workers = max(int(num_workers), 1)
        self._seq_len = int(dataset.seq_len)
        self._batch_size = int(dataset.batch_size)
        self._block_tokens = int(dataset._block_tokens)
        self._rng = np.random.default_rng(dataset.seed + 97_973 * int(worker_id))
        self._shard_paths, self._shard_tokens = resolve_worker_shards(
            manifest_dir=dataset._manifest_dir,
            shards=dataset._shards,
            worker_id=int(self._worker_id),
            num_workers=int(self._num_workers),
        )
        validate_worker_window_capacity(
            manifest_path=str(dataset.manifest_path),
            worker_id=int(self._worker_id),
            num_workers=int(self._num_workers),
            block_tokens=int(self._block_tokens),
            shard_tokens=self._shard_tokens,
        )
        self._epoch = 0
        self._order: list[int] = []
        self._order_pos = 0
        self._current_pos: int | None = None
        self._consumed_batches = 0
        self._memmaps = ShardMemmapPool(
            shard_paths=self._shard_paths,
            shard_tokens=self._shard_tokens,
            block_tokens=self._block_tokens,
            dtype=self._dataset._dtype,
            preload_enabled=bool(dataset.shard_preload),
            preload_bytes=int(dataset.shard_preload_bytes),
        )

    def __iter__(self) -> TokenStreamIterator:
        return self

    def _start_new_epoch(self) -> None:
        self._order = list(range(len(self._shard_paths)))
        self._rng.shuffle(self._order)
        self._order_pos = 0
        self._current_pos = None
        self._epoch += 1

    def _ensure_yieldable_window(self) -> tuple[int, int]:
        while True:
            if not self._order or self._order_pos >= len(self._order):
                self._start_new_epoch()
            shard_idx = int(self._order[self._order_pos])
            n_tokens = int(self._shard_tokens[shard_idx])
            if n_tokens < int(self._block_tokens):
                self._order_pos += 1
                self._current_pos = None
                continue
            max_start = int(n_tokens) - int(self._block_tokens)
            if self._current_pos is None:
                start_cap = (
                    min(int(self._seq_len), int(max_start) + 1) if max_start > 0 else 1
                )
                self._current_pos = int(self._rng.integers(0, int(start_cap)))
            if int(self._current_pos) > int(max_start):
                self._order_pos += 1
                self._current_pos = None
                continue
            return shard_idx, int(self._current_pos)

    def _consume_next_window(self) -> tuple[int, int]:
        shard_idx, start = self._ensure_yieldable_window()
        n_tokens = int(self._shard_tokens[shard_idx])
        max_start = int(n_tokens) - int(self._block_tokens)
        next_pos = int(start) + int(self._block_tokens)
        if int(next_pos) <= int(max_start):
            self._current_pos = int(next_pos)
        else:
            self._order_pos += 1
            self._current_pos = None
        self._consumed_batches += 1
        return int(shard_idx), int(start)

    def _open_shard_memmap(self, shard_idx: int) -> np.memmap:
        return self._memmaps.open_shard(
            int(shard_idx),
            order=self._order,
            order_pos=self._order_pos,
        )

    def next_batch_spec(self) -> TokenWindowBatchSpec:
        shard_idx, start = self._consume_next_window()
        return TokenWindowBatchSpec(
            shard_idx=int(shard_idx),
            shard_path=str(self._shard_paths[int(shard_idx)]),
            dtype=self._dataset._dtype,
            start=int(start),
            seq_len=int(self._seq_len),
            batch_size=int(self._batch_size),
            block_tokens=int(self._block_tokens),
        )

    def __next__(self) -> dict[str, torch.Tensor]:
        spec = self.next_batch_spec()
        mm = self._open_shard_memmap(int(spec.shard_idx))
        return materialize_window_batch(
            memmap=mm,
            start=int(spec.start),
            seq_len=int(spec.seq_len),
            batch_size=int(spec.batch_size),
            block_tokens=int(spec.block_tokens),
        )

    def state_dict(self) -> dict[str, object]:
        return TokenStreamResumeState(
            version=TOKEN_STREAM_STATE_VERSION,
            cursor_semantics=TOKEN_STREAM_CURSOR_SEMANTICS,
            seq_len=int(self._seq_len),
            batch_size=int(self._batch_size),
            block_tokens=int(self._block_tokens),
            worker_id=int(self._worker_id),
            num_workers=int(self._num_workers),
            epoch=int(self._epoch),
            order=tuple(int(x) for x in self._order),
            order_pos=int(self._order_pos),
            current_pos=(
                None if self._current_pos is None else int(self._current_pos)
            ),
            consumed_batches=int(self._consumed_batches),
            rng_state=self._rng.bit_generator.state,
        ).to_payload()

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        resume_state = parse_resume_state(state)
        resume_state.validate_worker_topology(
            worker_id=self._worker_id,
            num_workers=self._num_workers,
        )
        self._order = [int(x) for x in resume_state.order]
        self._order_pos = int(resume_state.order_pos)
        self._current_pos = (
            None if resume_state.current_pos is None else int(resume_state.current_pos)
        )
        self._epoch = int(resume_state.epoch)
        self._consumed_batches = int(resume_state.consumed_batches)
        saved_seq_len, saved_batch_size, saved_block_tokens = resolve_saved_shape(
            resume_state=resume_state,
            current_seq_len=int(self._seq_len),
            current_batch_size=int(self._batch_size),
            current_block_tokens=int(self._block_tokens),
        )
        self._rng.bit_generator.state = resume_state.rng_state
        validate_shape_changed_resume_state(
            shard_tokens=self._shard_tokens,
            order=self._order,
            order_pos=int(self._order_pos),
            current_pos=self._current_pos,
            current_block_tokens=int(self._block_tokens),
            current_seq_len=int(self._seq_len),
            current_batch_size=int(self._batch_size),
            saved_seq_len=int(saved_seq_len),
            saved_batch_size=int(saved_batch_size),
            saved_block_tokens=int(saved_block_tokens),
        )
        self._memmaps.close()

    def close(self) -> None:
        self._memmaps.close()
        self._order = []
        self._current_pos = None


__all__ = [
    "TOKEN_STREAM_CURSOR_SEMANTICS",
    "TOKEN_STREAM_STATE_VERSION",
    "ShardMemmapPool",
    "TokenStreamIterator",
    "TokenStreamResumeState",
    "TokenWindowBatchSpec",
    "materialize_window_batch",
    "materialize_window_batch_spec",
    "parse_resume_state",
    "resolve_saved_shape",
    "resolve_worker_shards",
    "validate_shape_changed_resume_state",
    "validate_worker_window_capacity",
]
