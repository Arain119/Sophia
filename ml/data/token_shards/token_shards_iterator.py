from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from .token_shards_support import torch_from_numpy_readonly

if TYPE_CHECKING:
    from .token_shards_dataset import TokenStreamDataset


TOKEN_STREAM_STATE_VERSION = 2
TOKEN_STREAM_CURSOR_SEMANTICS = "next_unread_window_index"


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
    window_order: torch.Tensor
    window_order_pos: int
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
        order_raw = state.get("window_order")
        if torch.is_tensor(order_raw):
            window_order = order_raw.detach().to(device="cpu", dtype=torch.int32).clone()
        elif isinstance(order_raw, list):
            try:
                window_order = torch.tensor(order_raw, dtype=torch.int32)
            except (TypeError, ValueError) as exc:
                raise ValueError("token-stream `window_order` must contain integers") from exc
        else:
            raise ValueError("token-stream resume state is missing `window_order`")
        if window_order.ndim != 1:
            raise ValueError("token-stream `window_order` must be one-dimensional")
        rng_state = state.get("rng_state")
        if rng_state is None:
            raise ValueError("token-stream resume state is missing `rng_state`")
        return cls(
            version=version,
            cursor_semantics=cursor_semantics or TOKEN_STREAM_CURSOR_SEMANTICS,
            seq_len=int(state.get("seq_len", 0) or 0),
            batch_size=int(state.get("batch_size", 0) or 0),
            block_tokens=int(state.get("block_tokens", 0) or 0),
            worker_id=int(state.get("worker_id", 0) or 0),
            num_workers=int(state.get("num_workers", 1) or 1),
            epoch=int(state.get("epoch", 0) or 0),
            window_order=window_order,
            window_order_pos=int(state.get("window_order_pos", 0) or 0),
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
            "window_order": self.window_order.detach().to(device="cpu", dtype=torch.int32).clone(),
            "window_order_pos": int(self.window_order_pos),
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
        dtype: np.dtype,
        preload_bytes: int,
    ) -> None:
        self._shard_paths = [str(path) for path in shard_paths]
        self._dtype = dtype
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
        preload_shard_idx: int | None,
    ) -> np.memmap:
        shard_idx = int(shard_idx)
        if self._active_shard_idx != shard_idx or self._active_mm is None:
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

        if preload_shard_idx is None or int(preload_shard_idx) == int(shard_idx):
            self._close_preload()
        elif int(preload_shard_idx) != int(self._preload_shard_idx or -1):
            self._close_preload()
            self._preload_mm = self._preload_shard(int(preload_shard_idx))
            self._preload_shard_idx = int(preload_shard_idx)
        assert self._active_mm is not None
        return self._active_mm

    def close(self) -> None:
        self._close_active()
        self._close_preload()


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
        self._window_counts = [
            int(tokens) // int(self._block_tokens) for tokens in self._shard_tokens
        ]
        self._window_prefix = list(accumulate(self._window_counts))
        self._total_windows = int(self._window_prefix[-1]) if self._window_prefix else 0
        if self._total_windows <= 0:
            raise ValueError(
                "token-stream manifest cannot yield a complete sequence window: "
                f"manifest={dataset.manifest_path!r} block_tokens={self._block_tokens}"
            )
        self._window_order = torch.empty((0,), dtype=torch.int32)
        self._window_order_pos = 0
        self._consumed_batches = 0
        self._memmaps = ShardMemmapPool(
            shard_paths=self._shard_paths,
            dtype=self._dataset._dtype,
            preload_bytes=int(dataset.shard_preload_bytes),
        )

    def __iter__(self) -> TokenStreamIterator:
        return self

    def _start_new_epoch(self) -> None:
        order = np.arange(self._total_windows, dtype=np.int32)
        self._rng.shuffle(order)
        self._window_order = torch.from_numpy(order).clone()
        self._window_order_pos = 0
        self._epoch += 1

    def _window_to_shard_start(self, window_index: int) -> tuple[int, int]:
        index = int(window_index)
        if index < 0 or index >= int(self._total_windows):
            raise ValueError(
                f"token-stream window index is outside the manifest: {index}"
            )
        shard_idx = int(bisect_right(self._window_prefix, index))
        previous = 0 if shard_idx == 0 else int(self._window_prefix[shard_idx - 1])
        local_index = int(index) - int(previous)
        return shard_idx, int(local_index) * int(self._block_tokens)

    def _next_preload_shard(self) -> int | None:
        if not self._dataset.shard_preload:
            return None
        next_pos = int(self._window_order_pos)
        if next_pos >= int(self._window_order.numel()):
            return None
        shard_idx, _ = self._window_to_shard_start(
            int(self._window_order[next_pos].item())
        )
        return int(shard_idx)

    def _consume_next_window(self) -> tuple[int, int]:
        if (
            self._window_order.numel() == 0
            or int(self._window_order_pos) >= int(self._window_order.numel())
        ):
            self._start_new_epoch()
        window_index = int(self._window_order[int(self._window_order_pos)].item())
        self._window_order_pos += 1
        shard_idx, start = self._window_to_shard_start(window_index)
        self._consumed_batches += 1
        return int(shard_idx), int(start)

    def _open_shard_memmap(self, shard_idx: int) -> np.memmap:
        return self._memmaps.open_shard(
            int(shard_idx),
            preload_shard_idx=self._next_preload_shard(),
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
            window_order=self._window_order,
            window_order_pos=int(self._window_order_pos),
            consumed_batches=int(self._consumed_batches),
            rng_state=self._rng.bit_generator.state,
        ).to_payload()

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        resume_state = parse_resume_state(state)
        resume_state.validate_worker_topology(
            worker_id=self._worker_id,
            num_workers=self._num_workers,
        )
        if (
            int(resume_state.seq_len) != int(self._seq_len)
            or int(resume_state.batch_size) != int(self._batch_size)
            or int(resume_state.block_tokens) != int(self._block_tokens)
        ):
            raise ValueError(
                "token-stream resume requires the same sequence geometry used by "
                "the checkpoint"
            )
        window_order = resume_state.window_order
        if (
            window_order.numel() == 0
            and int(resume_state.epoch) == 0
            and int(resume_state.window_order_pos) == 0
            and int(resume_state.consumed_batches) == 0
        ):
            self._window_order = torch.empty((0,), dtype=torch.int32)
            self._window_order_pos = 0
            self._epoch = 0
            self._consumed_batches = 0
            self._rng.bit_generator.state = resume_state.rng_state
            self._memmaps.close()
            return
        if int(window_order.numel()) != int(self._total_windows):
            raise ValueError(
                "token-stream resume window inventory does not match the current "
                f"manifest: saved={int(window_order.numel())} current={self._total_windows}"
            )
        if int(window_order.numel()) > 0:
            sorted_order = torch.sort(window_order).values
            expected_order = torch.arange(self._total_windows, dtype=torch.int32)
            if not torch.equal(sorted_order, expected_order):
                raise ValueError("token-stream resume window order is not a permutation")
        window_order_pos = int(resume_state.window_order_pos)
        if window_order_pos < 0 or window_order_pos > int(window_order.numel()):
            raise ValueError(
                "token-stream resume window cursor is outside the saved permutation"
            )
        self._window_order = window_order.clone()
        self._window_order_pos = window_order_pos
        self._epoch = int(resume_state.epoch)
        self._consumed_batches = int(resume_state.consumed_batches)
        self._rng.bit_generator.state = resume_state.rng_state
        self._memmaps.close()

    def close(self) -> None:
        self._memmaps.close()
        self._window_order = torch.empty((0,), dtype=torch.int32)
        self._window_order_pos = 0


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
    "resolve_worker_shards",
    "validate_worker_window_capacity",
]
