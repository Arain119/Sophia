from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from ml.data.token_shards import TokenStreamDataset
from ml.data.token_shards.token_shards_iterator import (
    materialize_window_batch_spec,
)
from ml.training.pretrain.resources import (
    PretrainBatch,
    PretrainDataIter,
)
from ml.core.common.device_batch import ResumableCudaPrefetcher
from ml.training.pretrain.train_state import DataIterResumeState


def _build_token_stream_iter(
    dataset: TokenStreamDataset,
    *,
    resume_state: DataIterResumeState | Mapping[str, object] | None = None,
):
    token_iter = iter(dataset)
    if resume_state is None:
        return token_iter
    load_state = getattr(token_iter, "load_state_dict", None)
    if not callable(load_state):
        raise RuntimeError("token iterator does not support resume state restoration.")
    load_state(
        resume_state.to_payload()
        if isinstance(resume_state, DataIterResumeState)
        else dict(resume_state)
    )
    return token_iter


def _capture_iter_state(token_iter) -> dict[str, object]:
    state_fn = getattr(token_iter, "state_dict", None)
    if not callable(state_fn):
        raise RuntimeError("token iterator does not expose a resumable state_dict().")
    state = state_fn()
    if not isinstance(state, dict):
        raise RuntimeError("token iterator returned an invalid resume state.")
    return dict(state)


def _close_token_iter(token_iter) -> None:
    close_fn = getattr(token_iter, "close", None)
    if callable(close_fn):
        close_fn()


class _BatchNormalizationMixin:
    @staticmethod
    def _cast_token_ids(x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.int32:
            return x
        if x.dtype in (
            torch.uint16,
            torch.uint32,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int64,
        ):
            return x.to(dtype=torch.int32)
        raise RuntimeError(f"Unsupported token-id dtype: {x.dtype}")

    def _normalize_batch(self, batch: dict[str, torch.Tensor]) -> PretrainBatch:
        out: dict[str, torch.Tensor | bool] = {}
        ids = batch["input_ids"]
        labels = batch.get("labels")
        labels_is_ids = bool(batch.get("labels_is_input_ids", False))
        ids = self._cast_token_ids(ids)
        out["input_ids"] = ids
        if labels is not None:
            if labels_is_ids or (labels is batch["input_ids"]):
                out["labels"] = ids
            else:
                labels = self._cast_token_ids(labels)
                out["labels"] = labels
        if "labels_is_input_ids" in batch:
            out["labels_is_input_ids"] = bool(batch["labels_is_input_ids"])
        return out  # type: ignore[return-value]


class _SingleProcessBatchIter(_BatchNormalizationMixin):
    def __init__(
        self,
        dataset: TokenStreamDataset,
        *,
        device: torch.device,
        resume_state: DataIterResumeState | Mapping[str, object] | None = None,
    ) -> None:
        del device
        self._it = _build_token_stream_iter(
            dataset,
            resume_state=resume_state,
        )

    def __iter__(self) -> _SingleProcessBatchIter:
        return self

    def __next__(self) -> PretrainBatch:
        return self._normalize_batch(next(self._it))

    def state_dict(self) -> dict[str, object]:
        return _capture_iter_state(self._it)

    def load_state_dict(
        self,
        state: DataIterResumeState | Mapping[str, object],
    ) -> None:
        load_state = getattr(self._it, "load_state_dict", None)
        if not callable(load_state):
            raise RuntimeError("Single-process token iterator does not support resume state restoration.")
        load_state(
            state.to_payload() if isinstance(state, DataIterResumeState) else dict(state)
        )

    def close(self) -> None:
        _close_token_iter(self._it)
        self._it = iter(())

@dataclass(frozen=True)
class _QueuedPrefetchBatch:
    resume_state: dict[str, object]
    future: Future[dict[str, torch.Tensor]]


class _ParallelBatchIter(_BatchNormalizationMixin):
    def __init__(
        self,
        dataset: TokenStreamDataset,
        *,
        device: torch.device,
        num_workers: int,
        prefetch_factor: int,
        dataloader_persistent_workers: int | None = None,
        resume_state: DataIterResumeState | Mapping[str, object] | None = None,
    ) -> None:
        del device, dataloader_persistent_workers
        self._dataset = dataset
        self._num_workers = max(int(num_workers), 1)
        self._prefetch_factor = max(int(prefetch_factor), 0)
        self._queue_target = max(
            int(self._num_workers) * max(int(self._prefetch_factor), 1),
            1,
        )
        self._it = None
        self._executor: ThreadPoolExecutor | None = None
        self._pending: deque[_QueuedPrefetchBatch] = deque()
        self._rebuild_stream(resume_state=resume_state)

    def __iter__(self) -> _ParallelBatchIter:
        return self

    def _rebuild_stream(
        self,
        *,
        resume_state: DataIterResumeState | Mapping[str, object] | None,
    ) -> None:
        self._shutdown(wait=True)
        self._it = _build_token_stream_iter(
            self._dataset,
            resume_state=resume_state,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=int(self._num_workers),
            thread_name_prefix="sophia_pretrain_data",
        )
        self._pending.clear()
        self._fill_prefetch()

    def _shutdown(self, *, wait: bool) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=bool(wait), cancel_futures=True)
        token_iter = self._it
        self._it = None
        if token_iter is not None:
            _close_token_iter(token_iter)
        self._pending.clear()

    def _schedule_next(self) -> bool:
        token_iter = self._it
        executor = self._executor
        if token_iter is None or executor is None:
            return False
        try:
            resume_state = _capture_iter_state(token_iter)
            spec = token_iter.next_batch_spec()
        except StopIteration:
            return False
        future = executor.submit(materialize_window_batch_spec, spec)
        self._pending.append(
            _QueuedPrefetchBatch(
                resume_state=resume_state,
                future=future,
            )
        )
        return True

    def _fill_prefetch(self) -> None:
        while len(self._pending) < int(self._queue_target):
            if not self._schedule_next():
                return

    def __next__(self) -> PretrainBatch:
        if not self._pending and not self._schedule_next():
            raise StopIteration
        queued = self._pending.popleft()
        batch = queued.future.result()
        self._fill_prefetch()
        return self._normalize_batch(batch)

    def state_dict(self) -> dict[str, object]:
        if self._pending:
            return dict(self._pending[0].resume_state)
        token_iter = self._it
        if token_iter is None:
            raise RuntimeError("Parallel token iterator is closed.")
        return _capture_iter_state(token_iter)

    def load_state_dict(
        self,
        state: DataIterResumeState | Mapping[str, object],
    ) -> None:
        self._rebuild_stream(resume_state=state)

    def close(self) -> None:
        self._shutdown(wait=True)


def build_pretrain_data_iter(
    *,
    manifest_path: str,
    seq_len: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    num_workers: int,
    dataloader_prefetch_factor: int,
    dataloader_persistent_workers: int | None = None,
    shard_preload: int = 0,
    shard_preload_bytes: int = 0,
    resume_state: DataIterResumeState | Mapping[str, object] | None = None,
) -> PretrainDataIter:
    """
    Build the exact-resumable token-shard iterator for pretraining.
    """
    sp = int(shard_preload)
    if sp < 0:
        raise ValueError("shard_preload must be >= 0")
    spb = int(shard_preload_bytes)
    if spb < 0:
        raise ValueError("shard_preload_bytes must be >= 0")
    if int(sp) == 0:
        spb = 0

    train_ds = TokenStreamDataset(
        manifest_path,
        seq_len=int(seq_len),
        seed=int(seed),
        batch_size=int(batch_size),
        shard_preload=int(sp),
        shard_preload_bytes=int(spb),
    )

    if int(num_workers) <= 0:
        base_iter: PretrainDataIter = _SingleProcessBatchIter(
            train_ds,
            device=device,
            resume_state=resume_state,
        )
    else:
        base_iter = _ParallelBatchIter(
            train_ds,
            device=device,
            num_workers=int(num_workers),
            prefetch_factor=int(dataloader_prefetch_factor),
            dataloader_persistent_workers=dataloader_persistent_workers,
            resume_state=resume_state,
        )
    if torch.device(device).type == "cuda":
        return ResumableCudaPrefetcher(base_iter, device=device)
    return base_iter
