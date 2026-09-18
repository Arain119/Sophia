from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import torch


BatchScalar = bool | int | float | str | None
BatchValue = torch.Tensor | BatchScalar
DeviceBatch = dict[str, BatchValue]


@runtime_checkable
class ResumableBatchIterator(Protocol):
    def __iter__(self) -> ResumableBatchIterator: ...

    def __next__(self) -> DeviceBatch: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: dict[str, object]) -> None: ...


@runtime_checkable
class SupportsClose(Protocol):
    def close(self) -> None: ...


class MoveBatchToDeviceFn(Protocol):
    def __call__(
        self,
        batch: Mapping[str, BatchValue],
        *,
        device: torch.device,
    ) -> DeviceBatch: ...


def _capture_resumable_state(source: ResumableBatchIterator) -> dict[str, object]:
    state = source.state_dict()
    if not isinstance(state, dict):
        raise RuntimeError("batch iterator returned an invalid resumable state.")
    return dict(state)


def _move_tensor_to_device(value: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    target = torch.device(device)
    if value.device == target:
        return value
    tensor = value
    if target.type == "cuda" and tensor.device.type == "cpu":
        if not tensor.is_pinned():
            tensor = tensor.pin_memory()
        return tensor.to(device=target, non_blocking=True)
    return tensor.to(device=target, non_blocking=False)


def move_batch_to_device(
    batch: Mapping[str, BatchValue],
    *,
    device: torch.device,
) -> DeviceBatch:
    out: DeviceBatch = {}
    tensor_cache: dict[int, torch.Tensor] = {}
    for raw_key, raw_value in batch.items():
        key = str(raw_key)
        if torch.is_tensor(raw_value):
            cache_key = id(raw_value)
            moved = tensor_cache.get(cache_key)
            if moved is None:
                moved = _move_tensor_to_device(raw_value, device=device)
                tensor_cache[cache_key] = moved
            out[key] = moved
        else:
            out[key] = raw_value
    return out


class ResumableCudaPrefetcher:
    """One-batch CUDA prefetch with exact resumable semantics."""

    def __init__(
        self,
        loader: ResumableBatchIterator,
        *,
        device: torch.device,
        move_batch_to_device_fn: MoveBatchToDeviceFn | None = None,
    ) -> None:
        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError("ResumableCudaPrefetcher requires a CUDA device.")
        self._loader = loader
        self._device = dev
        self._move_batch_to_device = move_batch_to_device_fn or move_batch_to_device
        self._stream = torch.cuda.Stream(device=dev)
        self._it = iter(loader)
        self._initial_state = _capture_resumable_state(loader)
        self._resume_state: dict[str, object] | None = None
        self._next_resume_state: dict[str, object] | None = None
        self._next: DeviceBatch | None = None
        self._preload()

    def __iter__(self) -> ResumableCudaPrefetcher:
        return self

    def _preload(self) -> None:
        try:
            batch = next(self._it)
        except StopIteration:
            self._next = None
            self._next_resume_state = None
            return
        resume_state = _capture_resumable_state(self._loader)
        with torch.cuda.stream(self._stream):
            self._next = self._move_batch_to_device(batch, device=self._device)
        self._next_resume_state = resume_state

    def __next__(self) -> DeviceBatch:
        if self._next is None or self._next_resume_state is None:
            raise StopIteration
        torch.cuda.current_stream(self._device).wait_stream(self._stream)
        batch = self._next
        resume_state = dict(self._next_resume_state)
        self._preload()
        self._resume_state = resume_state
        return batch

    def state_dict(self) -> dict[str, object]:
        if self._resume_state is None:
            return dict(self._initial_state)
        return dict(self._resume_state)

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._loader.load_state_dict(dict(state))
        self._it = iter(self._loader)
        self._initial_state = dict(state)
        self._resume_state = None
        self._next_resume_state = None
        self._next = None
        self._preload()

    def close(self) -> None:
        if isinstance(self._loader, SupportsClose):
            self._loader.close()
        self._it = iter(())
        self._next = None
        self._next_resume_state = None
