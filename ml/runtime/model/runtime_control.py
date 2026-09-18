from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol


class _RuntimeArgs(Protocol):
    use_cache: bool
    def ensure_runtime_batch_capacity(self, batch_size: int) -> int: ...
    def ensure_runtime_sequence_capacity(self, max_seq_len: int) -> int: ...


class _AttentionControl(Protocol):
    use_cache: bool
    def ensure_batch_capacity(self, batch_size: int) -> None: ...
    def ensure_sequence_capacity(self, max_seq_len: int) -> None: ...
    def rebuild_runtime_buffers(self, max_seq_len: int) -> None: ...
    def reset(self) -> None: ...
    def refresh_state_buffers(self) -> None: ...


class _RuntimeBlock(Protocol):
    attn: _AttentionControl


class RuntimeControlModel(Protocol):
    args: _RuntimeArgs
    layers: Sequence[_RuntimeBlock]
    def runtime_reserve_batch_capacity(self, batch_size: int) -> int: ...
    def runtime_reserve_sequence_capacity(self, max_seq_len: int) -> int: ...
    def runtime_sequence_capacity(self) -> int: ...


def iter_attn_modules(model: RuntimeControlModel) -> Iterable[_AttentionControl]:
    for layer in model.layers:
        yield layer.attn


def enable_runtime_cache(model: RuntimeControlModel) -> None:
    model.args.use_cache = True
    for attn in iter_attn_modules(model):
        attn.use_cache = True


def ensure_batch_capacity(model: RuntimeControlModel, batch_size: int) -> None:
    required = int(model.runtime_reserve_batch_capacity(batch_size))
    for attn in iter_attn_modules(model):
        attn.ensure_batch_capacity(required)


def ensure_sequence_capacity(model: RuntimeControlModel, max_seq_len: int) -> None:
    required = int(max_seq_len)
    if required <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {required}")
    if required <= int(model.runtime_sequence_capacity()):
        return
    required = int(model.runtime_reserve_sequence_capacity(required))
    for attn in iter_attn_modules(model):
        attn.ensure_sequence_capacity(required)


def rebuild_runtime_buffers(model: RuntimeControlModel) -> None:
    required = int(model.runtime_sequence_capacity())
    for attn in iter_attn_modules(model):
        attn.rebuild_runtime_buffers(required)


def reset_runtime_cache(model: RuntimeControlModel) -> None:
    for attn in iter_attn_modules(model):
        attn.reset()


def refresh_state_buffers(model: RuntimeControlModel) -> None:
    for attn in iter_attn_modules(model):
        attn.refresh_state_buffers()


__all__ = [
    "RuntimeControlModel",
    "enable_runtime_cache",
    "ensure_batch_capacity",
    "ensure_sequence_capacity",
    "rebuild_runtime_buffers",
    "refresh_state_buffers",
    "reset_runtime_cache",
]
