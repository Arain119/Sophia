from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

import torch


class _RuntimeArgs(Protocol):
    max_batch_size: int
    max_seq_len: int
    use_cache: bool
    rope_head_dim: int
    rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int

    def ensure_runtime_batch_capacity(self, batch_size: int) -> int: ...
    def runtime_batch_capacity(self) -> int: ...
    def ensure_runtime_sequence_capacity(self, max_seq_len: int) -> int: ...
    def runtime_sequence_capacity(self) -> int: ...
    def runtime_rope_kwargs(
        self,
        *,
        max_seq_len: int | None = None,
    ) -> dict[str, int | float]: ...


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
    freqs_cis: torch.Tensor
    gradient_checkpointing_exclude_first: int
    gradient_checkpointing_exclude_last: int

    def runtime_batch_capacity(self) -> int: ...
    def runtime_sequence_capacity(self) -> int: ...
    def runtime_rope_kwargs(
        self,
        *,
        max_seq_len: int | None = None,
    ) -> dict[str, int | float]: ...
    def runtime_reserve_batch_capacity(self, batch_size: int) -> int: ...
    def runtime_reserve_sequence_capacity(self, max_seq_len: int) -> int: ...


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


def _recompute_freqs_cis(
    model: RuntimeControlModel,
    required: int,
    *,
    precompute_freqs_cis: Callable[..., torch.Tensor],
) -> None:
    """Rebuild ``model.freqs_cis`` for ``required`` positions, preserving device."""
    current_device = model.freqs_cis.device
    rope_kwargs = model.runtime_rope_kwargs(max_seq_len=required)
    model.freqs_cis = precompute_freqs_cis(
        int(rope_kwargs["rope_head_dim"]),
        int(rope_kwargs["max_seq_len"]),
        float(rope_kwargs["rope_theta"]),
        original_seq_len=int(rope_kwargs["original_seq_len"]),
        rope_factor=float(rope_kwargs["rope_factor"]),
        beta_fast=int(rope_kwargs["beta_fast"]),
        beta_slow=int(rope_kwargs["beta_slow"]),
    ).to(device=current_device)


def ensure_sequence_capacity(
    model: RuntimeControlModel,
    max_seq_len: int,
    *,
    precompute_freqs_cis: Callable[..., torch.Tensor],
) -> None:
    required = max(int(max_seq_len), 1)
    if required <= int(model.runtime_sequence_capacity()):
        return
    required = int(model.runtime_reserve_sequence_capacity(required))
    _recompute_freqs_cis(model, required, precompute_freqs_cis=precompute_freqs_cis)
    for attn in iter_attn_modules(model):
        attn.ensure_sequence_capacity(required)


def rebuild_runtime_buffers(
    model: RuntimeControlModel,
    *,
    precompute_freqs_cis: Callable[..., torch.Tensor],
) -> None:
    required = int(model.runtime_sequence_capacity())
    if required <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {required}")
    _recompute_freqs_cis(model, required, precompute_freqs_cis=precompute_freqs_cis)
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
