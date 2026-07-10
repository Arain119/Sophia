from __future__ import annotations

from threading import RLock
from typing import Protocol, runtime_checkable

import torch

from ml.runtime.model.state import RuntimeCacheSnapshot


@runtime_checkable
class SupportsRuntimeStateRefreshControl(Protocol):
    def refresh_state_buffers(self) -> None: ...


@runtime_checkable
class SupportsRuntimeStateResetControl(Protocol):
    def reset_runtime_cache(self) -> None: ...


@runtime_checkable
class SupportsRuntimeRecipeControl(Protocol):
    def supports_loss_chunk_size(self) -> bool: ...

    def supports_checkpoint_excludes(self) -> bool: ...

    def runtime_recipe_knobs(self) -> tuple[int, int, int]: ...

    def apply_runtime_recipe_knobs(
        self,
        *,
        loss_chunk_size: int,
        gradient_checkpointing_exclude_first: int,
        gradient_checkpointing_exclude_last: int,
    ) -> tuple[int, int, int]: ...

    def ensure_runtime_max_seq_len(self, max_seq_len: int) -> None: ...

    def runtime_max_seq_len(self) -> int: ...

    def runtime_prefix_replay_len(self) -> int: ...

    def sync_runtime_batch_capacity(self, max_batch_size: int) -> int: ...


@runtime_checkable
class RuntimeHost(
    SupportsRuntimeStateRefreshControl,
    SupportsRuntimeStateResetControl,
    SupportsRuntimeRecipeControl,
    Protocol,
):
    def rebuild_runtime_buffers(self) -> None: ...

    def replay_with_cache(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        return_all_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

    def forward_with_last_hidden(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        return_all_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

    def cache_dump(
        self,
        device: str = "cpu",
        *,
        cache_pos: int | None = None,
        batch_size: int | None = None,
    ) -> RuntimeCacheSnapshot: ...

    def cache_load(self, cache_snapshot: RuntimeCacheSnapshot) -> None: ...

    def cache_dump_prefix(
        self,
        device: str = "cpu",
        *,
        prefix_len: int | None = None,
        batch_size: int | None = None,
    ) -> RuntimeCacheSnapshot: ...


@runtime_checkable
class RuntimeBackedModel(Protocol):
    runtime: RuntimeHost
    runtime_lock: RLock


@runtime_checkable
class Runtime(Protocol):
    @property
    def runtime_model(self) -> RuntimeBackedModel: ...

    @property
    def runtime_host(self) -> RuntimeHost: ...

    @property
    def runtime_lock(self) -> RLock: ...


__all__ = [
    "RuntimeBackedModel",
    "RuntimeHost",
    "Runtime",
    "SupportsRuntimeRecipeControl",
    "SupportsRuntimeStateRefreshControl",
    "SupportsRuntimeStateResetControl",
]
