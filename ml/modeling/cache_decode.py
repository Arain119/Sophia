from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ml.runtime.model.state import RuntimeCacheSnapshot


@dataclass
class RuntimeCacheState:
    cache: RuntimeCacheSnapshot
    batch_size: int
    cache_pos: int


class RuntimeCacheDecodeModel(Protocol):
    def replay_with_cache(
        self,
        input_ids: torch.Tensor,
        *,
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

    def reset_runtime_cache(self) -> None: ...

def clone_runtime_cache_state(cache_state: RuntimeCacheState) -> RuntimeCacheState:
    return RuntimeCacheState(
        cache=cache_state.cache.clone(),
        batch_size=int(cache_state.batch_size),
        cache_pos=int(cache_state.cache_pos),
    )


def cache_state_from_runtime_cache(
    cache_input: RuntimeCacheState | None,
) -> RuntimeCacheState | None:
    if cache_input is None:
        return None
    if isinstance(cache_input, RuntimeCacheState):
        return clone_runtime_cache_state(cache_input)
    raise TypeError("cache must be a Sophia RuntimeCacheState returned by SophiaDecoder")


def prefill_runtime_cache(
    model: RuntimeCacheDecodeModel,
    input_ids: torch.Tensor,
    *,
    start_pos: int,
    logits_to_keep: int | None = None,
) -> torch.Tensor:
    if int(input_ids.size(1)) <= 0:
        raise ValueError("input_ids must contain at least one token")
    return_all_logits = int(logits_to_keep or 0) != 1
    logits, _ = model.replay_with_cache(
        input_ids,
        start_pos=int(start_pos),
        return_all_logits=bool(return_all_logits),
    )
    if not bool(return_all_logits):
        logits = logits.unsqueeze(1)
    if logits_to_keep is not None and int(logits_to_keep) > 0:
        return logits[:, -int(logits_to_keep) :, :]
    return logits


def forward_cached_decode(
    *,
    model: RuntimeCacheDecodeModel,
    input_ids: torch.Tensor,
    cache_state: RuntimeCacheState | None,
    start_pos: int | None,
    logits_to_keep: int | None,
) -> tuple[torch.Tensor, RuntimeCacheState]:
    if logits_to_keep is not None and int(logits_to_keep) < 0:
        raise ValueError(f"logits_to_keep must be >= 0 when provided, got {logits_to_keep}")
    cache_start = 0 if start_pos is None else int(start_pos)
    if cache_state is None:
        if cache_start != 0:
            raise ValueError("cache_state is required when start_pos > 0 for cached decode")
        model.reset_runtime_cache()
        logits = prefill_runtime_cache(
            model,
            input_ids,
            start_pos=cache_start,
            logits_to_keep=logits_to_keep,
        )
        next_cache_pos = int(cache_start) + int(input_ids.size(1))
        return logits, RuntimeCacheState(
            cache=model.cache_dump(
                device="cpu",
                cache_pos=int(next_cache_pos),
                batch_size=int(input_ids.size(0)),
            ),
            cache_pos=int(next_cache_pos),
            batch_size=int(input_ids.size(0)),
        )

    cache_state = clone_runtime_cache_state(cache_state)
    cache_pos = int(cache_state.cache_pos)
    if cache_pos < 0:
        raise ValueError(f"cache cache_pos must be >= 0, got {cache_pos}")
    if start_pos is not None and int(start_pos) != cache_pos:
        raise ValueError(
            f"start_pos ({start_pos}) must match cache cache_pos ({cache_pos})"
        )
    batch_size = int(cache_state.batch_size or int(input_ids.size(0)))
    if batch_size <= 0:
        raise ValueError(f"cache batch_size must be > 0, got {batch_size}")
    if batch_size != int(input_ids.size(0)):
        raise ValueError(
            "cache batch_size does not match input_ids: "
            f"{batch_size} != {int(input_ids.size(0))}"
        )
    model.cache_load(cache_state.cache)
    logits = prefill_runtime_cache(
        model,
        input_ids,
        start_pos=cache_pos,
        logits_to_keep=logits_to_keep,
    )
    next_cache_pos = int(cache_pos) + int(input_ids.size(1))
    return logits, RuntimeCacheState(
        cache=model.cache_dump(
            device="cpu",
            cache_pos=int(next_cache_pos),
            batch_size=int(input_ids.size(0)),
        ),
        cache_pos=int(next_cache_pos),
        batch_size=int(input_ids.size(0)),
    )


__all__ = [
    "RuntimeCacheDecodeModel",
    "RuntimeCacheState",
    "cache_state_from_runtime_cache",
    "clone_runtime_cache_state",
    "forward_cached_decode",
    "prefill_runtime_cache",
]
