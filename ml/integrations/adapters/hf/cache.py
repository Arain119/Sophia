"""HF cache adapter owned by the Sophia HF adapter layer."""

from __future__ import annotations

import weakref

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from ml.modeling.cache_decode import RuntimeCacheState
from ml.runtime.model.state import AttentionCacheSnapshot, RuntimeCacheSnapshot


CacheState = RuntimeCacheState


class SophiaCacheLayer(CacheLayerMixin):
    def __init__(
        self,
        *,
        name: str,
        snapshot: AttentionCacheSnapshot,
        seq_length: int,
        max_cache_shape: int,
    ):
        super().__init__()
        self.name = str(name)
        self.snapshot = snapshot.clone()
        self.payload = {
            str(key): value
            for key, value in self.snapshot.to_payload(prefix=self.name).items()
        }
        self._seq_length = int(seq_length)
        self._max_cache_shape = int(max_cache_shape)
        kv = self.snapshot.kv
        if kv is not None:
            self.keys = kv.unsqueeze(1)
            self.values = self.keys
            self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        del key_states, value_states
        raise NotImplementedError("SophiaCacheLayer is immutable and cannot be initialized lazily")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del key_states, value_states, cache_kwargs
        raise NotImplementedError("SophiaCacheLayer is an exported runtime snapshot and does not support update()")

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        del cache_position
        return self.get_seq_length(), 0

    def get_seq_length(self) -> int:
        return self._seq_length

    def get_max_cache_shape(self) -> int:
        return self._max_cache_shape

    @property
    def max_batch_size(self) -> int:
        if self.keys is None:
            return 0
        return int(self.keys.size(0))

    @property
    def max_cache_len(self) -> int:
        return int(self._max_cache_shape)

    @property
    def device(self) -> torch.device:
        if self.keys is None:
            return torch.device("cpu")
        return self.keys.device


class SophiaCache(Cache):
    def __init__(
        self,
        *,
        owner: object,
        cache: RuntimeCacheSnapshot,
        cache_pos: int,
        batch_size: int,
    ):
        self._cache = cache.clone()
        self._cache_pos = int(cache_pos)
        self._batch_size = int(batch_size)
        self._owner_ref = weakref.ref(owner)
        super().__init__(layers=self._build_layers())

    def _build_layers(self) -> list[SophiaCacheLayer]:
        owner = self._owner_ref()
        max_cache_shape = (
            -1
            if owner is None
            else int(getattr(owner.config, "max_position_embeddings", 0) or -1)
        )
        return [
            SophiaCacheLayer(
                name=name,
                snapshot=snapshot,
                seq_length=self._cache_pos,
                max_cache_shape=max_cache_shape,
            )
            for name, snapshot in self._cache.named_snapshots()
        ]

    def to_runtime_cache(self) -> RuntimeCacheSnapshot:
        return self._cache.clone()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return int(self._cache_pos)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        del layer_idx
        owner = self._owner_ref()
        if owner is None:
            return -1
        return int(getattr(owner.config, "max_position_embeddings", 0) or -1)

    @property
    def cache_pos(self) -> int:
        return int(self._cache_pos)

    @property
    def batch_size(self) -> int:
        return int(self._batch_size)


def cache_state_from_past_key_values(
    past_key_values: SophiaCache | None,
) -> CacheState | None:
    if past_key_values is None:
        return None
    if isinstance(past_key_values, SophiaCache):
        return CacheState(
            cache=past_key_values.to_runtime_cache(),
            batch_size=int(past_key_values.batch_size),
            cache_pos=int(past_key_values.cache_pos),
        )
    raise TypeError("past_key_values must be a SophiaCache returned by the HF adapter")


__all__ = [
    "CacheState",
    "SophiaCache",
    "SophiaCacheLayer",
    "cache_state_from_past_key_values",
]
