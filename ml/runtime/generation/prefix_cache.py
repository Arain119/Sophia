"""Prefix cache for efficient prompt reuse in the runtime implementation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from threading import RLock

from ml.runtime.model.state import RuntimeCacheSnapshot


class PrefixCacheSnapshotError(RuntimeError):
    """Raised when a persisted prefix-cache snapshot violates the runtime contract."""


@dataclass
class PrefixSnapshot:
    """Snapshot of KV cache state for a prefix."""
    prefix_len: int
    cache: RuntimeCacheSnapshot
    replay_len: int


class PrefixCache:
    """Thread-safe prefix cache for storing and retrieving prompt snapshots."""

    def __init__(self, max_entries: int = 100):
        self.max_entries = int(max_entries)
        if self.max_entries <= 0:
            raise ValueError(f"max_entries must be > 0, got {max_entries}")
        self.cache: dict[str, PrefixSnapshot] = {}
        self._token_keys: list[tuple[str, tuple[int, ...], str]] = []
        self._lock = RLock()

    @staticmethod
    def _cache_key(tokens: tuple[int, ...], *, namespace: str) -> str:
        payload = f"{namespace}\n" + ",".join(str(int(t)) for t in tokens)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get(self, tokens: list[int], *, namespace: str = "") -> PrefixSnapshot | None:
        """Retrieve the exact snapshot for the given token sequence."""
        key = self._cache_key(tuple(tokens), namespace=str(namespace))
        with self._lock:
            return self.cache.get(key)

    def get_longest_prefix(
        self,
        tokens: list[int],
        *,
        namespace: str = "",
        min_tokens: int = 1,
    ) -> tuple[int, PrefixSnapshot | None]:
        """Retrieve the longest cached prefix snapshot for the given token sequence."""
        token_tuple = tuple(int(token) for token in tokens)
        namespace_str = str(namespace)
        minimum = max(int(min_tokens), 1)
        with self._lock:
            for prefix_len in range(len(token_tuple), minimum - 1, -1):
                key = self._cache_key(
                    token_tuple[:prefix_len],
                    namespace=namespace_str,
                )
                snapshot = self.cache.get(key)
                if snapshot is not None:
                    return prefix_len, snapshot
        return 0, None

    def put(self, tokens: list[int], snapshot: PrefixSnapshot, *, namespace: str = "") -> None:
        """Store a snapshot for the given token sequence."""
        token_tuple = tuple(int(t) for t in tokens)
        namespace_str = str(namespace)
        key = self._cache_key(token_tuple, namespace=namespace_str)
        with self._lock:
            if key not in self.cache:
                if len(self.cache) >= self.max_entries:
                    # Evict the first inserted entry (simple FIFO).
                    if self._token_keys:
                        _evicted_namespace, _evicted_tokens, evicted_key = self._token_keys.pop(0)
                        self.cache.pop(evicted_key, None)
                self._token_keys.append((namespace_str, token_tuple, key))
            self.cache[key] = snapshot

    def delete(self, tokens: list[int], *, namespace: str = "") -> None:
        """Remove a snapshot for the given token sequence if present."""
        token_tuple = tuple(int(t) for t in tokens)
        namespace_str = str(namespace)
        key = self._cache_key(token_tuple, namespace=namespace_str)
        with self._lock:
            self.cache.pop(key, None)
            self._token_keys = [entry for entry in self._token_keys if entry[2] != key]

    def dump_snapshots(self) -> dict[str, object]:
        """Export serialized snapshot contents for debugging or checkpointing."""
        with self._lock:
            return {
                k: {
                    "prefix_len": v.prefix_len,
                    "replay_len": v.replay_len,
                    "cache": v.cache.to_payload(),
                }
                for k, v in self.cache.items()
            }


def model_prefix_cache_namespace(model: object) -> str:
    namespace = getattr(model, "prefix_cache_namespace", None)
    if not isinstance(namespace, str):
        raise TypeError(
            "prefix-cache-enabled models must expose an explicit string "
            "`prefix_cache_namespace` property"
        )
    namespace = namespace.strip()
    if not namespace:
        raise ValueError("prefix_cache_namespace must be a non-empty string")
    return namespace


__all__ = [
    "PrefixCache",
    "PrefixCacheSnapshotError",
    "PrefixSnapshot",
    "model_prefix_cache_namespace",
]
