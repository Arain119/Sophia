from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import RLock

import torch


def _clone_optional_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().clone()


_CACHE_PAYLOAD_SUFFIXES: tuple[tuple[str, str], ...] = (("_kv", "kv"),)


@dataclass(frozen=True)
class AttentionCacheSnapshot:
    kv: torch.Tensor | None = None

    def clone(self) -> AttentionCacheSnapshot:
        return AttentionCacheSnapshot(kv=_clone_optional_tensor(self.kv))

    def is_empty(self) -> bool:
        return not any(value is not None for _name, value in self.tensor_fields())

    def batch_size(self) -> int:
        batch_size = 0
        for _name, value in self.tensor_fields():
            if value is not None:
                batch_size = max(batch_size, int(value.size(0)))
        return batch_size

    def tensor_fields(self) -> tuple[tuple[str, torch.Tensor | None], ...]:
        return (("kv", self.kv),)

    def to_payload(self, *, prefix: str) -> dict[str, torch.Tensor]:
        payload: dict[str, torch.Tensor] = {}
        for field_name, value in self.tensor_fields():
            if value is None:
                continue
            payload[f"{prefix}_{field_name}"] = value.detach().clone()
        return payload

    @classmethod
    def from_field_map(
        cls,
        field_map: Mapping[str, object] | None,
    ) -> AttentionCacheSnapshot | None:
        if field_map is None:
            return None
        values: dict[str, torch.Tensor | None] = {}
        for field_name in ("kv",):
            value = field_map.get(field_name)
            if value is None:
                values[field_name] = None
                continue
            if not torch.is_tensor(value):
                raise TypeError(
                    f"attention cache field {field_name!r} must be a tensor or None"
                )
            values[field_name] = value.detach().clone()
        snapshot = cls(**values)
        if snapshot.is_empty():
            return None
        return snapshot


@dataclass(frozen=True)
class RuntimeCacheSnapshot:
    layers: tuple[AttentionCacheSnapshot | None, ...] = ()

    def clone(self) -> RuntimeCacheSnapshot:
        return RuntimeCacheSnapshot(
            layers=tuple(None if snap is None else snap.clone() for snap in self.layers),
        )

    def is_empty(self) -> bool:
        return all(snap is None or snap.is_empty() for snap in self.layers)

    def batch_size(self) -> int:
        batch_size = 0
        for _name, snapshot in self.named_snapshots():
            batch_size = max(batch_size, snapshot.batch_size())
        return batch_size

    def named_snapshots(self) -> tuple[tuple[str, AttentionCacheSnapshot], ...]:
        pairs: list[tuple[str, AttentionCacheSnapshot]] = []
        for index, snapshot in enumerate(self.layers):
            if snapshot is not None and not snapshot.is_empty():
                pairs.append((f"layer_{index}", snapshot))
        return tuple(pairs)

    def to_payload(self) -> dict[str, torch.Tensor]:
        payload: dict[str, torch.Tensor] = {}
        for prefix, snapshot in self.named_snapshots():
            payload.update(snapshot.to_payload(prefix=prefix))
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> RuntimeCacheSnapshot:
        layer_fields: dict[int, dict[str, object]] = {}
        for raw_key, value in payload.items():
            key = str(raw_key)
            for suffix, field_name in _CACHE_PAYLOAD_SUFFIXES:
                if not key.endswith(suffix):
                    continue
                prefix = key[: -len(suffix)]
                if prefix.startswith("layer_"):
                    index = int(prefix.removeprefix("layer_"))
                    layer_fields.setdefault(index, {})[field_name] = value
                    break
            else:
                raise ValueError(f"unrecognized runtime cache payload key: {key}")
        return cls(layers=_build_cache_track(layer_fields))


def _build_cache_track(
    field_maps: Mapping[int, Mapping[str, object]],
) -> tuple[AttentionCacheSnapshot | None, ...]:
    if not field_maps:
        return ()
    size = max(int(index) for index in field_maps) + 1
    track: list[AttentionCacheSnapshot | None] = [None] * size
    for index, field_map in field_maps.items():
        track[int(index)] = AttentionCacheSnapshot.from_field_map(field_map)
    return tuple(track)


@dataclass
class TransformerRuntimeState:
    lock: RLock = field(default_factory=RLock)
    loss_chunk_size: int = 0


__all__ = [
    "AttentionCacheSnapshot",
    "RuntimeCacheSnapshot",
    "TransformerRuntimeState",
]
