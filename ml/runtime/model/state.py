from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import RLock

import torch


def _clone(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().clone()


_CACHE_SUFFIXES = (
    ("_recurrent", "recurrent"),
    ("_conv", "conv"),
    ("_latent", "latent"),
)


@dataclass(frozen=True)
class LayerCacheSnapshot:
    recurrent: torch.Tensor | None = None
    conv: torch.Tensor | None = None
    latent: torch.Tensor | None = None

    def clone(self) -> LayerCacheSnapshot:
        return LayerCacheSnapshot(
            recurrent=_clone(self.recurrent),
            conv=_clone(self.conv),
            latent=_clone(self.latent),
        )

    def tensor_fields(self) -> tuple[tuple[str, torch.Tensor | None], ...]:
        return (
            ("recurrent", self.recurrent),
            ("conv", self.conv),
            ("latent", self.latent),
        )

    def is_empty(self) -> bool:
        return all(value is None for _name, value in self.tensor_fields())

    def batch_size(self) -> int:
        return max(
            (int(value.size(0)) for _name, value in self.tensor_fields() if value is not None),
            default=0,
        )

    def to_payload(self, *, prefix: str) -> dict[str, torch.Tensor]:
        return {
            f"{prefix}_{name}": value.detach().clone()
            for name, value in self.tensor_fields()
            if value is not None
        }

    @classmethod
    def from_field_map(
        cls,
        field_map: Mapping[str, object] | None,
    ) -> LayerCacheSnapshot | None:
        if field_map is None:
            return None
        values: dict[str, torch.Tensor | None] = {}
        for name in ("recurrent", "conv", "latent"):
            value = field_map.get(name)
            if value is not None and not torch.is_tensor(value):
                raise TypeError(f"layer cache field {name!r} must be a tensor or None")
            values[name] = _clone(value)
        snapshot = cls(**values)
        return None if snapshot.is_empty() else snapshot


@dataclass(frozen=True)
class RuntimeCacheSnapshot:
    layers: tuple[LayerCacheSnapshot | None, ...] = ()

    def clone(self) -> RuntimeCacheSnapshot:
        return RuntimeCacheSnapshot(
            layers=tuple(None if item is None else item.clone() for item in self.layers)
        )

    def is_empty(self) -> bool:
        return all(item is None or item.is_empty() for item in self.layers)

    def batch_size(self) -> int:
        return max((item.batch_size() for item in self.layers if item is not None), default=0)

    def named_snapshots(self) -> tuple[tuple[str, LayerCacheSnapshot], ...]:
        return tuple(
            (f"layer_{index}", item)
            for index, item in enumerate(self.layers)
            if item is not None and not item.is_empty()
        )

    def to_payload(self) -> dict[str, torch.Tensor]:
        payload: dict[str, torch.Tensor] = {}
        for prefix, snapshot in self.named_snapshots():
            payload.update(snapshot.to_payload(prefix=prefix))
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> RuntimeCacheSnapshot:
        fields_by_layer: dict[int, dict[str, object]] = {}
        for raw_key, value in payload.items():
            key = str(raw_key)
            for suffix, field_name in _CACHE_SUFFIXES:
                if key.endswith(suffix):
                    prefix = key[: -len(suffix)]
                    if not prefix.startswith("layer_"):
                        break
                    index = int(prefix.removeprefix("layer_"))
                    fields_by_layer.setdefault(index, {})[field_name] = value
                    break
            else:
                raise ValueError(f"unrecognized runtime cache payload key: {key}")
        if not fields_by_layer:
            return cls()
        layers: list[LayerCacheSnapshot | None] = [None] * (max(fields_by_layer) + 1)
        for index, field_map in fields_by_layer.items():
            layers[index] = LayerCacheSnapshot.from_field_map(field_map)
        return cls(layers=tuple(layers))


@dataclass
class TransformerRuntimeState:
    lock: RLock = field(default_factory=RLock)
    loss_chunk_size: int = 0


__all__ = ["LayerCacheSnapshot", "RuntimeCacheSnapshot", "TransformerRuntimeState"]
