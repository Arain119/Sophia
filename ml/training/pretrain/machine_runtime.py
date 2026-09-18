from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from ml.training.pretrain.value_semantics import coerce_int
from ml.training.pretrain.engine.step_execution import (
    normalize_step_execution_backend,
)

MACHINE_RUNTIME_FIELD_NAMES: tuple[str, ...] = (
    "dataloader_num_workers",
    "dataloader_prefetch_factor",
    "dataloader_persistent_workers",
    "shard_preload",
    "shard_preload_bytes",
)
MACHINE_RUNTIME_STATE_FIELD_NAMES: tuple[str, ...] = (
    "step_execution_backend",
    *MACHINE_RUNTIME_FIELD_NAMES,
)
_MACHINE_RUNTIME_DEFAULTS: dict[str, object] = {
    "step_execution_backend": "",
    "dataloader_num_workers": -1,
    "dataloader_prefetch_factor": -1,
    "dataloader_persistent_workers": -1,
    "shard_preload": -1,
    "shard_preload_bytes": -1,
}


@dataclass(frozen=True)
class PretrainMachineRuntime:
    step_execution_backend: str = ""
    dataloader_num_workers: int = -1
    dataloader_prefetch_factor: int = -1
    dataloader_persistent_workers: int = -1
    shard_preload: int = -1
    shard_preload_bytes: int = -1

    def __post_init__(self) -> None:
        backend = (
            ""
            if not str(self.step_execution_backend or "").strip()
            else normalize_step_execution_backend(self.step_execution_backend)
        )
        object.__setattr__(self, "step_execution_backend", backend)
        object.__setattr__(
            self,
            "dataloader_num_workers",
            coerce_int(self.dataloader_num_workers, default=-1),
        )
        object.__setattr__(
            self,
            "dataloader_prefetch_factor",
            coerce_int(self.dataloader_prefetch_factor, default=-1),
        )
        object.__setattr__(
            self,
            "dataloader_persistent_workers",
            coerce_int(self.dataloader_persistent_workers, default=-1),
        )
        object.__setattr__(
            self,
            "shard_preload",
            coerce_int(self.shard_preload, default=-1),
        )
        object.__setattr__(
            self,
            "shard_preload_bytes",
            coerce_int(self.shard_preload_bytes, default=-1),
        )

    @classmethod
    def from_source(cls, source: object | None) -> PretrainMachineRuntime:
        if source is None:
            return cls()
        if isinstance(source, cls):
            return source
        read = source.get if isinstance(source, Mapping) else getattr
        payload = {
            field_name: (
                read(field_name, default)
                if isinstance(source, Mapping)
                else read(source, field_name, default)
            )
            for field_name, default in _MACHINE_RUNTIME_DEFAULTS.items()
        }
        return cls(**payload)

    def apply_to_runtime_state(self, state: object) -> None:
        if str(self.step_execution_backend).strip():
            state.step_execution_backend = self.step_execution_backend
        for field_name in MACHINE_RUNTIME_FIELD_NAMES:
            value = int(getattr(self, field_name))
            if value >= 0:
                setattr(state, field_name, value)

    def apply_to_run_config(self, cfg):
        payload = {
            field_name: int(getattr(self, field_name))
            for field_name in MACHINE_RUNTIME_FIELD_NAMES
        }
        if str(self.step_execution_backend).strip() and hasattr(cfg, "step_execution_backend"):
            payload["step_execution_backend"] = str(self.step_execution_backend)
        return replace(cfg, **payload)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if str(self.step_execution_backend).strip():
            payload["step_execution_backend"] = str(self.step_execution_backend)
        for field_name in MACHINE_RUNTIME_FIELD_NAMES:
            value = int(getattr(self, field_name))
            if value >= 0:
                payload[field_name] = value
        return payload


__all__ = [
    "MACHINE_RUNTIME_FIELD_NAMES",
    "MACHINE_RUNTIME_STATE_FIELD_NAMES",
    "PretrainMachineRuntime",
]
