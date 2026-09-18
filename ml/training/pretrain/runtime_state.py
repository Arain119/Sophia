from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from ml.training.pretrain.machine_runtime import (
    MACHINE_RUNTIME_STATE_FIELD_NAMES,
    PretrainMachineRuntime,
)
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.value_semantics import (
    coerce_float,
    coerce_int,
)

_RUNTIME_STATE_RUN_CONFIG_FIELDS: dict[str, str] = {
    "seq_len": "seq_len",
    "batch_size": "batch_size",
    "accumulation_steps": "accumulation_steps",
    "learning_rate": "learning_rate",
    "max_steps": "max_steps",
    "gradient_checkpointing": "gradient_checkpointing",
    "gradient_checkpointing_exclude_first": "gradient_checkpointing_exclude_first",
    "gradient_checkpointing_exclude_last": "gradient_checkpointing_exclude_last",
    "loss_chunk_size": "loss_chunk_size",
    "eval_steps": "eval_steps",
    "log_interval": "log_interval",
    "eval_interval": "eval_interval",
    "save_interval": "save_interval",
    "target_tokens_per_update": "target_tokens_per_update",
}
_RUNTIME_STATE_COERCERS: dict[str, object] = {
    "seq_len": coerce_int,
    "batch_size": coerce_int,
    "accumulation_steps": coerce_int,
    "learning_rate": coerce_float,
    "max_steps": coerce_int,
    "gradient_checkpointing": coerce_int,
    "gradient_checkpointing_exclude_first": coerce_int,
    "gradient_checkpointing_exclude_last": coerce_int,
    "loss_chunk_size": coerce_int,
    "eval_steps": coerce_int,
    "log_interval": coerce_int,
    "eval_interval": coerce_int,
    "save_interval": coerce_int,
    "target_tokens_per_update": coerce_int,
}


@dataclass
class PretrainRuntimeState:
    seq_len: int = 0
    batch_size: int = 0
    accumulation_steps: int = 0
    learning_rate: float = 0.0
    max_steps: int = 0
    gradient_checkpointing: int = 0
    gradient_checkpointing_exclude_first: int = 0
    gradient_checkpointing_exclude_last: int = 0
    loss_chunk_size: int = 0
    eval_steps: int = 0
    log_interval: int = 0
    eval_interval: int = 0
    save_interval: int = 0
    target_tokens_per_update: int = 0
    _machine_runtime: PretrainMachineRuntime = field(
        default_factory=PretrainMachineRuntime,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_machine_runtime",
            PretrainMachineRuntime.from_source(self._machine_runtime),
        )

    def __getattr__(self, field_name: str) -> object:
        if str(field_name) in MACHINE_RUNTIME_STATE_FIELD_NAMES:
            return getattr(self._machine_runtime, str(field_name))
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {field_name!r}")

    def __setattr__(self, field_name: str, value: object) -> None:
        if str(field_name) == "_machine_runtime":
            object.__setattr__(self, "_machine_runtime", value)
            return
        if str(field_name) in MACHINE_RUNTIME_STATE_FIELD_NAMES:
            self._set_machine_runtime_field(str(field_name), value)
            return
        coercer = _RUNTIME_STATE_COERCERS.get(str(field_name))
        if coercer is None:
            object.__setattr__(self, field_name, value)
            return
        object.__setattr__(self, str(field_name), coercer(value))

    def _set_machine_runtime_field(self, field_name: str, value: object) -> None:
        machine_runtime = PretrainMachineRuntime.from_source(
            getattr(self, "_machine_runtime", None)
        )
        object.__setattr__(
            self,
            "_machine_runtime",
            replace(machine_runtime, **{str(field_name): value}),
        )

    @classmethod
    def owns_field(cls, field_name: str) -> bool:
        return str(field_name) in _RUNTIME_STATE_RUN_CONFIG_FIELDS or str(
            field_name
        ) in MACHINE_RUNTIME_STATE_FIELD_NAMES

    def apply_mapping(
        self,
        values: Mapping[str, object],
        *,
        field_names: tuple[str, ...] | list[str],
    ) -> None:
        for field_name in field_names:
            field_name = str(field_name)
            if field_name not in values:
                continue
            setattr(self, field_name, values[field_name])

    @classmethod
    def from_config(cls, cfg: PretrainRunConfig) -> PretrainRuntimeState:
        state = cls(
            **{
                field_name: _RUNTIME_STATE_COERCERS[field_name](
                    getattr(cfg, arg_name)
                )
                for field_name, arg_name in _RUNTIME_STATE_RUN_CONFIG_FIELDS.items()
            }
        )
        state.apply_machine_runtime(PretrainMachineRuntime.from_source(cfg))
        return state

    @staticmethod
    def _machine_runtime_from_source(
        source: PretrainRuntimeState | PretrainMachineRuntime | None,
    ) -> PretrainMachineRuntime | None:
        if source is None:
            return None
        if isinstance(source, PretrainMachineRuntime):
            return source
        return source.machine_runtime()

    @classmethod
    def from_config_with_machine_runtime(
        cls,
        cfg: PretrainRunConfig,
        *,
        source: PretrainRuntimeState | PretrainMachineRuntime | None,
    ) -> PretrainRuntimeState:
        state = cls.from_config(cfg)
        state.apply_machine_runtime(cls._machine_runtime_from_source(source))
        return state

    def project_run_config(self, cfg: PretrainRunConfig) -> PretrainRunConfig:
        payload = {
            arg_name: getattr(self, field_name)
            for field_name, arg_name in _RUNTIME_STATE_RUN_CONFIG_FIELDS.items()
        }
        projected = replace(cfg, **payload)
        return self.machine_runtime().apply_to_run_config(projected)

    def machine_runtime(self) -> PretrainMachineRuntime:
        return self._machine_runtime

    def apply_machine_runtime(
        self,
        machine_runtime: PretrainMachineRuntime | None,
    ) -> None:
        if machine_runtime is None:
            return
        payload = self._machine_runtime.to_payload()
        payload.update(machine_runtime.to_payload())
        object.__setattr__(
            self,
            "_machine_runtime",
            PretrainMachineRuntime.from_source(payload),
        )


__all__ = [
    "PretrainRuntimeState",
]
