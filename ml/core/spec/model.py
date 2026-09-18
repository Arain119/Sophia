from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from .semantics import canonicalize_model_values, validate_model_values

if TYPE_CHECKING:
    from ml.modeling.config import SophiaModelConfig
    from ml.runtime.model.config import ModelArgs


def project_model_spec[TargetT](
    spec: ModelSpec,
    *,
    target_cls: type[TargetT],
    overrides: Mapping[str, object] | None = None,
) -> TargetT:
    kwargs: dict[str, object] = {}
    for field_info in fields(target_cls):
        name = str(field_info.name)
        if hasattr(spec, name):
            kwargs[name] = getattr(spec, name)
    if overrides:
        kwargs.update(dict(overrides))
    return target_cls(**kwargs)


@dataclass(frozen=True)
class ModelSpec:
    vocab_size: int = 65536
    dim: int = 1536
    n_layers: int = 28
    num_heads: int = 16
    head_dim: int = 128
    ffn_hidden: int = 3968
    kda_decay_rank: int = 128
    kda_output_gate_rank: int = 128
    kda_output_gate_full_rank: bool = True
    kda_decay_lower_bound: float = -5.0
    kda_dt_min: float = 1e-3
    kda_dt_max: float = 1e-1
    kda_dt_floor: float = 1e-4
    kda_a_log_init: float = 0.0
    mla_q_rank: int = 384
    mla_kv_rank: int = 128
    short_conv_kernel: int = 4
    attn_res_block_size: int = 4
    situ_gate_softcap: float = 4.0
    situ_up_softcap: float = 25.0
    norm_eps: float = 1e-5
    max_seq_len: int = 4096
    max_batch_size: int = 4
    dropout: float = 0.0
    initializer_range: float = 0.02
    kda_backend: str = "auto"

    def __post_init__(self) -> None:
        normalized = canonicalize_model_values(vars(self))
        for name, value in normalized.items():
            object.__setattr__(self, name, value)
        validate_model_values(normalized)

    @classmethod
    def default(cls) -> ModelSpec:
        return cls()

    @classmethod
    def from_config(cls, config: SophiaModelConfig | object) -> ModelSpec:
        if not hasattr(config, "to_dict"):
            raise TypeError("config must provide to_dict()")
        return cls.from_mapping(config.to_dict())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ModelSpec:
        raw = dict(payload)
        allowed = {field_info.name for field_info in fields(cls)}
        unknown = sorted(str(key) for key in raw if key not in allowed)
        if unknown:
            raise ValueError(
                "ModelSpec.from_mapping only accepts native Sophia Hybrid fields; "
                f"unknown keys: {', '.join(unknown)}"
            )
        return cls(**raw)

    def to_config(self) -> SophiaModelConfig:
        from ml.modeling.config import SophiaModelConfig

        return project_model_spec(self, target_cls=SophiaModelConfig)

    def to_model_args(self, *, runtime_max_seq_len: int | None = None) -> ModelArgs:
        from ml.runtime.model.config import ModelArgs

        max_seq_len = int(self.max_seq_len)
        if runtime_max_seq_len is not None:
            max_seq_len = min(max_seq_len, int(runtime_max_seq_len))
        return project_model_spec(
            self,
            target_cls=ModelArgs,
            overrides={"max_seq_len": max_seq_len},
        )

    def to_dict(self) -> dict[str, object]:
        return self.to_config().to_dict()

    def with_overrides(self, **overrides: object) -> ModelSpec:
        allowed = {field_info.name for field_info in fields(type(self))}
        unknown = sorted(str(key) for key in overrides if key not in allowed)
        if unknown:
            raise ValueError(
                "ModelSpec.with_overrides only accepts native Sophia Hybrid fields; "
                f"unknown override keys: {', '.join(unknown)}"
            )
        payload = {name: getattr(self, name) for name in allowed}
        payload.update(overrides)
        return type(self).from_mapping(payload)


__all__ = ["ModelSpec", "project_model_spec"]
