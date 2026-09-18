"""Canonical configuration for the native Sophia Hybrid decoder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass

from ml.core.spec.semantics import canonicalize_model_values, validate_model_values
from ml.modeling.runtime_backend import resolve_runtime_backend


def _object_mapping(config: object) -> dict[str, object]:
    canonical_payload = {
        field.name: getattr(config, field.name)
        for field in fields(SophiaModelConfig)
        if hasattr(config, field.name)
    }
    if canonical_payload:
        return canonical_payload
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


@dataclass
class SophiaModelConfig:
    """The only supported Sophia architecture schema."""

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
        normalized = canonicalize_model_values(asdict(self))
        for field_info in fields(type(self)):
            setattr(self, field_info.name, normalized[field_info.name])
        validate_model_values(normalized)

    @classmethod
    def get_defaults(cls) -> dict[str, object]:
        defaults: dict[str, object] = {}
        for field_info in fields(cls):
            if field_info.default is not MISSING:
                defaults[field_info.name] = field_info.default
            elif field_info.default_factory is not MISSING:
                defaults[field_info.name] = field_info.default_factory()
        return defaults

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> SophiaModelConfig:
        raw = dict(values)
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(str(key) for key in raw if key not in allowed)
        if unknown:
            raise ValueError(
                "SophiaModelConfig only accepts native Sophia Hybrid fields; "
                f"unknown keys: {', '.join(unknown)}"
            )
        return cls(**raw)

    @classmethod
    def from_object(cls, config: object) -> SophiaModelConfig:
        return cls.from_mapping(_object_mapping(config))

    def to_model_spec(self):
        from ml.core.spec import ModelSpec

        return ModelSpec.from_config(self)

    def to_model_args(self, *, runtime_max_seq_len: int | None = None) -> object:
        from ml.modeling.config_projection import build_runtime_model_args

        return build_runtime_model_args(
            self,
            model_args_cls=resolve_runtime_backend().model_args_cls,
            runtime_max_seq_len=runtime_max_seq_len,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["SophiaModelConfig"]
