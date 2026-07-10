from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, TypeVar

from .semantics import default_rope_head_dim

if TYPE_CHECKING:
    from ml.modeling.config import SophiaModelConfig
    from ml.runtime.model.config import ModelArgs


TargetT = TypeVar("TargetT")


def _validate_model_spec(spec: ModelSpec) -> None:
    if int(spec.vocab_size) <= 0:
        raise ValueError(f"vocab_size must be > 0, got {spec.vocab_size}")
    if int(spec.dim) <= 0:
        raise ValueError(f"dim must be > 0, got {spec.dim}")
    if int(spec.n_layers) <= 0:
        raise ValueError(f"n_layers must be > 0, got {spec.n_layers}")
    if int(spec.n_heads) <= 0:
        raise ValueError(f"n_heads must be > 0, got {spec.n_heads}")
    if int(spec.head_dim) <= 0:
        raise ValueError(f"head_dim must be > 0, got {spec.head_dim}")
    if int(spec.num_key_value_heads) <= 0:
        raise ValueError("num_key_value_heads must be > 0")
    if int(spec.n_heads) % int(spec.num_key_value_heads) != 0:
        raise ValueError("n_heads must be divisible by num_key_value_heads")
    if int(spec.max_seq_len) <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {spec.max_seq_len}")
    if int(spec.max_batch_size) <= 0:
        raise ValueError(f"max_batch_size must be > 0, got {spec.max_batch_size}")
    if int(spec.ffn_hidden) <= 0:
        raise ValueError(f"ffn_hidden must be > 0, got {spec.ffn_hidden}")
    if int(spec.rope_head_dim) <= 0:
        raise ValueError(f"rope_head_dim must be > 0, got {spec.rope_head_dim}")
    if int(spec.rope_head_dim) > int(spec.head_dim):
        raise ValueError(
            f"rope_head_dim ({spec.rope_head_dim}) must be <= head_dim ({spec.head_dim})"
        )
    if int(spec.rope_head_dim) % 2 != 0:
        raise ValueError("rope_head_dim must be even")
    if float(spec.rope_factor) <= 0.0:
        raise ValueError(f"rope_factor must be > 0, got {spec.rope_factor}")
    if int(spec.original_seq_len) < 0:
        raise ValueError(f"original_seq_len must be >= 0, got {spec.original_seq_len}")


def project_model_spec(
    spec: ModelSpec,
    *,
    target_cls: type[TargetT],
    overrides: Mapping[str, object] | None = None,
) -> TargetT:
    """Build ``target_cls`` from the spec's fields, applying ``overrides`` last."""
    kwargs: dict[str, object] = {}
    for field_info in fields(target_cls):
        name = str(field_info.name)
        if not hasattr(spec, name):
            continue
        value = getattr(spec, name)
        kwargs[name] = list(value) if isinstance(value, (list, tuple)) else value
    if overrides:
        kwargs.update(dict(overrides))
    return target_cls(**kwargs)


@dataclass(frozen=True)
class ModelSpec:
    vocab_size: int = 49152
    dim: int = 1536
    n_layers: int = 30
    n_heads: int = 12
    head_dim: int = 128
    num_key_value_heads: int = 4
    rope_head_dim: int | None = None
    rope_theta: float = 500000.0
    original_seq_len: int = 0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    ffn_hidden: int = 4096
    use_qk_norm: bool = True
    norm_eps: float = 1e-6
    max_seq_len: int = 4096
    max_batch_size: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        rope_head_dim = (
            int(self.rope_head_dim)
            if self.rope_head_dim is not None
            else default_rope_head_dim(head_dim=int(self.head_dim))
        )
        object.__setattr__(self, "rope_head_dim", rope_head_dim)

        _validate_model_spec(self)

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
        raw: dict[str, object] = dict(payload)
        model_field_names = {field_info.name for field_info in fields(cls)}
        unknown = sorted(str(key) for key in raw if key not in model_field_names)
        if unknown:
            raise ValueError(
                "ModelSpec.from_mapping only accepts model fields; "
                f"unknown keys: {', '.join(unknown)}"
            )
        filtered = {key: value for key, value in raw.items() if key in model_field_names}
        return cls(**filtered)

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
            overrides={"max_seq_len": int(max_seq_len)},
        )

    def to_dict(self) -> dict[str, object]:
        return self.to_config().to_dict()

    def with_overrides(self, **overrides: object) -> ModelSpec:
        allowed_override_keys = {field_info.name for field_info in fields(type(self))}
        unknown = sorted(str(key) for key in overrides if key not in allowed_override_keys)
        if unknown:
            raise ValueError(
                "ModelSpec.with_overrides only accepts model fields; "
                f"unknown override keys: {', '.join(unknown)}"
            )
        payload: dict[str, object] = {
            field_info.name: getattr(self, field_info.name)
            for field_info in fields(type(self))
        }
        payload.update(dict(overrides))
        return type(self).from_mapping(payload)


__all__ = [
    "ModelSpec",
    "default_rope_head_dim",
    "project_model_spec",
]
