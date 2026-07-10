"""Sophia model configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass

from ml.core.spec.semantics import (
    canonicalize_model_values,
    validate_model_values,
)
from ml.modeling.runtime_backend import (
    resolve_runtime_backend,
)


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
    return canonical_payload


@dataclass
class SophiaModelConfig:
    """Canonical Sophia configuration for core models."""

    # Core dimensions
    vocab_size: int = 49152
    dim: int = 1536
    n_layers: int = 30
    n_heads: int = 12
    head_dim: int = 128

    # Grouped-query attention
    num_key_value_heads: int = 4

    # RoPE configuration
    rope_head_dim: int | None = None
    rope_theta: float = 500000.0
    original_seq_len: int = 0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1

    # Attention parameters
    ffn_hidden: int = 4096
    use_qk_norm: bool = True
    # Normalization
    norm_eps: float = 1e-6

    # Sequence length
    max_seq_len: int = 4096
    max_batch_size: int = 4

    # Dropout (for training)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        normalized = canonicalize_model_values(asdict(self))
        for field_info in fields(type(self)):
            if field_info.name in normalized:
                setattr(self, field_info.name, normalized[field_info.name])
        validate_model_values(normalized)

    @classmethod
    def get_defaults(cls) -> dict[str, object]:
        """Get the construction defaults before derived fields are resolved."""
        defaults: dict[str, object] = {}
        for field_info in fields(cls):
            if field_info.default is not MISSING:
                defaults[field_info.name] = field_info.default
            elif field_info.default_factory is not MISSING:
                defaults[field_info.name] = field_info.default_factory()
        return defaults

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object] | dict[str, object],
    ) -> SophiaModelConfig:
        from ml.modeling.config_projection import (
            canonicalize_canonical_config_values,
        )
        raw = dict(values)
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(str(key) for key in raw if key not in allowed)
        if unknown:
            raise ValueError(
                "SophiaModelConfig.from_mapping only accepts canonical Sophia model fields; "
                f"unknown keys: {', '.join(unknown)}"
            )
        normalized = canonicalize_canonical_config_values(raw)
        return cls(**{key: normalized[key] for key in allowed if key in normalized})

    @classmethod
    def from_object(cls, config: object) -> SophiaModelConfig:
        return cls.from_mapping(_object_mapping(config))

    def to_model_spec(self):
        """Project the mutable config dataclass into the frozen model spec."""
        from ml.core.spec import ModelSpec

        return ModelSpec.from_config(self)

    def to_model_args(
        self,
        *,
        runtime_max_seq_len: int | None = None,
    ) -> object:
        from ml.modeling.config_projection import build_runtime_model_args
        return build_runtime_model_args(
            self,
            model_args_cls=resolve_runtime_backend().model_args_cls,
            runtime_max_seq_len=runtime_max_seq_len,
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return asdict(self)


__all__ = ["SophiaModelConfig"]
