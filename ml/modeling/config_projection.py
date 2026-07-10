"""Config projection helpers shared by Sophia models."""

from __future__ import annotations

from dataclasses import fields
from typing import Protocol, TypeVar

from ml.modeling.config import SophiaModelConfig


class RuntimeModelArgsSource(Protocol):
    max_seq_len: int


def rope_head_dim_from_partial_factor(*, head_dim: int, partial_rotary_factor: float) -> int:
    rope_head_dim = int(round(float(head_dim) * float(partial_rotary_factor)))
    rope_head_dim = max(rope_head_dim, 2)
    if rope_head_dim % 2 != 0:
        rope_head_dim -= 1
    return max(rope_head_dim, 2)


def canonicalize_canonical_config_values(
    values: dict[str, object],
) -> dict[str, object]:
    canonical_fields = set(SophiaModelConfig.get_defaults())
    canonical_kwargs = {key: values[key] for key in canonical_fields if key in values}
    return SophiaModelConfig(**canonical_kwargs).to_dict()


RuntimeModelArgsT = TypeVar("RuntimeModelArgsT")


def build_runtime_model_args(
    config: RuntimeModelArgsSource,
    *,
    model_args_cls: type[RuntimeModelArgsT],
    runtime_max_seq_len: int | None = None,
) -> RuntimeModelArgsT:
    max_seq_len = int(config.max_seq_len)
    if runtime_max_seq_len is not None:
        runtime_limit = int(runtime_max_seq_len)
        if runtime_limit <= 0:
            raise ValueError(
                f"runtime_max_seq_len must be > 0 when provided, got {runtime_max_seq_len}"
            )
        if runtime_limit > max_seq_len:
            raise ValueError(
                "runtime_max_seq_len must be <= config.max_seq_len, "
                f"got {runtime_limit} > {max_seq_len}"
            )
        max_seq_len = runtime_limit
    kwargs: dict[str, object] = {}
    for arg_field in fields(model_args_cls):
        if not hasattr(config, arg_field.name):
            continue
        value = getattr(config, arg_field.name)
        kwargs[arg_field.name] = list(value) if isinstance(value, (list, tuple)) else value
    kwargs["max_seq_len"] = int(max_seq_len)
    return model_args_cls(**kwargs)


__all__ = [
    "build_runtime_model_args",
    "canonicalize_canonical_config_values",
    "rope_head_dim_from_partial_factor",
]
