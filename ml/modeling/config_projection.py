"""Projection helpers for the native Sophia Hybrid configuration."""

from __future__ import annotations

from dataclasses import fields
from typing import Protocol


class RuntimeModelArgsSource(Protocol):
    max_seq_len: int


def build_runtime_model_args[RuntimeModelArgsT](
    config: RuntimeModelArgsSource,
    *,
    model_args_cls: type[RuntimeModelArgsT],
    runtime_max_seq_len: int | None = None,
) -> RuntimeModelArgsT:
    max_seq_len = int(config.max_seq_len)
    if runtime_max_seq_len is not None:
        runtime_limit = int(runtime_max_seq_len)
        if not 0 < runtime_limit <= max_seq_len:
            raise ValueError(
                "runtime_max_seq_len must be in (0, config.max_seq_len], "
                f"got {runtime_limit} with max {max_seq_len}"
            )
        max_seq_len = runtime_limit
    kwargs: dict[str, object] = {}
    for arg_field in fields(model_args_cls):
        if hasattr(config, arg_field.name):
            kwargs[arg_field.name] = getattr(config, arg_field.name)
    kwargs["max_seq_len"] = max_seq_len
    return model_args_cls(**kwargs)


__all__ = ["build_runtime_model_args"]
