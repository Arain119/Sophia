from __future__ import annotations

from collections.abc import Mapping
import math


PINNED_SEMANTIC_ADAM_EPS = 1e-8


def canonicalize_model_values(values: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = dict(values)
    integer_fields = (
        "vocab_size",
        "dim",
        "n_layers",
        "num_heads",
        "head_dim",
        "ffn_hidden",
        "kda_decay_rank",
        "kda_output_gate_rank",
        "mla_q_rank",
        "mla_kv_rank",
        "short_conv_kernel",
        "attn_res_block_size",
        "max_seq_len",
        "max_batch_size",
    )
    float_fields = (
        "kda_decay_lower_bound",
        "kda_dt_min",
        "kda_dt_max",
        "kda_dt_floor",
        "kda_a_log_init",
        "norm_eps",
        "dropout",
        "initializer_range",
        "situ_gate_softcap",
        "situ_up_softcap",
    )
    for name in integer_fields:
        if name in normalized:
            normalized[name] = int(normalized[name])
    for name in float_fields:
        if name in normalized:
            normalized[name] = float(normalized[name])
    if "kda_backend" in normalized:
        normalized["kda_backend"] = str(normalized["kda_backend"])
    if "kda_output_gate_full_rank" in normalized:
        normalized["kda_output_gate_full_rank"] = bool(
            normalized["kda_output_gate_full_rank"]
        )
    return normalized


def validate_model_values(values: Mapping[str, object]) -> None:
    positive_ints = (
        "vocab_size",
        "dim",
        "n_layers",
        "num_heads",
        "head_dim",
        "ffn_hidden",
        "kda_decay_rank",
        "kda_output_gate_rank",
        "mla_q_rank",
        "mla_kv_rank",
        "short_conv_kernel",
        "attn_res_block_size",
        "max_seq_len",
        "max_batch_size",
    )
    for name in positive_ints:
        value = int(values[name])
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")

    lower_bound = float(values["kda_decay_lower_bound"])
    if not -5.0 <= lower_bound < 0.0:
        raise ValueError(
            "kda_decay_lower_bound must be in [-5, 0), "
            f"got {lower_bound}"
        )
    dt_min = float(values["kda_dt_min"])
    dt_max = float(values["kda_dt_max"])
    dt_floor = float(values["kda_dt_floor"])
    a_log_init = float(values["kda_a_log_init"])
    for name, value in (
        ("kda_dt_min", dt_min),
        ("kda_dt_max", dt_max),
        ("kda_dt_floor", dt_floor),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value}")
    if dt_min > dt_max:
        raise ValueError(
            f"kda_dt_min must be <= kda_dt_max, got {dt_min} > {dt_max}"
        )
    if dt_floor > dt_min:
        raise ValueError(
            f"kda_dt_floor must be <= kda_dt_min, got {dt_floor} > {dt_min}"
        )
    if not math.isfinite(a_log_init):
        raise ValueError(f"kda_a_log_init must be finite, got {a_log_init}")
    norm_eps = float(values["norm_eps"])
    if not math.isfinite(norm_eps) or norm_eps <= 0.0:
        raise ValueError(f"norm_eps must be finite and > 0, got {norm_eps}")
    dropout = float(values["dropout"])
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    initializer_range = float(values["initializer_range"])
    if not math.isfinite(initializer_range) or initializer_range <= 0.0:
        raise ValueError(
            f"initializer_range must be finite and > 0, got {initializer_range}"
        )
    for name in ("situ_gate_softcap", "situ_up_softcap"):
        value = float(values[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value}")
    backend = str(values.get("kda_backend", "auto"))
    if backend not in {"auto", "reference", "fla"}:
        raise ValueError(
            "kda_backend must be one of auto/reference/fla, "
            f"got {backend!r}"
        )


__all__ = [
    "PINNED_SEMANTIC_ADAM_EPS",
    "canonicalize_model_values",
    "validate_model_values",
]
