from __future__ import annotations

from collections.abc import Mapping

DEFAULT_N_LAYERS = 30

# Canonical pinned AdamW epsilon for release optimizer semantics.
PINNED_SEMANTIC_ADAM_EPS = 1e-8


def default_rope_head_dim(*, head_dim: int) -> int:
    scaled = max(int(head_dim) // 2, 2)
    if scaled % 2 != 0:
        scaled -= 1
    return max(scaled, 2)


def canonicalize_model_values(values: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = dict(values)
    head_dim = int(normalized.get("head_dim", 0) or 0)

    rope_head_dim = normalized.get("rope_head_dim")
    if rope_head_dim is None:
        normalized["rope_head_dim"] = int(default_rope_head_dim(head_dim=head_dim))
    else:
        normalized["rope_head_dim"] = int(rope_head_dim)

    if "ffn_hidden" in normalized:
        normalized["ffn_hidden"] = int(normalized["ffn_hidden"])
    if "use_qk_norm" in normalized:
        normalized["use_qk_norm"] = bool(normalized["use_qk_norm"])
    return normalized


def validate_core_shape(
    *,
    vocab_size: int,
    dim: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    num_key_value_heads: int,
    max_seq_len: int,
    max_batch_size: int,
    ffn_hidden: int = 0,
    use_qk_norm: bool = False,
) -> None:
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")
    if n_layers <= 0:
        raise ValueError(f"n_layers must be > 0, got {n_layers}")
    if n_heads <= 0:
        raise ValueError(f"n_heads must be > 0, got {n_heads}")
    if head_dim <= 0:
        raise ValueError(f"head_dim must be > 0, got {head_dim}")
    if num_key_value_heads <= 0:
        raise ValueError(
            "num_key_value_heads must be > 0 for the runtime route, "
            f"got {num_key_value_heads}"
        )
    if n_heads % num_key_value_heads != 0:
        raise ValueError(
            f"n_heads ({n_heads}) must be divisible by num_key_value_heads ({num_key_value_heads})"
        )
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be > 0, got {max_batch_size}")
    if ffn_hidden <= 0:
        raise ValueError(f"ffn_hidden must be > 0, got {ffn_hidden}")
    del use_qk_norm


def validate_rope(
    *,
    original_seq_len: int,
    rope_factor: float,
    rope_head_dim: int,
    head_dim: int,
) -> None:
    if original_seq_len < 0:
        raise ValueError(f"original_seq_len must be >= 0, got {original_seq_len}")
    if rope_factor <= 0:
        raise ValueError(f"rope_factor must be > 0, got {rope_factor}")
    if rope_head_dim <= 0:
        raise ValueError(f"rope_head_dim must be > 0, got {rope_head_dim}")
    if rope_head_dim > head_dim:
        raise ValueError(
            f"rope_head_dim ({rope_head_dim}) must be <= head_dim ({head_dim})"
        )
    if rope_head_dim % 2 != 0:
        raise ValueError(
            f"rope_head_dim ({rope_head_dim}) must be even for rotary embedding"
        )


def validate_model_values(values: Mapping[str, object]) -> None:
    vocab_size = int(values["vocab_size"])
    dim = int(values["dim"])
    n_layers = int(values["n_layers"])
    n_heads = int(values["n_heads"])
    head_dim = int(values["head_dim"])
    num_key_value_heads = int(values["num_key_value_heads"])
    rope_head_dim = int(values["rope_head_dim"])
    original_seq_len = int(values["original_seq_len"])
    rope_factor = float(values["rope_factor"])
    ffn_hidden = int(values.get("ffn_hidden", 0))
    use_qk_norm = bool(values.get("use_qk_norm", False))
    max_seq_len = int(values["max_seq_len"])
    max_batch_size = int(values["max_batch_size"])

    validate_core_shape(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        num_key_value_heads=num_key_value_heads,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        ffn_hidden=ffn_hidden,
        use_qk_norm=use_qk_norm,
    )
    validate_rope(
        original_seq_len=original_seq_len,
        rope_factor=rope_factor,
        rope_head_dim=rope_head_dim,
        head_dim=head_dim,
    )

__all__ = [
    "DEFAULT_N_LAYERS",
    "PINNED_SEMANTIC_ADAM_EPS",
    "canonicalize_model_values",
    "default_rope_head_dim",
    "validate_core_shape",
    "validate_model_values",
    "validate_rope",
]
