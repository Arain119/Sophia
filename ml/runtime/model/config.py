from __future__ import annotations

from dataclasses import dataclass

from ml.core.spec.semantics import (
    canonicalize_model_values,
    validate_model_values,
)


@dataclass
class ModelArgs:
    """Torch-free runtime model configuration."""

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
    use_cache: bool = True

    def __post_init__(self) -> None:
        normalized = canonicalize_model_values(vars(self))
        for name, value in normalized.items():
            if hasattr(self, name):
                setattr(self, name, value)
        validate_model_values(normalized)

    def ensure_runtime_batch_capacity(self, batch_size: int) -> int:
        required = max(int(batch_size), 1)
        if required > int(self.max_batch_size):
            self.max_batch_size = int(required)
        return int(self.max_batch_size)

    def runtime_batch_capacity(self) -> int:
        return max(int(self.max_batch_size), 1)

    def ensure_runtime_sequence_capacity(self, max_seq_len: int) -> int:
        required = max(int(max_seq_len), 1)
        if required > int(self.max_seq_len):
            self.max_seq_len = int(required)
        return int(self.max_seq_len)

    def runtime_sequence_capacity(self) -> int:
        return max(int(self.max_seq_len), 1)

    def runtime_max_seq_len(self) -> int:
        return self.runtime_sequence_capacity()

    def runtime_layer_count(self) -> int:
        return max(int(self.n_layers), 1)

    def runtime_prefix_replay_len(self) -> int:
        return int(self.runtime_sequence_capacity()) * int(self.runtime_layer_count())

    def runtime_rope_kwargs(self, *, max_seq_len: int | None = None) -> dict[str, int | float]:
        resolved_max_seq_len = (
            self.runtime_sequence_capacity()
            if max_seq_len is None
            else max(int(max_seq_len), 1)
        )
        return {
            "rope_head_dim": int(self.rope_head_dim),
            "max_seq_len": int(resolved_max_seq_len),
            "rope_theta": float(self.rope_theta),
            "original_seq_len": int(self.original_seq_len),
            "rope_factor": float(self.rope_factor),
            "beta_fast": int(self.beta_fast),
            "beta_slow": int(self.beta_slow),
        }
