from __future__ import annotations

from dataclasses import dataclass

from ml.core.spec.semantics import canonicalize_model_values, validate_model_values


@dataclass
class ModelArgs:
    """Torch-free runtime configuration for Sophia Hybrid."""

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
    use_cache: bool = True

    def __post_init__(self) -> None:
        normalized = canonicalize_model_values(vars(self))
        for name, value in normalized.items():
            if hasattr(self, name):
                setattr(self, name, value)
        validate_model_values(normalized)

    @property
    def attention_inner_dim(self) -> int:
        return int(self.num_heads) * int(self.head_dim)

    def layer_type(self, layer_idx: int) -> str:
        return "mla" if int(layer_idx) % 4 == 3 else "kda"

    def ensure_runtime_batch_capacity(self, batch_size: int) -> int:
        required = int(batch_size)
        if required <= 0:
            raise ValueError(f"batch_size must be > 0, got {required}")
        self.max_batch_size = max(int(self.max_batch_size), required)
        return int(self.max_batch_size)

    def runtime_batch_capacity(self) -> int:
        return int(self.max_batch_size)

    def ensure_runtime_sequence_capacity(self, max_seq_len: int) -> int:
        required = int(max_seq_len)
        if required <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {required}")
        configured = int(self.max_seq_len)
        if required > configured:
            raise ValueError(
                "runtime sequence capacity cannot exceed configured max_seq_len: "
                f"required={required} configured={configured}"
            )
        return configured

    def runtime_sequence_capacity(self) -> int:
        return int(self.max_seq_len)

    def runtime_max_seq_len(self) -> int:
        return self.runtime_sequence_capacity()

    def runtime_layer_count(self) -> int:
        return int(self.n_layers)
