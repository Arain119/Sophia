"""HF adapter support for config metadata and remote-code registration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Protocol, TypeVar

from ml.modeling.config import SophiaModelConfig
from ml.integrations.adapters.hf.remote_code import (
    HF_CAUSAL_LM_CLASS,
    HF_CONFIG_CLASS,
    HF_MODELING_MODULE,
    HF_RUNTIME_FILENAME,
    HF_SUPPORT_FILENAME,
)


class AutoMapConfig(Protocol):
    architectures: list[str]
    auto_map: dict[str, str]


class ConfiguredModel(Protocol):
    config: AutoMapConfig | None


class MetadataConfig(AutoMapConfig, Protocol):
    max_seq_len: int
    dim: int
    n_layers: int
    n_heads: int
    norm_eps: float
    dropout: float
    rope_head_dim: int
    head_dim: int
    rope_theta: float
    original_seq_len: int
    rope_factor: float
    beta_fast: int
    beta_slow: int
    max_position_embeddings: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    sliding_window: int
    rms_norm_eps: float
    attention_dropout: float
    loss_chunk_size: int
    z_loss_weight: float
    gradient_checkpointing_exclude_first: int
    gradient_checkpointing_exclude_last: int
    partial_rotary_factor: float
    qk_rope_head_dim: int
    rope_parameters: dict[str, object]
    return_logits_in_train: bool
    use_cache: bool


ConfigT = TypeVar("ConfigT")


def _to_dict_values(config: object) -> dict[str, object] | None:
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        values = to_dict()
        if isinstance(values, Mapping):
            return dict(values)
    return None


def apply_auto_map(config: AutoMapConfig) -> AutoMapConfig:
    config.architectures = [HF_CAUSAL_LM_CLASS]
    config.auto_map = {
        "AutoConfig": f"{HF_MODELING_MODULE}.{HF_CONFIG_CLASS}",
        "AutoModelForCausalLM": f"{HF_MODELING_MODULE}.{HF_CAUSAL_LM_CLASS}",
    }
    return config


def ensure_auto_map(model: ConfiguredModel) -> None:
    cfg = model.config
    if cfg is None:
        raise RuntimeError("Model has no config; unable to set HuggingFace auto_map.")
    apply_auto_map(cfg)


def apply_config_metadata(
    config: MetadataConfig,
    *,
    return_logits_in_train: bool,
    use_cache: bool,
    loss_chunk_size: int,
    z_loss_weight: float,
    gradient_checkpointing_exclude_first: int,
    gradient_checkpointing_exclude_last: int,
) -> None:
    config.max_position_embeddings = int(config.max_seq_len)
    config.hidden_size = int(config.dim)
    config.num_hidden_layers = int(config.n_layers)
    config.num_attention_heads = int(config.n_heads)
    config.sliding_window = int(config.max_seq_len)
    config.rms_norm_eps = float(config.norm_eps)
    config.attention_dropout = float(config.dropout)
    config.loss_chunk_size = max(int(loss_chunk_size), 0)
    config.z_loss_weight = max(float(z_loss_weight), 0.0)
    config.gradient_checkpointing_exclude_first = max(
        int(gradient_checkpointing_exclude_first),
        0,
    )
    config.gradient_checkpointing_exclude_last = max(
        int(gradient_checkpointing_exclude_last),
        0,
    )
    config.partial_rotary_factor = float(config.rope_head_dim) / float(config.head_dim)
    config.qk_rope_head_dim = int(config.rope_head_dim)
    rope_type = "yarn" if int(config.original_seq_len) > 0 else "default"
    config.rope_parameters = {
        "rope_type": rope_type,
        "rope_theta": float(config.rope_theta),
        "partial_rotary_factor": float(config.partial_rotary_factor),
        "original_max_position_embeddings": int(config.original_seq_len),
        "factor": float(config.rope_factor),
        "beta_fast": int(config.beta_fast),
        "beta_slow": int(config.beta_slow),
    }
    config.return_logits_in_train = bool(return_logits_in_train)
    config.use_cache = bool(use_cache)
    config.architectures = [HF_CAUSAL_LM_CLASS]


def build_config(
    config: object,
    *,
    config_cls: Callable[..., ConfigT] | None = None,
) -> ConfigT:
    if config_cls is None:
        from ml.integrations.adapters.hf.config import SophiaConfig

        config_cls = SophiaConfig

    if isinstance(config, Mapping):
        values = dict(config)
    elif is_dataclass(config):
        values = asdict(config)
    else:
        values = _to_dict_values(config)
        if values is None:
            canonical_fields = set(SophiaModelConfig.get_defaults())
            values = {
                key: getattr(config, key)
                for key in canonical_fields
                if hasattr(config, key)
            }

    return config_cls(**values)


__all__ = [
    "HF_CAUSAL_LM_CLASS",
    "HF_CONFIG_CLASS",
    "HF_MODELING_MODULE",
    "HF_RUNTIME_FILENAME",
    "HF_SUPPORT_FILENAME",
    "AutoMapConfig",
    "ConfiguredModel",
    "MetadataConfig",
    "apply_auto_map",
    "apply_config_metadata",
    "build_config",
    "ensure_auto_map",
]
