from __future__ import annotations

from transformers import PretrainedConfig

from ml.integrations.adapters.hf.support import (
    apply_config_metadata,
    build_config,
)
from ml.modeling.config import SophiaModelConfig
from ml.modeling.config_projection import build_runtime_model_args
from ml.modeling.runtime_backend import resolve_runtime_backend


_REMOVED_FIELDS = {
    "n_heads",
    "num_key_value_heads",
    "rope_head_dim",
    "rope_theta",
    "original_seq_len",
    "rope_factor",
    "beta_fast",
    "beta_slow",
    "use_qk_norm",
}


class SophiaConfig(PretrainedConfig):
    """Hugging Face adapter for the native Sophia Hybrid schema."""

    model_type = "sophia_hybrid"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"intermediate_size": "ffn_hidden"}

    def __init__(
        self,
        *,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        unk_token_id: int | None = None,
        tie_word_embeddings: bool = True,
        return_logits_in_train: bool = True,
        use_cache: bool = True,
        loss_chunk_size: int = 0,
        gradient_checkpointing_exclude_first: int = 0,
        gradient_checkpointing_exclude_last: int = 0,
        **kwargs: object,
    ) -> None:
        removed = sorted(_REMOVED_FIELDS.intersection(kwargs))
        if removed:
            raise ValueError(
                "legacy Sophia Transformer fields are not supported: "
                + ", ".join(removed)
            )
        defaults = SophiaModelConfig.get_defaults()
        aliases = {
            "hidden_size": "dim",
            "num_hidden_layers": "n_layers",
            "num_attention_heads": "num_heads",
            "rms_norm_eps": "norm_eps",
            "max_position_embeddings": "max_seq_len",
            "attention_dropout": "dropout",
        }
        canonical = dict(defaults)
        extras = dict(kwargs)
        for adapter_name, canonical_name in aliases.items():
            if adapter_name in extras:
                canonical[canonical_name] = extras.pop(adapter_name)
        for name in defaults:
            if name in extras:
                canonical[name] = extras.pop(name)
        canonical = SophiaModelConfig(**canonical).to_dict()

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            unk_token_id=unk_token_id,
            tie_word_embeddings=bool(tie_word_embeddings),
            return_logits_in_train=bool(return_logits_in_train),
            use_cache=bool(use_cache),
            loss_chunk_size=int(loss_chunk_size),
            gradient_checkpointing_exclude_first=int(
                gradient_checkpointing_exclude_first
            ),
            gradient_checkpointing_exclude_last=int(
                gradient_checkpointing_exclude_last
            ),
            **extras,
        )
        for name, value in canonical.items():
            setattr(self, name, value)
        self.hidden_size = int(self.dim)
        self.num_hidden_layers = int(self.n_layers)
        self.num_attention_heads = int(self.num_heads)
        self.max_position_embeddings = int(self.max_seq_len)
        self.rms_norm_eps = float(self.norm_eps)
        self.attention_dropout = float(self.dropout)
        apply_config_metadata(
            self,
            return_logits_in_train=bool(return_logits_in_train),
            use_cache=bool(use_cache),
            loss_chunk_size=int(loss_chunk_size),
            gradient_checkpointing_exclude_first=int(
                gradient_checkpointing_exclude_first
            ),
            gradient_checkpointing_exclude_last=int(
                gradient_checkpointing_exclude_last
            ),
        )

    def to_model_args(self, *, runtime_max_seq_len: int | None = None) -> object:
        return build_runtime_model_args(
            self,
            model_args_cls=resolve_runtime_backend().model_args_cls,
            runtime_max_seq_len=runtime_max_seq_len,
        )


__all__ = ["SophiaConfig", "apply_config_metadata", "build_config"]
