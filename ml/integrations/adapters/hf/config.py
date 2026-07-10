from __future__ import annotations


from transformers import PretrainedConfig

from ml.modeling.config import SophiaModelConfig
from ml.modeling.config_projection import (
    build_runtime_model_args as _build_runtime_model_args,
    canonicalize_canonical_config_values as _canonicalize_canonical_config_values,
    rope_head_dim_from_partial_factor,
)
from ml.integrations.adapters.hf.support import (
    apply_config_metadata as _apply_config_metadata,
    apply_config_metadata,
    build_config,
)
from ml.modeling.runtime_backend import resolve_runtime_backend


def _canonicalize_config_values(
    values: dict[str, object],
    *,
    explicit_keys: set[str],
) -> dict[str, object]:
    normalized = dict(values)
    field_map = {
        "hidden_size": "dim",
        "num_hidden_layers": "n_layers",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "num_key_value_heads",
        "rms_norm_eps": "norm_eps",
        "max_position_embeddings": "max_seq_len",
        "attention_dropout": "dropout",
    }
    for adapter_name, canonical_name in field_map.items():
        if adapter_name not in explicit_keys:
            continue
        if (
            canonical_name in explicit_keys
            and normalized.get(canonical_name) != normalized.get(adapter_name)
        ):
            raise ValueError(
                f"{adapter_name} and {canonical_name} were both provided with different values"
            )
        if canonical_name not in explicit_keys:
            normalized[canonical_name] = normalized[adapter_name]

    rope_parameters = normalized.get("rope_parameters")
    if isinstance(rope_parameters, dict):
        if "rope_theta" in rope_parameters and "rope_theta" not in normalized:
            normalized["rope_theta"] = rope_parameters["rope_theta"]
        if "partial_rotary_factor" in rope_parameters and "partial_rotary_factor" not in normalized:
            normalized["partial_rotary_factor"] = rope_parameters["partial_rotary_factor"]
        if (
            "original_max_position_embeddings" in rope_parameters
            and "original_seq_len" not in normalized
        ):
            normalized["original_seq_len"] = rope_parameters[
                "original_max_position_embeddings"
            ]
        if "factor" in rope_parameters and "rope_factor" not in normalized:
            normalized["rope_factor"] = rope_parameters["factor"]
        if "beta_fast" in rope_parameters and "beta_fast" not in normalized:
            normalized["beta_fast"] = rope_parameters["beta_fast"]
        if "beta_slow" in rope_parameters and "beta_slow" not in normalized:
            normalized["beta_slow"] = rope_parameters["beta_slow"]

    partial_rotary_factor = normalized.get("partial_rotary_factor")
    if "partial_rotary_factor" in explicit_keys and partial_rotary_factor is not None:
        rope_head_dim = rope_head_dim_from_partial_factor(
            head_dim=int(normalized.get("head_dim", 0) or 0),
            partial_rotary_factor=float(partial_rotary_factor),
        )
        if (
            "rope_head_dim" in explicit_keys
            and int(normalized["rope_head_dim"]) != int(rope_head_dim)
        ):
            raise ValueError("partial_rotary_factor and rope_head_dim are inconsistent")
        normalized["rope_head_dim"] = int(rope_head_dim)

    return normalized


class SophiaConfig(PretrainedConfig):
    """Hugging Face configuration adapter for the Sophia runtime."""

    model_type = "sophia"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"intermediate_size": "ffn_hidden"}

    @staticmethod
    def _canonicalize_canonical_config_values(
        values: dict[str, object],
    ) -> dict[str, object]:
        return _canonicalize_canonical_config_values(values)

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
        z_loss_weight: float = 1e-4,
        gradient_checkpointing_exclude_first: int = 0,
        gradient_checkpointing_exclude_last: int = 0,
        **kwargs: object,
    ) -> None:
        defaults = SophiaModelConfig.get_defaults()
        explicit_keys = set(kwargs)

        merged = {**defaults, **kwargs}
        merged.update(_canonicalize_config_values(merged, explicit_keys=explicit_keys))
        merged.update(self._canonicalize_canonical_config_values(merged))

        merged["tie_word_embeddings"] = bool(tie_word_embeddings)
        tie_word_embeddings = bool(merged.pop("tie_word_embeddings", True))

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            unk_token_id=unk_token_id,
            tie_word_embeddings=tie_word_embeddings,
            return_logits_in_train=bool(return_logits_in_train),
            use_cache=bool(use_cache),
            loss_chunk_size=int(loss_chunk_size),
            z_loss_weight=float(z_loss_weight),
            gradient_checkpointing_exclude_first=int(gradient_checkpointing_exclude_first),
            gradient_checkpointing_exclude_last=int(gradient_checkpointing_exclude_last),
            **merged,
        )

        for key, value in merged.items():
            setattr(self, key, value)

        _apply_config_metadata(
            self,
            return_logits_in_train=bool(return_logits_in_train),
            use_cache=bool(use_cache),
            loss_chunk_size=int(loss_chunk_size),
            z_loss_weight=float(z_loss_weight),
            gradient_checkpointing_exclude_first=int(gradient_checkpointing_exclude_first),
            gradient_checkpointing_exclude_last=int(gradient_checkpointing_exclude_last),
        )

    def to_model_args(
        self,
        *,
        runtime_max_seq_len: int | None = None,
    ) -> object:
        return _build_runtime_model_args(
            self,
            model_args_cls=resolve_runtime_backend().model_args_cls,
            runtime_max_seq_len=runtime_max_seq_len,
        )


__all__ = ["SophiaConfig", "apply_config_metadata", "build_config"]
