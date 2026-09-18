from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import os

import torch
from torch import nn

from ml.modeling.config import SophiaModelConfig
from ml.modeling.decoder_output import (
    DecoderFormattedOutput,
    DecoderOutput,
    format_decoder_output,
    resolve_return_dict,
)
from ml.modeling.decoder_forward import DecoderForwardMixin
from ml.modeling.decoder_host import DecoderHostMixin
from ml.modeling.cache_decode import (
    RuntimeCacheState,
    cache_state_from_runtime_cache,
    clone_runtime_cache_state,
)
from ml.modeling.decoder_runtime import (
    DecoderModelMixin,
    DecoderRecipeMixin,
    DecoderRuntimeMixin,
)
from ml.modeling.runtime_backend import resolve_runtime_backend


@dataclass
class SophiaDecoderConfig(SophiaModelConfig):
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    unk_token_id: int | None = None
    return_logits_in_train: bool = True
    return_dict: bool = True
    use_cache: bool = False
    tie_word_embeddings: bool = True
    loss_chunk_size: int = 0
    gradient_checkpointing_exclude_first: int = 0
    gradient_checkpointing_exclude_last: int = 0

    @classmethod
    def from_pretrained_mapping(
        cls,
        values: Mapping[str, object],
    ) -> SophiaDecoderConfig:
        raw = dict(values)
        aliases = {
            "hidden_size": "dim",
            "num_hidden_layers": "n_layers",
            "num_attention_heads": "num_heads",
            "rms_norm_eps": "norm_eps",
            "max_position_embeddings": "max_seq_len",
            "attention_dropout": "dropout",
        }
        metadata = {
            "architectures",
            "auto_map",
            "model_type",
            "sliding_window",
            "transformers_version",
        }
        adapter_fields = set(aliases) | metadata
        present_adapter_fields = adapter_fields.intersection(raw)
        if not present_adapter_fields:
            return cls.from_mapping(raw)
        missing = sorted(adapter_fields - set(raw))
        if missing:
            raise ValueError(
                "incomplete Sophia Hugging Face export metadata; missing keys: "
                + ", ".join(missing)
            )
        allowed = {field.name for field in fields(cls)} | adapter_fields
        unknown = sorted(str(key) for key in raw if key not in allowed)
        if unknown:
            raise ValueError(
                "Sophia exported config contains unknown keys: " + ", ".join(unknown)
            )

        for alias, canonical in aliases.items():
            if canonical not in raw or raw[alias] != raw[canonical]:
                raise ValueError(
                    f"Sophia exported config alias mismatch: {alias} != {canonical}"
                )
        if raw["sliding_window"] != raw["max_seq_len"]:
            raise ValueError(
                "Sophia exported config alias mismatch: sliding_window != max_seq_len"
            )

        from ml.integrations.adapters.hf.remote_code import (
            HF_CAUSAL_LM_CLASS,
            HF_CONFIG_CLASS,
            HF_MODELING_MODULE,
        )

        expected_auto_map = {
            "AutoConfig": f"{HF_MODELING_MODULE}.{HF_CONFIG_CLASS}",
            "AutoModelForCausalLM": f"{HF_MODELING_MODULE}.{HF_CAUSAL_LM_CLASS}",
        }
        if raw["model_type"] != "sophia_hybrid":
            raise ValueError("Sophia exported config has an invalid model_type")
        if raw["architectures"] != [HF_CAUSAL_LM_CLASS]:
            raise ValueError("Sophia exported config has invalid architectures")
        if raw["auto_map"] != expected_auto_map:
            raise ValueError("Sophia exported config has an invalid auto_map")
        if not isinstance(raw["transformers_version"], str) or not str(
            raw["transformers_version"]
        ).strip():
            raise ValueError("Sophia exported config has no transformers_version")

        canonical = {
            key: value for key, value in raw.items() if key not in adapter_fields
        }
        return cls.from_mapping(canonical)


class SophiaDecoder(
    DecoderModelMixin,
    DecoderRuntimeMixin,
    DecoderRecipeMixin,
    DecoderHostMixin,
    DecoderForwardMixin,
    nn.Module,
):
    """Core decoder for training and local evaluation."""

    def __init__(
        self,
        config: SophiaDecoderConfig | None = None,
        *,
        runtime_max_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SophiaDecoderConfig()
        self.gradient_checkpointing = False
        self._initialize_decoder_runtime(
            config=self.config,
            runtime_max_seq_len=runtime_max_seq_len,
            runtime_backend=resolve_runtime_backend(),
            gradient_checkpointing_enabled=False,
        )

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: object = None,
    ) -> None:
        del gradient_checkpointing_func
        self._sync_runtime_gradient_checkpointing(enable=bool(enable))

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True
        self._set_gradient_checkpointing(True)

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False
        self._set_gradient_checkpointing(False)

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        *args: object,
        **kwargs: object,
    ) -> None:
        del args
        safe_serialization = bool(kwargs.pop("safe_serialization", True))
        state_dict = kwargs.pop("state_dict", None)
        if kwargs:
            unexpected = ", ".join(sorted(str(key) for key in kwargs))
            raise TypeError(f"unexpected save_pretrained kwargs: {unexpected}")
        save_dir = os.path.abspath(str(save_directory))
        os.makedirs(save_dir, exist_ok=True)
        from ml.modeling.pretrained_bundle import (
            save_canonical_config_bundle,
            save_pretrained_state_dict,
        )
        save_canonical_config_bundle(self.config, save_directory=save_dir)

        export_state = (
            self.state_dict()
            if state_dict is None
            else {
                str(key): value.detach().cpu()
                for key, value in dict(state_dict).items()
            }
        )
        save_pretrained_state_dict(
            export_state,
            save_directory=save_dir,
            safe_serialization=bool(safe_serialization),
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *model_args: object,
        **kwargs: object,
    ) -> SophiaDecoder:
        del model_args
        base_dtype = kwargs.pop("dtype", torch.float32)
        device = torch.device(kwargs.pop("device", "cpu"))
        runtime_max_seq_len = kwargs.pop("runtime_max_seq_len", None)
        if kwargs:
            unexpected = ", ".join(sorted(str(key) for key in kwargs))
            raise TypeError(f"unexpected from_pretrained kwargs: {unexpected}")
        export_dir = os.path.abspath(str(pretrained_model_name_or_path))
        config = cls.load_config_from_pretrained(export_dir)
        model = cls(config, runtime_max_seq_len=runtime_max_seq_len)
        from ml.modeling.pretrained_bundle import load_pretrained_state_dict

        model.load_state_dict(load_pretrained_state_dict(export_dir), strict=True)
        return model.to(device=device, dtype=base_dtype)

    @classmethod
    def load_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
    ) -> SophiaDecoderConfig:
        from ml.modeling.pretrained_bundle import (
            load_canonical_config_from_pretrained,
        )

        return load_canonical_config_from_pretrained(
            pretrained_model_name_or_path,
            config_cls=SophiaDecoderConfig,
        )

    @staticmethod
    def _cache_state_from_cache(
        cache: RuntimeCacheState | None,
    ) -> RuntimeCacheState | None:
        return cache_state_from_runtime_cache(cache)

    def _cache_output_from_state(self, cache_state: RuntimeCacheState) -> RuntimeCacheState:
        return clone_runtime_cache_state(cache_state)

    def _resolve_runtime_return_dict(self, *, return_dict: bool | None) -> bool:
        return resolve_return_dict(
            config=self.config,
            return_dict=return_dict,
        )

    def _format_decoder_output(
        self,
        *,
        loss: torch.Tensor | None,
        logits: torch.Tensor | None,
        cache: RuntimeCacheState | None,
        return_dict: bool,
    ) -> DecoderFormattedOutput:
        return format_decoder_output(
            loss=loss,
            logits=logits,
            cache=cache,
            return_dict=return_dict,
        )


__all__ = ["SophiaDecoderConfig", "DecoderOutput", "SophiaDecoder"]
