from __future__ import annotations

from dataclasses import dataclass
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
    z_loss_weight: float = 1e-4
    gradient_checkpointing_exclude_first: int = 0
    gradient_checkpointing_exclude_last: int = 0


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
        model._prefix_cache_namespace = f"{cls.__module__}.{cls.__qualname__}:{export_dir}"
        state_dict = cls._load_pretrained_state_dict(export_dir)
        if state_dict is not None:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "Sophia pretrained load produced key mismatches.\n"
                    f"missing={sorted(missing)}\n"
                    f"unexpected={sorted(unexpected)}"
                )
        return model.to(device=device, dtype=base_dtype)

    @staticmethod
    def _load_pretrained_state_dict(
        export_dir: str | os.PathLike,
    ) -> dict[str, torch.Tensor] | None:
        from ml.modeling.pretrained_bundle import load_pretrained_state_dict

        return load_pretrained_state_dict(export_dir)

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
