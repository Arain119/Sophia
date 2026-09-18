"""
Sophia model implementation.

This file contains:
- `SophiaForCausalLM`: Hugging Face causal language model adapter over the Sophia runtime.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ml.modeling.runtime_backend import resolve_runtime_backend
from ml.integrations.adapters.hf.cache import SophiaCache
from ml.integrations.adapters.hf.generation import (
    CausalLMForwardMixin,
    GenerationCacheMixin,
)
from ml.integrations.adapters.hf.lifecycle import PreTrainedLifecycleMixin
from ml.integrations.adapters.hf.config import SophiaConfig
from ml.modeling.decoder_output import (
    format_hf_causal_lm_output as _format_hf_causal_lm_output,
    resolve_return_dict as _resolve_return_dict,
)
from ml.modeling.decoder_host import DecoderHostMixin
from ml.modeling.decoder_runtime import (
    DecoderModelMixin,
    DecoderRecipeMixin,
    DecoderRuntimeMixin,
)


class SophiaForCausalLM(
    DecoderModelMixin,
    DecoderRuntimeMixin,
    DecoderRecipeMixin,
    DecoderHostMixin,
    CausalLMForwardMixin,
    PreTrainedLifecycleMixin,
    GenerationCacheMixin,
    PreTrainedModel,
    GenerationMixin,
):
    config_class = SophiaConfig
    base_model_prefix = "model"
    _tied_weights_keys = {"model.output.weight": "model.tok_embeddings.weight"}
    supports_gradient_checkpointing = True
    _is_stateful = True

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    def __init__(
        self,
        config: SophiaConfig | None = None,
        *,
        runtime_max_seq_len: int | None = None,
    ):
        config = config or SophiaConfig()
        super().__init__(config)
        self._initialize_decoder_runtime(
            config=config,
            runtime_max_seq_len=runtime_max_seq_len,
            runtime_backend=resolve_runtime_backend(),
            gradient_checkpointing_enabled=bool(
                getattr(self, "gradient_checkpointing", False)
            ),
            register_tied_weights=True,
        )

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: object = None,
    ) -> None:
        del gradient_checkpointing_func
        self.gradient_checkpointing = bool(enable)
        self._sync_runtime_gradient_checkpointing(enable=bool(enable))

    def _resolve_runtime_return_dict(self, *, return_dict: bool | None) -> bool:
        return _resolve_return_dict(
            config=self.config,
            return_dict=return_dict,
        )

    def _format_decoder_output(
        self,
        *,
        loss: torch.Tensor | None,
        logits: torch.Tensor | None,
        cache: SophiaCache | None,
        return_dict: bool,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...]:
        return _format_hf_causal_lm_output(
            output_cls=CausalLMOutputWithPast,
            loss=loss,
            logits=logits,
            past_key_values=cache,
            return_dict=return_dict,
        )
