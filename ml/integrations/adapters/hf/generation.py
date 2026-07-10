from __future__ import annotations

import torch

from ml.modeling.cache_decode import RuntimeCacheState
from ml.modeling.decoder_forward import _DecoderForwardBase
from ml.modeling.decoder_full import validate_decoder_inputs
from ml.integrations.adapters.hf.cache import (
    SophiaCache,
    cache_state_from_past_key_values,
)


class CausalLMForwardMixin(
    _DecoderForwardBase[SophiaCache | None, SophiaCache, object]
):
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: SophiaCache | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = None,
        start_pos: int | None = None,
        compute_loss: bool = False,
        **_: object,
    ) -> object:
        input_ids = validate_decoder_inputs(
            input_ids=input_ids,
            labels=labels,
            compute_loss=bool(compute_loss),
        )
        resolved_use_cache = bool(
            self.config.use_cache if use_cache is None else use_cache
        )
        resolved_return_dict = self._resolve_runtime_return_dict(
            return_dict=return_dict
        )
        if self._requires_full_path(
            training=bool(self.training),
            labels=labels,
            use_cache=resolved_use_cache,
        ):
            return self._forward_full_decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                compute_loss=bool(compute_loss),
                return_dict=bool(resolved_return_dict),
            )
        return self._forward_cached_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=past_key_values,
            start_pos=start_pos,
            logits_to_keep=logits_to_keep,
            return_dict=bool(resolved_return_dict),
        )


class GenerationCacheMixin:
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: SophiaCache | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        model_inputs.pop("cache_position", None)
        return model_inputs

    @staticmethod
    def _reorder_cache(
        past_key_values: SophiaCache | None,
        beam_idx: torch.Tensor,
    ) -> SophiaCache:
        del past_key_values, beam_idx
        raise NotImplementedError(
            "Sophia HF generate does not support beam cache reordering"
        )

    @staticmethod
    def _cache_state_from_cache(
        cache: SophiaCache | None,
    ) -> RuntimeCacheState | None:
        return cache_state_from_past_key_values(cache)

    def _cache_output_from_state(self, cache_state: RuntimeCacheState) -> SophiaCache:
        return SophiaCache(
            owner=self,
            cache=cache_state.cache,
            cache_pos=int(cache_state.cache_pos),
            batch_size=int(cache_state.batch_size),
        )


__all__ = ["CausalLMForwardMixin", "GenerationCacheMixin"]
