from __future__ import annotations


import torch

from ml.modeling.cache_decode import RuntimeCacheState, forward_cached_decode
from ml.modeling.input_mask import is_all_ones_mask
from ml.modeling.decoder_full import (
    forward_decoder_full,
    validate_decoder_inputs,
)
from ml.modeling.decoder_output import DecoderFormattedOutput


class _DecoderForwardBase[
    CacheInputT, CacheOutputT, RuntimeOutputT
]:
    def _resolve_runtime_return_dict(self, *, return_dict: bool | None) -> bool:
        raise NotImplementedError

    def _format_decoder_output(
        self,
        *,
        loss: torch.Tensor | None,
        logits: torch.Tensor | None,
        cache: CacheOutputT | None,
        return_dict: bool,
    ) -> RuntimeOutputT:
        raise NotImplementedError

    def _cache_state_from_cache(
        self,
        cache: CacheInputT,
    ) -> RuntimeCacheState | None:
        raise NotImplementedError

    def _cache_output_from_state(self, cache_state: RuntimeCacheState) -> CacheOutputT:
        raise NotImplementedError

    def _forward_full_decoder(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        compute_loss: bool,
        return_dict: bool,
    ) -> RuntimeOutputT:
        with self._model_runtime_lock:
            loss, logits = forward_decoder_full(
                runtime_model=self.model,
                config=self.config,
                training=bool(self.training),
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                compute_loss=bool(compute_loss),
                output_weight=self.model.output.weight,
            )
            return self._format_decoder_output(
                loss=loss,
                logits=logits,
                cache=None,
                return_dict=bool(return_dict),
            )

    def _forward_cached_decoder(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache: CacheInputT,
        start_pos: int | None,
        logits_to_keep: int | None,
        return_dict: bool,
    ) -> RuntimeOutputT:
        if attention_mask is not None and not is_all_ones_mask(attention_mask):
            raise ValueError("Sophia cached decode only supports unpadded prompts")
        with self._model_runtime_lock:
            logits, next_cache = forward_cached_decode(
                model=self.model,
                input_ids=input_ids,
                cache_state=self._cache_state_from_cache(cache),
                start_pos=start_pos,
                logits_to_keep=logits_to_keep,
            )
            return self._format_decoder_output(
                loss=None,
                logits=logits,
                cache=self._cache_output_from_state(next_cache),
                return_dict=bool(return_dict),
            )

    @staticmethod
    def _requires_full_path(
        *,
        training: bool,
        labels: torch.Tensor | None,
        use_cache: bool,
    ) -> bool:
        return labels is not None or bool(training) or not bool(use_cache)


class DecoderForwardMixin(
    _DecoderForwardBase[
        RuntimeCacheState | None,
        RuntimeCacheState,
        DecoderFormattedOutput,
    ]
):
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache: RuntimeCacheState | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = None,
        start_pos: int | None = None,
        compute_loss: bool = False,
        **_: object,
    ) -> DecoderFormattedOutput:
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
            cache=cache,
            start_pos=start_pos,
            logits_to_keep=logits_to_keep,
            return_dict=bool(resolved_return_dict),
        )


__all__ = ["DecoderForwardMixin"]
