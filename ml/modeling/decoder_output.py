from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Protocol

import torch

from ml.modeling.cache_decode import RuntimeCacheState


@dataclass
class DecoderOutput:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    cache: RuntimeCacheState | None = None


DecoderLogitsOutput = tuple[torch.Tensor | None, torch.Tensor | None]
DecoderCachedOutput = tuple[torch.Tensor, RuntimeCacheState]
DecoderFormattedOutput = DecoderOutput | DecoderLogitsOutput | DecoderCachedOutput


class SupportsReturnDict(Protocol):
    return_dict: bool


def require_output_loss(output: object, *, context: str) -> torch.Tensor:
    loss: object | None
    if isinstance(output, Mapping):
        loss = output.get("loss")
    else:
        loss = getattr(output, "loss", None)
    if not torch.is_tensor(loss):
        raise RuntimeError(f"{context} returned loss=None")
    return loss


def require_output_logits(output: object, *, context: str) -> torch.Tensor:
    logits: object | None
    if isinstance(output, Mapping):
        logits = output.get("logits")
    else:
        logits = getattr(output, "logits", None)
    if not torch.is_tensor(logits):
        raise RuntimeError(f"{context} returned logits=None")
    return logits


def resolve_return_dict(
    *,
    config: SupportsReturnDict,
    return_dict: bool | None,
) -> bool:
    if return_dict is None:
        return bool(config.return_dict)
    return bool(return_dict)


def format_decoder_output(
    *,
    loss: torch.Tensor | None,
    logits: torch.Tensor | None,
    cache: RuntimeCacheState | None = None,
    return_dict: bool,
) -> DecoderFormattedOutput:
    if bool(return_dict):
        return DecoderOutput(
            loss=loss,
            logits=logits,
            cache=cache,
        )
    if cache is not None:
        if logits is None:
            raise ValueError("cached decode output requires logits")
        return logits, cache
    return loss, logits


def format_hf_causal_lm_output(
    *,
    output_cls: type,
    loss: torch.Tensor | None,
    logits: torch.Tensor | None,
    past_key_values: object = None,
    return_dict: bool,
) -> object:
    if bool(return_dict):
        return output_cls(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )
    if past_key_values is not None:
        if logits is None:
            raise ValueError("cached decode output requires logits")
        return logits, past_key_values
    if loss is not None:
        return loss, logits
    return (logits,)


__all__ = [
    "DecoderOutput",
    "DecoderFormattedOutput",
    "format_hf_causal_lm_output",
    "format_decoder_output",
    "require_output_logits",
    "require_output_loss",
    "resolve_return_dict",
]
