from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import importlib

import torch
from torch import nn

from ml.modeling.text.loss_stats import mean_loss_from_sum_and_count
from ml.modeling.text.loss_stats import shifted_loss_sum_and_count
from ml.modeling.decoder_types import DecoderConfig

Z_LOSS_TOKEN_CHUNK_SIZE = 256


def z_loss_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    token_chunk_size: int = Z_LOSS_TOKEN_CHUNK_SIZE,
) -> torch.Tensor:
    flat_labels = labels.reshape(-1).to(dtype=torch.long)
    flat_logits = logits.reshape(-1, int(logits.size(-1)))
    keep = flat_labels != int(ignore_index)
    if not torch.any(keep):
        return flat_logits.new_zeros(())
    chunk_size = max(int(token_chunk_size), 1)
    total = flat_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, int(flat_logits.size(0)), chunk_size):
        end = min(start + chunk_size, int(flat_logits.size(0)))
        chunk_keep = keep[start:end]
        chunk_logits = flat_logits[start:end]
        log_z = torch.logsumexp(chunk_logits.float(), dim=-1)
        total = total + (log_z.square() * chunk_keep.to(dtype=torch.float32)).sum()
    return total.to(dtype=flat_logits.dtype)


def loss_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return shifted_loss_sum_and_count(
        logits,
        labels,
        label_offset=label_offset,
        ignore_index=-100,
    )


def mean_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_offset: int = 0,
) -> torch.Tensor:
    loss_sum, count = loss_stats(logits, labels, label_offset=label_offset)
    return mean_loss_from_sum_and_count(
        loss_sum=loss_sum,
        count=count,
        reference=logits,
    )


@lru_cache(maxsize=1)
def _liger_fused_linear_ce_func() -> Callable[..., torch.Tensor]:
    module_name = "liger_kernel.transformers.functional"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == "liger_kernel":
            raise RuntimeError(
                "Liger fused linear cross entropy requires liger_kernel with "
                "liger_fused_linear_cross_entropy"
            ) from exc
        raise
    fn = getattr(module, "liger_fused_linear_cross_entropy", None)
    if fn is None:
        raise RuntimeError(f"{module_name}.liger_fused_linear_cross_entropy is unavailable")
    return fn


def _liger_linear_ce_loss_chunk_stats(
    hidden_chunk: torch.Tensor,
    chunk_labels: torch.Tensor,
    *,
    norm: nn.Module,
    output: nn.Module,
    z_loss_weight_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = getattr(output, "weight", None)
    bias = getattr(output, "bias", None)
    if not torch.is_tensor(weight):
        raise TypeError("liger_linear_ce loss backend requires output.weight")

    flat_labels = chunk_labels.reshape(-1).to(dtype=torch.long)
    keep = flat_labels != -100
    count = keep.sum().to(dtype=torch.float32)
    if not torch.any(keep):
        return hidden_chunk.new_zeros((), dtype=torch.float32), count

    hidden_norm = norm(hidden_chunk).reshape(-1, int(hidden_chunk.size(-1)))
    loss_sum = _call_liger_fused_linear_ce(
        hidden_norm,
        weight,
        flat_labels,
        bias=bias,
        z_loss_weight_value=float(z_loss_weight_value),
    )
    if not torch.is_tensor(loss_sum):
        raise RuntimeError("liger_fused_linear_cross_entropy must return a tensor loss")
    return loss_sum.float(), count


@torch.compiler.disable(reason="Liger fused CE is an external fused CUDA kernel boundary")
def _call_liger_fused_linear_ce(
    hidden_norm: torch.Tensor,
    weight: torch.Tensor,
    flat_labels: torch.Tensor,
    *,
    bias: torch.Tensor | None,
    z_loss_weight_value: float,
) -> torch.Tensor:
    fn = _liger_fused_linear_ce_func()
    try:
        return fn(
            hidden_norm,
            weight,
            flat_labels,
            bias=bias,
            ignore_index=-100,
            reduction="sum",
            lse_square_scale=float(z_loss_weight_value),
        )
    except TypeError as exc:
        raise RuntimeError(
            "installed liger_fused_linear_cross_entropy has an unsupported signature"
        ) from exc


def _loss_chunk_stats(
    hidden_chunk: torch.Tensor,
    chunk_labels: torch.Tensor,
    *,
    norm: nn.Module,
    output: nn.Module,
    z_loss_weight_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _liger_linear_ce_loss_chunk_stats(
        hidden_chunk,
        chunk_labels,
        norm=norm,
        output=output,
        z_loss_weight_value=float(z_loss_weight_value),
    )


def chunked_loss_stats_from_hidden(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_offset: int,
    norm: nn.Module,
    output: nn.Module,
    config: DecoderConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_pred_tokens = min(
        max(int(hidden.size(1)) - 1, 0),
        max(int(labels.size(1)) - int(label_offset) - 1, 0),
    )
    zero = hidden.new_zeros(())
    if max_pred_tokens <= 0:
        return zero, zero

    loss_sum = zero
    count = zero
    chunk_size = max(int(config.loss_chunk_size), 0)
    z_weight = max(float(getattr(config, "z_loss_weight", 0.0) or 0.0), 0.0)
    if chunk_size <= 0:
        chunk_size = int(max_pred_tokens)
    target_base = int(label_offset) + 1
    for token_start in range(0, max_pred_tokens, chunk_size):
        token_end = min(token_start + chunk_size, max_pred_tokens)
        chunk_labels = labels[
            :,
            target_base + token_start : target_base + token_end,
        ].contiguous()
        chunk_loss_sum, chunk_count = _loss_chunk_stats(
            hidden[:, token_start:token_end],
            chunk_labels,
            norm=norm,
            output=output,
            z_loss_weight_value=float(z_weight),
        )
        loss_sum = loss_sum + chunk_loss_sum
        count = count + chunk_count
    return loss_sum, count


__all__ = [
    "Z_LOSS_TOKEN_CHUNK_SIZE",
    "chunked_loss_stats_from_hidden",
    "loss_stats",
    "mean_cross_entropy_loss",
    "z_loss_sum",
]
