from __future__ import annotations

import torch

from ml.modeling.input_mask import is_all_ones_mask, slice_valid_tokens
from ml.modeling.decoder_types import DecoderConfig, DecoderCoreModel
from ml.modeling.decoder_loss import (
    loss_stats,
    mean_cross_entropy_loss,
)
from ml.modeling.decoder_loss_forward import forward_loss


def validate_decoder_inputs(
    *,
    input_ids: torch.Tensor | None,
    labels: torch.Tensor | None,
    compute_loss: bool,
) -> torch.Tensor:
    if input_ids is None:
        raise ValueError("input_ids is required")
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be [B,T]")
    if bool(compute_loss) and labels is None:
        raise ValueError("compute_loss=True requires labels")
    return input_ids


def forward_full_with_mask(
    *,
    runtime_model: DecoderCoreModel,
    vocab_size: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    output_weight: torch.Tensor,
) -> torch.Tensor:
    if attention_mask is None or is_all_ones_mask(attention_mask):
        return runtime_model.forward_full(input_ids)

    rows = slice_valid_tokens(input_ids, attention_mask)
    logits = output_weight.new_zeros(
        (int(input_ids.size(0)), int(input_ids.size(1)), int(vocab_size))
    )
    for batch_index, (start, end, row_tokens) in enumerate(rows):
        row_logits = runtime_model.forward_full(row_tokens)
        logits[batch_index, start:end] = row_logits[0]
    return logits


def masked_loss(
    *,
    runtime_model: DecoderCoreModel,
    config: DecoderConfig,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    output_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = slice_valid_tokens(input_ids, attention_mask)
    logits = output_weight.new_zeros(
        (int(input_ids.size(0)), int(input_ids.size(1)), int(vocab_size))
    )
    loss_sum = output_weight.new_zeros(())
    count = output_weight.new_zeros(())
    for batch_index, (start, end, row_tokens) in enumerate(rows):
        row_labels = labels[batch_index : batch_index + 1, start:end]
        row_logits = runtime_model.forward_full(row_tokens)
        logits[batch_index, start:end] = row_logits[0]
        row_sum, row_count = loss_stats(row_logits, row_labels, label_offset=0)
        loss_sum = loss_sum + row_sum
        count = count + row_count
    loss = torch.where(
        count > 0,
        loss_sum / count.clamp_min(1.0),
        output_weight.new_zeros(()),
    )
    return logits, loss


def forward_decoder_full(
    *,
    runtime_model: DecoderCoreModel,
    config: DecoderConfig,
    training: bool,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor | None,
    compute_loss: bool,
    output_weight: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if bool(compute_loss):
        loss = forward_loss(
            runtime_model=runtime_model,
            config=config,
            training=bool(training),
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_weight=output_weight,
        )
        return loss, None
    loss, logits = forward_full(
        runtime_model=runtime_model,
        config=config,
        training=bool(training),
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    if labels is not None and bool(training) and not bool(config.return_logits_in_train):
        logits = None
    return loss, logits


def forward_full(
    *,
    runtime_model: DecoderCoreModel,
    config: DecoderConfig,
    training: bool,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    del training
    if labels is not None:
        if attention_mask is not None and not is_all_ones_mask(attention_mask):
            logits, loss = masked_loss(
                runtime_model=runtime_model,
                config=config,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                vocab_size=int(getattr(config, "vocab_size", 0)),
                output_weight=runtime_model.output.weight,
            )
            return loss, logits

        logits = forward_full_with_mask(
            runtime_model=runtime_model,
            vocab_size=int(config.vocab_size),
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_weight=runtime_model.output.weight,
        )
        return mean_cross_entropy_loss(logits, labels, label_offset=0), logits

    logits = forward_full_with_mask(
        runtime_model=runtime_model,
        vocab_size=int(config.vocab_size),
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_weight=runtime_model.output.weight,
    )
    return None, logits


__all__ = [
    "forward_full_with_mask",
    "forward_decoder_full",
    "forward_full",
    "masked_loss",
    "validate_decoder_inputs",
]
