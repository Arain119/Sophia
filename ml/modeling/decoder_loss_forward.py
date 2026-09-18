from __future__ import annotations

import torch

from ml.modeling.input_mask import validate_right_padding_mask
from ml.modeling.text.loss_stats import mean_loss_from_sum_and_count
from ml.modeling.decoder_types import DecoderConfig
from ml.modeling.decoder_types import DecoderCoreModel
from .decoder_loss import chunked_loss_stats_from_hidden


def forward_loss(
    *,
    runtime_model: DecoderCoreModel,
    config: DecoderConfig,
    training: bool,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor,
    output_weight: torch.Tensor,
) -> torch.Tensor:
    del training
    ref = output_weight
    if attention_mask is not None:
        validate_right_padding_mask(attention_mask, input_ids=input_ids)
    hidden, _ = runtime_model._forward_hidden(input_ids, start_pos=0)
    base_sum, base_count = chunked_loss_stats_from_hidden(
        hidden,
        labels,
        label_offset=0,
        norm=runtime_model.norm,
        output=runtime_model.output,
        config=config,
    )

    return mean_loss_from_sum_and_count(
        loss_sum=base_sum,
        count=base_count,
        reference=ref,
    )


__all__ = ["forward_loss"]
