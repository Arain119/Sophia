from __future__ import annotations

import torch

from ml.modeling.input_mask import is_all_ones_mask
from ml.modeling.input_mask import slice_valid_tokens
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
    if attention_mask is None or is_all_ones_mask(attention_mask):
        hidden, _ = runtime_model._forward_hidden(input_ids, start_pos=0)
        base_sum, base_count = chunked_loss_stats_from_hidden(
            hidden,
            labels,
            label_offset=0,
            norm=runtime_model.norm,
            output=runtime_model.output,
            config=config,
        )
    else:
        rows = slice_valid_tokens(input_ids, attention_mask)
        base_sum = ref.new_zeros(())
        base_count = ref.new_zeros(())
        for batch_index, (start, end, row_tokens) in enumerate(rows):
            row_labels = labels[batch_index : batch_index + 1, start:end]
            row_hidden, _ = runtime_model._forward_hidden(row_tokens, start_pos=0)
            row_base_sum, row_base_count = chunked_loss_stats_from_hidden(
                row_hidden,
                row_labels,
                label_offset=0,
                norm=runtime_model.norm,
                output=runtime_model.output,
                config=config,
            )
            base_sum = base_sum + row_base_sum
            base_count = base_count + row_base_count

    return mean_loss_from_sum_and_count(
        loss_sum=base_sum,
        count=base_count,
        reference=ref,
    )


__all__ = ["forward_loss"]
