from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class DecoderConfig(Protocol):
    loss_chunk_size: int
    return_logits_in_train: bool
    vocab_size: int


class _OutputProjection(Protocol):
    weight: torch.Tensor


class DecoderCoreModel(Protocol):
    head_mixer: nn.Module
    norm: nn.Module
    output: _OutputProjection
    gradient_checkpointing: bool

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward_full(self, input_ids: torch.Tensor) -> torch.Tensor: ...


__all__ = ["DecoderConfig", "DecoderCoreModel"]
