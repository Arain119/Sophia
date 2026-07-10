from __future__ import annotations

from dataclasses import dataclass

import torch

from ml.modeling.text.conversation import ConversationMessage


@dataclass(frozen=True)
class ChatExample:
    messages: list[ConversationMessage]
    metadata: dict[str, object]


@dataclass(frozen=True)
class EncodedChatExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


__all__ = ["ChatExample", "EncodedChatExample"]
