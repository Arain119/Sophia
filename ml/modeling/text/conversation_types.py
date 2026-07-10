from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import torch

BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TEXT = "<｜end▁of▁sentence｜>"
USER_TOKEN = "<｜User｜>"
ASSISTANT_TOKEN = "<｜Assistant｜>"
LAST_REMINDER_TOKEN = "<｜latest_reminder｜>"

RESPONSE_FORMAT_TEMPLATE = (
    "## Response Format:\n\n"
    "You MUST strictly adhere to the following schema to reply:\n"
    "{schema}"
)

STOP_TOKENS = (EOS_TEXT,)

ConversationMessage = dict[str, object]
UserContentBlock = dict[str, object]


class SupportsEncodeText(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class AssistantCompletion:
    content: str
    raw_text: str


@dataclass(frozen=True)
class RenderedSegment:
    text: str
    supervise: bool


@dataclass(frozen=True)
class ConversationEncoding:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def stringify_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    raise TypeError(f"message content must be a string or null, got {type(value)!r}")


def to_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except UnicodeEncodeError:
        return json.dumps(value, ensure_ascii=True)


__all__ = [
    "ASSISTANT_TOKEN",
    "AssistantCompletion",
    "BOS_TOKEN",
    "ConversationEncoding",
    "ConversationMessage",
    "EOS_TEXT",
    "LAST_REMINDER_TOKEN",
    "RenderedSegment",
    "STOP_TOKENS",
    "SupportsEncodeText",
    "USER_TOKEN",
    "UserContentBlock",
    "stringify_text",
    "to_json",
]
