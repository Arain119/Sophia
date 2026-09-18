from __future__ import annotations

import copy

from ml.modeling.text.conversation_types import ConversationMessage, UserContentBlock, stringify_text


def normalize_user_blocks(message: ConversationMessage) -> list[UserContentBlock]:
    blocks = message.get("content_blocks")
    if blocks is not None:
        if not isinstance(blocks, list):
            raise TypeError("user.content_blocks must be a list when provided")
        out: list[UserContentBlock] = []
        for raw in blocks:
            if not isinstance(raw, dict):
                raise TypeError("user.content_blocks entries must be objects")
            block_type = str(raw.get("type", "")).strip().lower()
            if block_type == "text":
                out.append({"type": "text", "text": stringify_text(raw.get("text"))})
                continue
            raise ValueError(f"unsupported content block type: {block_type!r}")
        return out

    content = stringify_text(message.get("content"))
    if not content:
        return []
    return [{"type": "text", "text": content}]


def normalize_conversation_messages(
    messages: list[ConversationMessage],
) -> list[ConversationMessage]:
    merged: list[ConversationMessage] = []
    for raw_message in messages:
        message = copy.deepcopy(raw_message)
        role = str(message.get("role", "")).strip()
        if role == "user":
            content_blocks = normalize_user_blocks(message)
            if not content_blocks:
                content_blocks = [
                    {"type": "text", "text": stringify_text(message.get("content"))}
                ]
            control_keys = ("wo_eos", "mask", "response_format")
            has_controls = any(key in message for key in control_keys)
            if (
                merged
                and str(merged[-1].get("role", "")).strip() == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
                and not has_controls
            ):
                merged[-1]["content_blocks"].extend(content_blocks)
            else:
                new_message: ConversationMessage = {
                    "role": "user",
                    "content": stringify_text(message.get("content")),
                    "content_blocks": list(content_blocks),
                }
                for key in control_keys:
                    if key in message:
                        new_message[key] = message[key]
                merged.append(new_message)
            continue
        merged.append(message)
    return merged


__all__ = [
    "normalize_conversation_messages",
    "normalize_user_blocks",
]
