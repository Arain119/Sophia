from __future__ import annotations

from ml.modeling.text.conversation_blocks import (
    normalize_conversation_messages,
    normalize_user_blocks,
)
from ml.modeling.text.conversation_markup import render_response_format_prompt
from ml.modeling.text.conversation_types import (
    ASSISTANT_TOKEN,
    BOS_TOKEN,
    ConversationMessage,
    EOS_TEXT,
    LAST_REMINDER_TOKEN,
    RenderedSegment,
    USER_TOKEN,
    stringify_text,
)


def render_user_message(message: ConversationMessage) -> str:
    blocks = normalize_user_blocks(message)
    parts: list[str] = []
    for block in blocks:
        block_type = str(block.get("type", "")).strip().lower()
        if block_type == "text":
            parts.append(stringify_text(block.get("text")))
            continue
        parts.append(f"[Unsupported {block_type}]")
    return "\n\n".join(parts)


def assistant_payload(message: ConversationMessage) -> str:
    parts: list[str] = [stringify_text(message.get("content"))]
    if not bool(message.get("wo_eos")):
        parts.append(EOS_TEXT)
    return "".join(parts)


def transition_after_message(
    message: ConversationMessage,
    *,
    is_last: bool,
    next_role: str | None,
    add_generation_prompt: bool,
) -> str:
    need_transition = bool(is_last or next_role in {"assistant", "latest_reminder"})
    role = str(message.get("role", "")).strip()
    if need_transition and role in {"user", "developer"} and not (
        is_last and not bool(add_generation_prompt)
    ):
        return ASSISTANT_TOKEN
    return ""


def render_conversation_segments(
    messages: list[ConversationMessage],
    *,
    add_generation_prompt: bool,
) -> list[RenderedSegment]:
    prepared = normalize_conversation_messages(messages)
    segments: list[RenderedSegment] = [RenderedSegment(BOS_TOKEN, False)]

    for index, message in enumerate(prepared):
        role = str(message.get("role", "")).strip()
        is_last = index == (len(prepared) - 1)
        next_role = None if is_last else str(prepared[index + 1].get("role", "")).strip()

        if role in {"system", "developer"}:
            body = ""
            if role == "developer":
                body += USER_TOKEN
            body += stringify_text(message.get("content"))
            response_format = message.get("response_format")
            if response_format:
                body += "\n\n" + render_response_format_prompt(response_format)
            if body:
                segments.append(RenderedSegment(body, False))
        elif role == "user":
            segments.append(RenderedSegment(USER_TOKEN + render_user_message(message), False))
        elif role == "latest_reminder":
            segments.append(
                RenderedSegment(
                    LAST_REMINDER_TOKEN + stringify_text(message.get("content")),
                    False,
                )
            )
        elif role == "assistant":
            body = assistant_payload(message)
            if body:
                segments.append(RenderedSegment(body, True))
        else:
            raise ValueError(f"unsupported message role for rendering: {role!r}")

        transition = transition_after_message(
            message,
            is_last=is_last,
            next_role=next_role,
            add_generation_prompt=bool(add_generation_prompt),
        )
        if transition:
            segments.append(RenderedSegment(transition, False))

    return [segment for segment in segments if segment.text]


__all__ = ["render_conversation_segments", "render_user_message"]
