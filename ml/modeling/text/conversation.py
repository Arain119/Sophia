from __future__ import annotations

from ml.modeling.text.conversation_encode import (
    encode_conversation,
    encode_rendered_segments,
)
from ml.modeling.text.conversation_blocks import normalize_conversation_messages
from ml.modeling.text.conversation_markup import (
    extract_final_text,
    parse_assistant_completion,
    render_response_format_prompt,
    trim_generation_stops,
)
from ml.modeling.text.conversation_renderer import (
    render_conversation_segments,
    render_user_message,
)
from ml.modeling.text.conversation_types import (
    ASSISTANT_TOKEN,
    AssistantCompletion,
    BOS_TOKEN,
    ConversationEncoding,
    ConversationMessage,
    EOS_TEXT,
    LAST_REMINDER_TOKEN,
    RenderedSegment,
    SupportsEncodeText,
    USER_TOKEN,
    UserContentBlock,
)

__all__ = [
    "ASSISTANT_TOKEN",
    "AssistantCompletion",
    "BOS_TOKEN",
    "ConversationEncoding",
    "ConversationMessage",
    "EOS_TEXT",
    "LAST_REMINDER_TOKEN",
    "RenderedSegment",
    "SupportsEncodeText",
    "USER_TOKEN",
    "UserContentBlock",
    "encode_conversation",
    "encode_rendered_segments",
    "extract_final_text",
    "normalize_conversation_messages",
    "parse_assistant_completion",
    "render_conversation_segments",
    "render_response_format_prompt",
    "render_user_message",
    "trim_generation_stops",
]
