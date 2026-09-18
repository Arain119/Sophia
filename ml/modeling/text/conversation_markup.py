from __future__ import annotations

from ml.modeling.text.conversation_types import (
    AssistantCompletion,
    RESPONSE_FORMAT_TEMPLATE,
    STOP_TOKENS,
    to_json,
)
from ml.modeling.text.tool_protocol import parse_tool_calls


def trim_generation_stops(text: str) -> str:
    out = str(text or "")
    for token in STOP_TOKENS:
        idx = out.find(token)
        if idx >= 0:
            out = out[:idx]
    return out.strip()


def render_response_format_prompt(response_format: object) -> str:
    if not response_format:
        return ""
    return RESPONSE_FORMAT_TEMPLATE.format(schema=to_json(response_format))


def parse_assistant_completion(text: str) -> AssistantCompletion:
    content = trim_generation_stops(text)
    final_text, tool_calls = parse_tool_calls(content)
    return AssistantCompletion(
        content=final_text,
        raw_text=content,
        tool_calls=tool_calls,
    )


def extract_final_text(text: str) -> str:
    return parse_assistant_completion(text).content


__all__ = [
    "extract_final_text",
    "parse_assistant_completion",
    "render_response_format_prompt",
    "trim_generation_stops",
]
