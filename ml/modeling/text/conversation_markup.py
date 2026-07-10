from __future__ import annotations

from ml.modeling.text.conversation_types import (
    AssistantCompletion,
    RESPONSE_FORMAT_TEMPLATE,
    STOP_TOKENS,
    to_json,
)


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
    return AssistantCompletion(content=content, raw_text=content)


def extract_final_text(text: str) -> str:
    return trim_generation_stops(text)


__all__ = [
    "extract_final_text",
    "parse_assistant_completion",
    "render_response_format_prompt",
    "trim_generation_stops",
]
