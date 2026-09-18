from __future__ import annotations

import json
import re
from collections.abc import Mapping

from ml.modeling.text.conversation_types import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESULT_CLOSE,
    TOOL_RESULT_OPEN,
)


_CALL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


def normalize_tool_call(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError("tool call must be an object")
    call_id = str(raw.get("id") or "").strip()
    name = str(raw.get("name") or "").strip()
    arguments = raw.get("arguments", {})
    if not _CALL_ID.fullmatch(call_id):
        raise ValueError(f"invalid tool call id: {call_id!r}")
    if not _TOOL_NAME.fullmatch(name):
        raise ValueError(f"invalid tool name: {name!r}")
    if not isinstance(arguments, Mapping):
        raise TypeError("tool call arguments must be an object")
    return {"id": call_id, "name": name, "arguments": dict(arguments)}


def normalize_tool_result(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError("tool result must be an object")
    call_id = str(raw.get("tool_call_id") or "").strip()
    name = str(raw.get("name") or "").strip()
    if not _CALL_ID.fullmatch(call_id):
        raise ValueError(f"invalid tool result call id: {call_id!r}")
    if not _TOOL_NAME.fullmatch(name):
        raise ValueError(f"invalid tool result name: {name!r}")
    return {
        "tool_call_id": call_id,
        "name": name,
        "ok": bool(raw.get("ok", True)),
        "content": raw.get("content"),
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
    )


def render_tool_call(raw: object) -> str:
    return TOOL_CALL_OPEN + _canonical_json(normalize_tool_call(raw)) + TOOL_CALL_CLOSE


def render_tool_result(raw: object) -> str:
    return (
        TOOL_RESULT_OPEN
        + _canonical_json(normalize_tool_result(raw))
        + TOOL_RESULT_CLOSE
    )


def parse_tool_calls(text: str) -> tuple[str, tuple[dict[str, object], ...]]:
    source = str(text or "")
    decoder = json.JSONDecoder()
    calls: list[dict[str, object]] = []
    content_parts: list[str] = []
    cursor = 0
    while True:
        marker = source.find(TOOL_CALL_OPEN, cursor)
        if marker < 0:
            content_parts.append(source[cursor:])
            break
        content_parts.append(source[cursor:marker])
        json_start = marker + len(TOOL_CALL_OPEN)
        try:
            raw, json_end = decoder.raw_decode(source, json_start)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid tool call JSON at character {json_start}") from exc
        if not source.startswith(TOOL_CALL_CLOSE, json_end):
            raise ValueError("tool call JSON is not followed by the closing marker")
        calls.append(normalize_tool_call(raw))
        cursor = json_end + len(TOOL_CALL_CLOSE)
    return "".join(content_parts).strip(), tuple(calls)


def render_tool_catalog(tools: object) -> str:
    if tools is None:
        return ""
    if not isinstance(tools, list):
        raise TypeError("tools catalog must be a list")
    normalized: list[dict[str, object]] = []
    names: set[str] = set()
    for raw in tools:
        if not isinstance(raw, Mapping):
            raise TypeError("tool definitions must be objects")
        name = str(raw.get("name") or "").strip()
        if not _TOOL_NAME.fullmatch(name) or name in names:
            raise ValueError(f"invalid or duplicate tool definition name: {name!r}")
        parameters = raw.get("parameters")
        if not isinstance(parameters, Mapping):
            raise TypeError(f"tool {name!r} parameters must be a JSON Schema object")
        normalized.append(
            {
                "name": name,
                "description": str(raw.get("description") or ""),
                "parameters": dict(parameters),
            }
        )
        names.add(name)
    return "## Tools\n" + _canonical_json(normalized)


__all__ = [
    "normalize_tool_call",
    "normalize_tool_result",
    "parse_tool_calls",
    "render_tool_call",
    "render_tool_catalog",
    "render_tool_result",
]
