from __future__ import annotations

import re


_HUMAN_BOT_TAG_RE = re.compile(r"<\s*/?\s*(human|bot)\b", re.IGNORECASE)
_AI_MENTION_RE = re.compile(
    r"(?:作为|身为|我是).{0,24}?(?:AI|人工智能|大语言模型|语言模型|智能助手)"
    r"|(?:\b(?:i\s*(?:am|'m)|as\s+an?)\s+(?:ai|artificial\s+intelligence|llm|(?:large\s+)?language\s+model)\b)",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(
    r"(?:TODO|TBA|<\.\.\.>|<\s*\.\.\.\s*>|\[\.\.\.\]|\{\.{3}\})",
    re.IGNORECASE,
)

_THINK_TAG_RE = re.compile(r"<\s*/?\s*think\s*>", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(
    r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL
)


def has_human_bot_tags(text: str) -> bool:
    return bool(_HUMAN_BOT_TAG_RE.search(str(text or "")))


def mentions_ai(text: str) -> bool:
    return bool(_AI_MENTION_RE.search(str(text or "")))


def has_placeholders(text: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(str(text or "")))


def strip_think_blocks(text: str) -> str:
    s = str(text or "")
    s = _THINK_BLOCK_RE.sub("", s)
    s = _THINK_TAG_RE.sub("", s)
    return s.strip()


def has_emoji(text: str) -> bool:
    for ch in str(text or ""):
        codepoint = ord(ch)
        if 0x1F1E6 <= codepoint <= 0x1F1FF:
            return True
        if 0x1F300 <= codepoint <= 0x1FAFF:
            return True
        if 0x2600 <= codepoint <= 0x26FF:
            return True
        if 0x2700 <= codepoint <= 0x27BF:
            return True
        if 0xFE00 <= codepoint <= 0xFE0F:
            return True
    return False
