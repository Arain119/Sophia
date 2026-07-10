from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_LIST_LINE_RE = re.compile(
    r"^\s*(?:"
    r"[-*+]\s+"
    r"|\(?\d{1,3}\)?[.)]\s+"
    r"|[A-Za-z][.)]\s+"
    r"|[一二三四五六七八九十]+[、.]\s*"
    r"|第[一二三四五六七八九十0-9]+[点章节部分步]\s*"
    r")"
)
_HEADING_LINE_RE = re.compile(r"^\s*(?:#{1,6}\s+|\*\*[^*]{1,80}\*\*\s*$|__[^_]{1,80}__\s*$)")
_CODE_FENCE_RE = re.compile(r"^\s*```")
_TABLE_LINE_RE = re.compile(r"^\s*\|.+\|\s*$")
_JSONISH_RE = re.compile(r'^\s*[\[{].{0,1600}"[^"]+"\s*:', flags=re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"[。！？.!?]+")
_INLINE_ENUM_RE = re.compile(r"(?:^|[ \t])\d{1,2}[.)][ \t]+")

_ZH_TEMPLATE_OPENERS: tuple[str, ...] = (
    "以下是",
    "下面是",
    "可以从以下",
    "总结如下",
    "分为以下",
    "有几点",
    "几点建议",
    "先说结论",
)
_EN_TEMPLATE_OPENERS: tuple[str, ...] = (
    "here are",
    "below is",
    "below are",
    "the following",
    "to summarize",
    "in summary",
    "here's a breakdown",
)
_CONNECTIVES: tuple[str, ...] = (
    "但是",
    "不过",
    "同时",
    "另外",
    "因此",
    "所以",
    "如果",
    "而且",
    "however",
    "meanwhile",
    "therefore",
    "because",
    "if ",
    "at the same time",
)
_AI_DISCLAIMER_MARKERS: tuple[str, ...] = (
    "as an ai language model",
    "as a language model",
    "作为一个ai",
    "作为人工智能",
)
_BOTISH_MARKERS: tuple[str, ...] = (
    "i'd be happy to help",
    "i would be happy to help",
    "i hope this helps",
    "please let me know if you need anything else",
    "certainly",
    "certainly!",
    "of course",
    "if you need further assistance",
    "很高兴为你",
    "很高兴为您",
    "希望这能帮到你",
    "希望这能帮到您",
    "如果您还需要",
    "如需进一步",
    "以下是",
    "下面是",
)
_HUMANLIKE_MARKERS: tuple[str, ...] = (
    "i think",
    "maybe",
    "probably",
    "honestly",
    "if you want",
    "you can",
    "that's",
    "it's",
    "其实",
    "我觉得",
    "不一定",
    "可以啊",
    "如果你想",
    "说真的",
    "倒也",
    "未必",
)


@dataclass(frozen=True)
class ParagraphStyleDecision:
    keep: bool
    reason: str
    score: int
    chars: int
    paragraphs: int
    lines: int
    list_lines: int
    structured_lines: int
    code_fence_lines: int
    table_lines: int
    template_opener: bool


def _normalize_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]


def _template_opener(text: str) -> bool:
    first_line = (_lines(text) or [""])[0].lower()
    lowered = str(text or "").strip().lower()
    return any(first_line.startswith(item) or lowered.startswith(item) for item in _EN_TEMPLATE_OPENERS) or any(
        first_line.startswith(item) or str(text or "").strip().startswith(item)
        for item in _ZH_TEMPLATE_OPENERS
    )


def judge_paragraph_style(
    value: Any,
    *,
    min_chars: int = 80,
    max_chars: int = 4000,
    max_paragraphs: int = 4,
    max_structured_ratio: float = 0.34,
    allow_code_fences: bool = False,
    min_score: int = 2,
) -> ParagraphStyleDecision:
    text = _normalize_text(value)
    chars = len(text)
    if chars < int(min_chars):
        return ParagraphStyleDecision(False, "too_short", -10, chars, 0, 0, 0, 0, 0, 0, False)
    if chars > int(max_chars):
        return ParagraphStyleDecision(False, "too_long", -10, chars, 0, 0, 0, 0, 0, 0, False)
    if any(marker in text.lower() for marker in _AI_DISCLAIMER_MARKERS):
        return ParagraphStyleDecision(False, "ai_disclaimer", -8, chars, 0, 0, 0, 0, 0, 0, False)

    paragraphs = _paragraphs(text)
    lines = _lines(text)
    paragraph_count = len(paragraphs)
    line_count = len(lines)
    list_lines = sum(1 for line in lines if _LIST_LINE_RE.match(line))
    heading_lines = sum(1 for line in lines if _HEADING_LINE_RE.match(line))
    code_fence_lines = sum(1 for line in lines if _CODE_FENCE_RE.match(line))
    table_lines = sum(1 for line in lines if _TABLE_LINE_RE.match(line))
    structured_lines = int(list_lines + heading_lines + code_fence_lines + table_lines)
    template = _template_opener(text)
    json_like = bool(_JSONISH_RE.match(text))
    inline_enum_hits = len(_INLINE_ENUM_RE.findall(text))

    if not bool(allow_code_fences) and code_fence_lines > 0:
        return ParagraphStyleDecision(
            False,
            "code_fence",
            -7,
            chars,
            paragraph_count,
            line_count,
            list_lines,
            structured_lines,
            code_fence_lines,
            table_lines,
            template,
        )
    if json_like:
        return ParagraphStyleDecision(
            False,
            "json_like",
            -7,
            chars,
            paragraph_count,
            line_count,
            list_lines,
            structured_lines,
            code_fence_lines,
            table_lines,
            template,
        )
    if table_lines >= 2:
        return ParagraphStyleDecision(
            False,
            "table_like",
            -7,
            chars,
            paragraph_count,
            line_count,
            list_lines,
            structured_lines,
            code_fence_lines,
            table_lines,
            template,
        )
    if inline_enum_hits >= 3:
        return ParagraphStyleDecision(
            False,
            "inline_list_heavy",
            -6,
            chars,
            paragraph_count,
            line_count,
            list_lines,
            structured_lines,
            code_fence_lines,
            table_lines,
            template,
        )
    if paragraph_count > int(max_paragraphs):
        return ParagraphStyleDecision(
            False,
            "too_many_paragraphs",
            -4,
            chars,
            paragraph_count,
            line_count,
            list_lines,
            structured_lines,
            code_fence_lines,
            table_lines,
            template,
        )

    if line_count >= 4:
        structured_ratio = float(structured_lines) / float(max(line_count, 1))
        if list_lines >= 2 and structured_ratio >= float(max_structured_ratio):
            return ParagraphStyleDecision(
                False,
                "list_heavy",
                -6,
                chars,
                paragraph_count,
                line_count,
                list_lines,
                structured_lines,
                code_fence_lines,
                table_lines,
                template,
            )
        if structured_lines >= 3 and structured_ratio >= float(max_structured_ratio):
            return ParagraphStyleDecision(
                False,
                "structured_heavy",
                -5,
                chars,
                paragraph_count,
                line_count,
                list_lines,
                structured_lines,
                code_fence_lines,
                table_lines,
                template,
            )

    sentence_count = sum(1 for part in _SENTENCE_SPLIT_RE.split(text) if part.strip())
    short_lines = sum(1 for line in lines if len(line) <= 32)
    short_line_ratio = float(short_lines) / float(max(line_count, 1))
    has_connective = any(token in text.lower() for token in _CONNECTIVES) or any(
        token in text for token in _CONNECTIVES if not token.isascii()
    )

    score = 0
    if 1 <= paragraph_count <= 4:
        score += 2
    if 2 <= paragraph_count <= 4:
        score += 1
    if paragraph_count == 1 and chars >= max(int(min_chars), 120):
        score += 1
    if sentence_count >= 2:
        score += 1
    if has_connective:
        score += 1
    if template:
        score -= 1
    if list_lines > 0:
        score -= 2
    if line_count >= 4 and short_line_ratio > 0.5:
        score -= 1

    keep = int(score) >= int(min_score)
    reason = "ok" if keep else "low_paragraph_score"
    return ParagraphStyleDecision(
        keep=keep,
        reason=reason,
        score=int(score),
        chars=chars,
        paragraphs=paragraph_count,
        lines=line_count,
        list_lines=int(list_lines),
        structured_lines=int(structured_lines),
        code_fence_lines=int(code_fence_lines),
        table_lines=int(table_lines),
        template_opener=bool(template),
    )


def judge_humanlike_chat_style(
    value: Any,
    *,
    min_chars: int = 20,
    max_chars: int = 600,
    max_paragraphs: int = 3,
    max_sentences: int = 5,
    min_score: int = 1,
) -> ParagraphStyleDecision:
    text = _normalize_text(value)
    chars = len(text)
    if chars < int(min_chars):
        return ParagraphStyleDecision(False, "too_short", -10, chars, 0, 0, 0, 0, 0, 0, False)
    if chars > int(max_chars):
        return ParagraphStyleDecision(False, "too_long", -10, chars, 0, 0, 0, 0, 0, 0, False)
    lowered = text.lower()
    if any(marker in lowered for marker in _AI_DISCLAIMER_MARKERS):
        return ParagraphStyleDecision(False, "ai_disclaimer", -8, chars, 0, 0, 0, 0, 0, 0, False)

    paragraphs = _paragraphs(text)
    lines = _lines(text)
    paragraph_count = len(paragraphs)
    line_count = len(lines)
    list_lines = sum(1 for line in lines if _LIST_LINE_RE.match(line))
    heading_lines = sum(1 for line in lines if _HEADING_LINE_RE.match(line))
    code_fence_lines = sum(1 for line in lines if _CODE_FENCE_RE.match(line))
    table_lines = sum(1 for line in lines if _TABLE_LINE_RE.match(line))
    structured_lines = int(list_lines + heading_lines + code_fence_lines + table_lines)
    template = _template_opener(text)
    json_like = bool(_JSONISH_RE.match(text))
    inline_enum_hits = len(_INLINE_ENUM_RE.findall(text))

    if code_fence_lines > 0:
        return ParagraphStyleDecision(
            False,
            "code_fence",
            -7,
            chars,
            paragraph_count,
            line_count,
            list_lines,
            structured_lines,
            code_fence_lines,
            table_lines,
            template,
        )
    if json_like:
        return ParagraphStyleDecision(
            False,
            "json_like",
            -7,
            chars,
            paragraph_count,
            line_count,
            list_lines,
            structured_lines,
            code_fence_lines,
            table_lines,
            template,
        )
    if table_lines >= 1:
        return ParagraphStyleDecision(
            False,
            "table_like",
            -7,
            chars,
            paragraph_count,
            line_count,
            list_lines,
            structured_lines,
            code_fence_lines,
            table_lines,
            template,
        )
    if list_lines >= 1:
        return ParagraphStyleDecision(
            False,
            "list_heavy",
            -6,
            chars,
            paragraph_count,
            line_count,
            list_lines,
            structured_lines,
            code_fence_lines,
            table_lines,
            template,
        )
    if inline_enum_hits >= 2:
        return ParagraphStyleDecision(
            False,
            "inline_list_heavy",
            -6,
            chars,
            paragraph_count,
            line_count,
            list_lines,
            structured_lines,
            code_fence_lines,
            table_lines,
            template,
        )
    if paragraph_count > int(max_paragraphs):
        return ParagraphStyleDecision(
            False,
            "too_many_paragraphs",
            -4,
            chars,
            paragraph_count,
            line_count,
            list_lines,
            structured_lines,
            code_fence_lines,
            table_lines,
            template,
        )

    sentence_count = sum(1 for part in _SENTENCE_SPLIT_RE.split(text) if part.strip())
    if sentence_count > int(max_sentences):
        return ParagraphStyleDecision(
            False,
            "too_many_sentences",
            -4,
            chars,
            paragraph_count,
            line_count,
            list_lines,
            structured_lines,
            code_fence_lines,
            table_lines,
            template,
        )

    score = 0
    if 1 <= paragraph_count <= 2:
        score += 1
    if 1 <= sentence_count <= int(max_sentences):
        score += 1
    if chars <= 260:
        score += 1
    if any(marker in lowered for marker in _HUMANLIKE_MARKERS):
        score += 1
    if template:
        score -= 1
    if any(marker in lowered for marker in _BOTISH_MARKERS):
        score -= 2
    if text.endswith(":") or text.endswith("："):
        score -= 1

    keep = int(score) >= int(min_score)
    reason = "ok" if keep else "botish_or_stiff"
    return ParagraphStyleDecision(
        keep=keep,
        reason=reason,
        score=int(score),
        chars=chars,
        paragraphs=paragraph_count,
        lines=line_count,
        list_lines=int(list_lines),
        structured_lines=int(structured_lines),
        code_fence_lines=int(code_fence_lines),
        table_lines=int(table_lines),
        template_opener=bool(template),
    )
