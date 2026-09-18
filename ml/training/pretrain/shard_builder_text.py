from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from ml.data.jsonl_stream import JsonlFieldStats, iter_jsonl_field


def _iter_jsonl_field(
    path: str,
    *,
    stats: JsonlFieldStats | None = None,
) -> Iterator[str]:
    yield from iter_jsonl_field(
        path,
        "text",
        stats=stats,
        strict_json=True,
        max_json_errors=0,
    )


_PARA_SPLIT_RE = re.compile(r"\n{2,}")
_SENT_BOUNDARY_RE = re.compile(r"([。！？.!?]+[\s]*)")


def _iter_text_chunks(text: str, *, max_chars: int, min_chars: int) -> Iterator[str]:
    t = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not t:
        return
    max_chars = int(max_chars)
    min_chars = int(min_chars)
    if max_chars <= 0 or len(t) <= max_chars:
        yield t
        return

    def flush(buf: list[str]) -> str | None:
        if not buf:
            return None
        s = "\n\n".join(x for x in buf if x).strip()
        return s if len(s) >= min_chars else None

    def split_long_segment(seg: str) -> Iterator[str]:
        seg = seg.strip()
        if not seg:
            return
        if len(seg) <= max_chars:
            if len(seg) >= min_chars:
                yield seg
            return

        parts = _SENT_BOUNDARY_RE.split(seg)
        units: list[str] = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and _SENT_BOUNDARY_RE.fullmatch(parts[i + 1] or ""):
                units.append((parts[i] + parts[i + 1]).strip())
                i += 2
            else:
                units.append(str(parts[i]).strip())
                i += 1
        units = [u for u in units if u]
        if not units:
            for j in range(0, len(seg), max_chars):
                chunk = seg[j : j + max_chars].strip()
                if len(chunk) >= min_chars:
                    yield chunk
            return

        buf: list[str] = []
        cur = 0
        for u in units:
            ulen = len(u)
            add = ulen + (1 if buf else 0)
            if buf and cur + add > max_chars:
                out = flush(buf)
                if out is not None:
                    yield out
                buf = [u]
                cur = ulen
            else:
                buf.append(u)
                cur += add
        out = flush(buf)
        if out is not None:
            yield out

    paragraphs = [p.strip() for p in _PARA_SPLIT_RE.split(t) if p.strip()]
    buf: list[str] = []
    cur = 0
    for p in paragraphs:
        if len(p) > max_chars:
            out = flush(buf)
            if out is not None:
                yield out
            buf = []
            cur = 0
            yield from split_long_segment(p)
            continue

        add = len(p) + (2 if buf else 0)
        if buf and cur + add > max_chars:
            out = flush(buf)
            if out is not None:
                yield out
            buf = [p]
            cur = len(p)
        else:
            buf.append(p)
            cur += add
    out = flush(buf)
    if out is not None:
        yield out


class ShardBuilderTextConfig(Protocol):
    jsonl: tuple[str, ...]
    max_doc_chars: int
    min_chunk_chars: int


@dataclass
class JsonlTextStream:
    config: ShardBuilderTextConfig
    stats: JsonlFieldStats
    emitted_docs: int = 0

    def iter_texts(self) -> Iterator[str]:
        max_doc_chars = max(int(self.config.max_doc_chars), 0)
        min_chunk_chars = max(int(self.config.min_chunk_chars), 0)
        for path in self.config.jsonl:
            for text in _iter_jsonl_field(
                str(path),
                stats=self.stats,
            ):
                if max_doc_chars > 0:
                    emitted = False
                    for chunk in _iter_text_chunks(
                        text,
                        max_chars=max_doc_chars,
                        min_chars=min_chunk_chars,
                    ):
                        emitted = True
                        self.emitted_docs += 1
                        yield chunk
                    if not emitted:
                        self.stats.empty_after_strip += 1
                    continue
                stripped = str(text or "").strip()
                if stripped:
                    self.emitted_docs += 1
                    yield stripped
                else:
                    self.stats.empty_after_strip += 1

__all__ = [
    "JsonlTextStream",
    "ShardBuilderTextConfig",
    "_iter_jsonl_field",
    "_iter_text_chunks",
]
