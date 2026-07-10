from __future__ import annotations

from dataclasses import dataclass
import re
import zlib


_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"[。！？；!?;]+")


def normalize_for_poison_scan(text: str) -> str:
    """
    Normalize text for poison / junk detection.

    Current policy is conservative: strip whitespace only. We do NOT lowercase or
    otherwise rewrite content so that any downstream hashing/debugging can be
    stable and reversible.
    """

    return _WS_RE.sub("", str(text or ""))


def is_poison_repetitive_text(
    text: str,
    *,
    min_chars: int = 800,
    zlib_level: int = 1,
    compression_ratio_lt: float = 0.15,
    distinct2_lt: float = 0.12,
) -> bool:
    """
    Heuristic poison detector for extreme low-entropy / highly repetitive text.

    This targets samples that can dominate training loss with garbage patterns
    (e.g., `*_*_*` spam, repeated phrases, bracket noise).

    The detector is intentionally conservative:
    - Only scans after whitespace removal
    - Only activates above `min_chars`
    - Requires BOTH unusually good compression AND low local diversity (char bigram distinct-2)
    """

    norm = normalize_for_poison_scan(text)
    if len(norm) < int(min_chars):
        return False

    data = norm.encode("utf-8", errors="ignore")
    if not data:
        return False

    level = int(zlib_level)
    if level < 0 or level > 9:
        raise ValueError("zlib_level must be in [0, 9]")

    compressed = zlib.compress(data, level)
    ratio = float(len(compressed)) / float(len(data))
    if ratio >= float(compression_ratio_lt):
        return False

    if len(norm) < 2:
        return False
    total = int(len(norm) - 1)
    bigrams = {norm[index : index + 2] for index in range(total)}
    distinct2 = float(len(bigrams)) / float(total)
    return distinct2 < float(distinct2_lt)


def has_repeated_sentences(
    text: str, *, min_sentence_chars: int = 12, repeats: int = 3
) -> bool:
    s = str(text or "").replace("\r", "").strip()
    if not s:
        return False
    parts = [part.strip() for part in _SENT_SPLIT_RE.split(s) if part.strip()]
    if not parts:
        return False
    seen: dict[str, int] = {}
    for part in parts:
        if len(part) < int(min_sentence_chars):
            continue
        seen[part] = seen.get(part, 0) + 1
        if seen[part] >= int(repeats):
            return True
    return False


def normalize_text(text: str) -> str:
    s = str(text or "").replace("\u0000", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\ufffd", "")
    return s.strip("\ufeff").strip()


@dataclass(frozen=True)
class CleanResult:
    text: str
    drop_reason: str
