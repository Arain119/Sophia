from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

import ahocorasick
import xxhash

from ml.data.pretrain_filters import (
    has_repeated_sentences,
    is_poison_repetitive_text,
    normalize_text,
)


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_TAG_RE = re.compile(r"</?[A-Za-z][^>]{0,200}>")
_OCR_LINE_RE = re.compile(r"^\s*\d{1,4}\s*[、.)．]\s*")
_TRAILING_BOILERPLATE_RE = re.compile(
    r"(?:订阅|手机报|免责声明|版权所有|备案号|ICP备|联系我们|扫码关注|"
    r"all rights reserved|privacy policy|cookie policy)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CleanDecision:
    text: str
    reason: str
    cjk_ratio: float
    latin_ratio: float


def _language_ratios(text: str) -> tuple[float, float]:
    cjk = len(_CJK_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    denominator = max(cjk + latin, 1)
    return float(cjk) / denominator, float(latin) / denominator


def _strip_trailing_boilerplate(text: str) -> str:
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    removed = 0
    while lines and removed < 8 and _TRAILING_BOILERPLATE_RE.search(lines[-1]):
        lines.pop()
        removed += 1
    return "\n".join(lines).strip()


def clean_document(
    text: str,
    *,
    language: str,
    min_chars: int = 200,
    reject_ocr_enumeration: bool = True,
    minimum_language_ratio: float | None = None,
) -> CleanDecision:
    cleaned = _strip_trailing_boilerplate(normalize_text(text))
    cjk_ratio, latin_ratio = _language_ratios(cleaned)
    if len(cleaned) < int(min_chars):
        return CleanDecision("", "too_short", cjk_ratio, latin_ratio)
    if is_poison_repetitive_text(cleaned):
        return CleanDecision("", "poison_repetition", cjk_ratio, latin_ratio)
    if has_repeated_sentences(cleaned, min_sentence_chars=16, repeats=3):
        return CleanDecision("", "repeated_sentences", cjk_ratio, latin_ratio)
    lines = [line for line in cleaned.splitlines() if line.strip()]
    if lines:
        ocr_fraction = sum(bool(_OCR_LINE_RE.match(line)) for line in lines) / len(lines)
        if reject_ocr_enumeration and len(lines) >= 6 and ocr_fraction >= 0.45:
            return CleanDecision("", "ocr_enumeration", cjk_ratio, latin_ratio)
        substantial = [dedup_normalize(line) for line in lines if len(line.strip()) >= 32]
        if len(substantial) >= 12:
            prefixes = Counter(line[:24] for line in substantial)
            if max(prefixes.values(), default=0) / len(substantial) >= 0.5:
                return CleanDecision("", "repeated_line_prefix", cjk_ratio, latin_ratio)
    if len(_URL_RE.findall(cleaned)) >= max(4, len(cleaned) // 1000):
        return CleanDecision("", "url_density", cjk_ratio, latin_ratio)
    if len(_TAG_RE.findall(cleaned)) >= max(6, len(cleaned) // 500):
        return CleanDecision("", "markup_density", cjk_ratio, latin_ratio)
    requested_language = str(language).lower()
    language_ratio = (
        float(minimum_language_ratio)
        if minimum_language_ratio is not None
        else (0.20 if requested_language == "zh" else 0.55)
    )
    letters = sum(unicodedata.category(character).startswith("L") for character in cleaned)
    if requested_language in {"zh", "en"} and letters < 32:
        return CleanDecision("", "language_mismatch", cjk_ratio, latin_ratio)
    if requested_language == "zh" and cjk_ratio < language_ratio:
        return CleanDecision("", "language_mismatch", cjk_ratio, latin_ratio)
    if requested_language == "en" and latin_ratio < language_ratio:
        return CleanDecision("", "language_mismatch", cjk_ratio, latin_ratio)
    return CleanDecision(cleaned, "", cjk_ratio, latin_ratio)


def dedup_normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(normalized.split())


_UINT64_MASK = (1 << 64) - 1
_NEAR_CHUNK_MIN = 32
_NEAR_CHUNK_MAX = 128
_NEAR_CHUNK_BOUNDARY_MASK = 63
_NEAR_MINHASH_SEEDS = tuple(
    xxhash.xxh64_intdigest(f"sophia-near-minhash-v3:{index}")
    for index in range(64)
)


def _content_defined_chunks(text: str) -> list[str]:
    """Split text at content-derived boundaries that survive insertions."""
    chunks: list[str] = []
    start = 0
    fingerprint = 0
    for index, character in enumerate(text):
        value = ord(character)
        gear = ((value + 1) * 0x9E3779B185EBCA87) & _UINT64_MASK
        fingerprint = ((fingerprint << 1) + gear) & _UINT64_MASK
        length = index + 1 - start
        if length < _NEAR_CHUNK_MIN:
            continue
        if (
            fingerprint & _NEAR_CHUNK_BOUNDARY_MASK == 0
            or length >= _NEAR_CHUNK_MAX
        ):
            chunks.append(text[start : index + 1])
            start = index + 1
    if start < len(text):
        chunks.append(text[start:])
    return chunks


def near_duplicate_signature(text: str, *, size: int = 16) -> tuple[int, ...]:
    """Return coordinate-aligned MinHash values over stable content chunks."""
    normalized = dedup_normalize(text)
    if len(normalized) < 5:
        return ()
    signature_size = int(size)
    if not 0 < signature_size <= len(_NEAR_MINHASH_SEEDS):
        raise ValueError("near-duplicate signature size must be in [1, 64]")
    chunks = _content_defined_chunks(normalized)
    return tuple(
        min(xxhash.xxh64_intdigest(chunk, seed=seed) for chunk in chunks)
        for seed in _NEAR_MINHASH_SEEDS[:signature_size]
    )


def near_signature_similarity(
    left: tuple[int, ...], right: tuple[int, ...]
) -> float:
    if not left or len(left) != len(right):
        return 0.0
    matches = sum(
        left_value == right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    return float(matches) / float(len(left))


class NearDeduper:
    def __init__(self, *, threshold: float = 0.75) -> None:
        self.threshold = float(threshold)
        self.signatures: list[tuple[int, ...]] = []
        self.index: dict[tuple[int, int], list[int]] = defaultdict(list)

    def add_if_unique(self, signature: tuple[int, ...]) -> bool:
        if len(signature) < 8:
            self.signatures.append(signature)
            return True
        coordinate_values = tuple(enumerate(signature))
        posting_lists = [
            self.index[(coordinate, int(value))]
            for coordinate, value in coordinate_values
            if (coordinate, int(value)) in self.index
        ]
        candidate_hits: Counter[int] = Counter()
        for posting_list in posting_lists:
            candidate_hits.update(posting_list)
        min_shared = max(int(len(signature) * self.threshold), 1)
        for candidate_id, shared in candidate_hits.items():
            if shared < min_shared:
                continue
            similarity = near_signature_similarity(
                signature, self.signatures[candidate_id]
            )
            if similarity >= self.threshold:
                return False
        signature_id = len(self.signatures)
        self.signatures.append(signature)
        for coordinate, value in coordinate_values:
            self.index[(coordinate, int(value))].append(signature_id)
        return True


def document_split(document_sha256: str) -> str:
    bucket = int(str(document_sha256)[:8], 16) % 10_000
    if bucket < 9600:
        return "train"
    if bucket < 9800:
        return "val"
    return "test"


class BenchmarkMatcher:
    def __init__(self, signatures: tuple[tuple[str, str], ...]) -> None:
        self.signatures = signatures
        self.anchor_size = 24
        self.by_anchor: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for case_id, signature in signatures:
            offset = max((len(signature) - self.anchor_size) // 2, 0)
            anchor = signature[offset : offset + self.anchor_size]
            self.by_anchor[anchor].append((case_id, signature))
        self.automaton = ahocorasick.Automaton()
        for anchor in self.by_anchor:
            self.automaton.add_word(anchor, anchor)
        if self.by_anchor:
            self.automaton.make_automaton()

    def match(self, normalized_text: str) -> str:
        if not self.by_anchor or len(normalized_text) < self.anchor_size:
            return ""
        seen: set[str] = set()
        for _end_offset, anchor in self.automaton.iter(normalized_text):
            for case_id, signature in self.by_anchor.get(anchor, ()):
                if case_id not in seen and signature in normalized_text:
                    return case_id
                seen.add(case_id)
        return ""


__all__ = [
    "BenchmarkMatcher",
    "CleanDecision",
    "NearDeduper",
    "clean_document",
    "dedup_normalize",
    "document_split",
    "near_duplicate_signature",
    "near_signature_similarity",
]
