from __future__ import annotations

from ml.data.pretrain_filters import is_poison_repetitive_text


def test_poison_filter_ignores_short_text() -> None:
    assert not is_poison_repetitive_text("hi", min_chars=800)


def test_poison_filter_flags_repetitive_noise() -> None:
    text = ("*_*_*" * 1200) + "\n"
    assert is_poison_repetitive_text(text, min_chars=800)


def test_poison_filter_keeps_high_entropy_text() -> None:
    # High-entropy string: compression ratio should be high, and distinct2 should be near 1.
    # Use a wide range of CJK codepoints to avoid trivial compression.
    chars = [chr(0x4E00 + (i % 2048)) for i in range(3000)]
    text = "".join(chars)
    assert not is_poison_repetitive_text(text, min_chars=800)
