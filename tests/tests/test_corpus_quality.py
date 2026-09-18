from __future__ import annotations

from ml.data.pretrain_filters import has_repeated_sentences, normalize_text
from ml.tooling.scripts.data.corpus_quality import (
    BenchmarkMatcher,
    NearDeduper,
    clean_document,
    dedup_normalize,
    document_split,
    near_duplicate_signature,
    near_signature_similarity,
)


def test_clean_document_filters_language_and_trailing_boilerplate() -> None:
    text = "".join(
        f"这是第{index}个用于测试清洗逻辑的中文段落，内容各不相同。"
        for index in range(20)
    ) + "\n订阅手机报，发送 ABC 到 10658000"
    decision = clean_document(text, language="zh")

    assert decision.reason == ""
    assert "订阅手机报" not in decision.text
    assert decision.cjk_ratio > 0.9
    assert clean_document("English only " * 30, language="zh").reason == "language_mismatch"


def test_clean_document_filters_ocr_enumeration() -> None:
    text = "\n".join(f"{index}、这是编号残片内容并且长度足够" for index in range(20))
    assert clean_document(text, language="zh").reason == "ocr_enumeration"
    assert (
        clean_document(text, language="zh", reject_ocr_enumeration=False).reason
        == ""
    )


def test_english_repetition_and_unicode_controls_are_normalized() -> None:
    repeated = ("This duplicated answer sentence contains meaningful text.\n" * 8) + (
        "A distinct explanatory paragraph with enough content.\n" * 2
    )
    assert has_repeated_sentences(repeated, min_sentence_chars=16, repeats=3)
    assert normalize_text("left\u200bright\x07\nnext") == "leftright\nnext"


def test_language_ratio_cannot_be_satisfied_by_one_letter_and_digits() -> None:
    assert clean_document("中" + "1234567890" * 40, language="zh").reason == (
        "language_mismatch"
    )


def test_clean_document_filters_repeated_line_prefixes() -> None:
    text = "\n".join(
        f"Dicuspiditermes hutsoni repeated taxonomy field number {index} with value"
        for index in range(20)
    )

    assert clean_document(text, language="en").reason in {
        "poison_repetition",
        "repeated_line_prefix",
    }


def test_near_deduper_rejects_small_document_edit() -> None:
    base = "机器学习需要高质量数据和可靠评测。" * 100
    edited = base + "补充一句说明。"
    deduper = NearDeduper()

    assert deduper.add_if_unique(near_duplicate_signature(base)) is True
    assert deduper.add_if_unique(near_duplicate_signature(edited)) is False


def test_near_signature_survives_prefix_insertion_in_long_realistic_text() -> None:
    base = "\n".join(
        f"第{index}节包含不同的事实、数字{index * 17}和用于训练的上下文说明。"
        for index in range(400)
    )
    original = near_duplicate_signature(base, size=8)
    prefixed = near_duplicate_signature("X" + base, size=8)

    assert near_signature_similarity(original, prefixed) >= 0.75


def test_near_deduper_compares_fixed_coordinates_not_sorted_set_overlap() -> None:
    deduper = NearDeduper()
    assert deduper.add_if_unique((1, 2, 3, 4, 5, 6, 100, 101))
    assert deduper.add_if_unique((0, 1, 2, 3, 4, 5, 6, 200))


def test_document_split_is_deterministic() -> None:
    value = "01234567" + "0" * 56
    assert document_split(value) == document_split(value)


def test_benchmark_matcher_uses_indexed_anchor() -> None:
    signature = "这是一个足够长的基准测试问题用于验证训练语料污染过滤能够正确命中"
    matcher = BenchmarkMatcher((("case", dedup_normalize(signature)),))

    assert matcher.match(dedup_normalize(f"前文。{signature}。后文")) == "case"
    assert matcher.match(dedup_normalize("完全无关的训练文档内容")) == ""
