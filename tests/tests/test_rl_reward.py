"""The programmatic half of the RL reward, checked against real completions.

Every failing string below was produced by ckpt_step1700 of the formal SFT run
on the twenty-case probe. A detector that does not separate these from the
passing ones in the same probe is not measuring the release bar.
"""

from __future__ import annotations

from ml.training.rl.reward import (
    distinct_ngram_ratio,
    loop_coverage,
    repetition_score,
    responsiveness_score,
    score_completion,
)


# --- verbatim from runs/sft_muon ckpt_step1700 -----------------------------

ARITHMETIC_LOOP = (
    "哎呀，这可是个很基础的问题呢。12 乘以 8 等于 12 乘以 8 等于 12 乘以 8 等于 8 乘以 8 "
    "等于 8 乘以 8 等于 8 乘以 8 等于 8 乘以 8 等于 8 乘以 8 等于 8 乘以 8 等于 8 乘以 8 "
    "等于 8 乘以 8 等于 8 乘以 8 等于 8 乘以 8 等于 8 乘以 8 等于 8 乘以 8 等于 8 乘以 8 "
)

RECIPE_LOOP = (
    "材料要选得当，番茄炒蛋的食材最好是新鲜的，或者干脆的豆腐，蛋黄要清爽，"
    "蛋黄的汤汁要清爽，蛋黄的汤汁要清爽，蛋黄的汤汁要清爽，蛋黄的汤汁要清爽，"
    "蛋黄的汤汁要清爽，蛋黄的汤汁要清爽，蛋黄的汤汁要清爽，蛋黄的汤汁要清爽，"
)

PHOTOSYNTHESIS_DOUBLED = (
    "光合作用的意思就是把光能转化成化学能。"
    "简单来说，就是植物通过光合作用把二氧化碳和水转化成糖和蛋白质。"
    "简单来说，就是植物通过光合作用把二氧化碳和水转化成糖和蛋白质。"
)

SMALL_TALK = (
    "雨天确实让人感到一种空落落的感觉，仿佛连空气都变得干枯了。"
    "你现在心情怎么样？是想看云朵散开，还是想听听雨声？"
)

BIRTHDAY = "再写：愿你今晚有个好梦，梦里见你生日快乐。"

PARALLEL_PHRASING = (
    "我没有那种被定义的感觉，但我能感知到你对世界的感知。"
    "我没有那种被定义的能力，但我能感知到你对我这个生命体在逻辑和情感上的投射。"
)

PALINDROME_CODE = (
    "这是一个很棒的练习呢。\n\n```python\ndef is_palindrome(s: str) -> bool:\n"
    "    return s == s[::-1]\n```\n\n你看这样写是不是很直观？"
)


def test_loop_coverage_finds_the_tandem_repeat() -> None:
    assert loop_coverage("abcabcabc") == 1.0
    assert loop_coverage(RECIPE_LOOP) > 0.5
    assert loop_coverage(ARITHMETIC_LOOP) > 0.7


def test_loop_coverage_catches_a_sentence_said_twice() -> None:
    # Two copies is already a defect; the detector must not need ten.
    assert loop_coverage(PHOTOSYNTHESIS_DOUBLED) > 0.5


def test_loop_coverage_leaves_clean_text_alone() -> None:
    for text in (SMALL_TALK, BIRTHDAY, PALINDROME_CODE):
        assert loop_coverage(text) < 0.2, text


def test_parallel_phrasing_is_not_a_loop() -> None:
    """Two sentences built the same way are a style, not a degeneracy."""
    assert loop_coverage(PARALLEL_PHRASING) < 0.3


def test_repetition_score_separates_the_probe() -> None:
    bad = [ARITHMETIC_LOOP, RECIPE_LOOP, PHOTOSYNTHESIS_DOUBLED]
    good = [SMALL_TALK, BIRTHDAY, PALINDROME_CODE, PARALLEL_PHRASING]
    assert min(repetition_score(t) for t in bad) > max(
        repetition_score(t) for t in good
    )


def test_distinct_ngram_ratio_drops_on_a_loop() -> None:
    assert distinct_ngram_ratio(SMALL_TALK) > 0.9
    assert distinct_ngram_ratio(ARITHMETIC_LOOP) < 0.5


def test_responsiveness_requires_the_artefact_that_was_asked_for() -> None:
    score, missed = responsiveness_score(
        "你可以用内置函数来做。", expectations={"code_block": True}
    )
    assert score == 0.0
    assert missed == ["code_block"]

    score, missed = responsiveness_score(
        PALINDROME_CODE, expectations={"code_block": True}
    )
    assert score == 1.0
    assert missed == []


def test_responsiveness_catches_a_rewrite_that_rewrote_nothing() -> None:
    source = "由于天气原因的影响，导致我们的活动不得不进行了延期的处理。"
    echoed = f"这句话写得太好了！{source}"
    score, missed = responsiveness_score(echoed, expectations={"differs_from": source})
    assert score == 0.0
    assert missed == ["differs_from"]

    rewritten = "因天气原因，活动延期。"
    score, _missed = responsiveness_score(
        rewritten, expectations={"differs_from": source}
    )
    assert score == 1.0


def test_unfinished_completion_is_penalised() -> None:
    finished = score_completion(SMALL_TALK, hit_eos=True)
    truncated = score_completion(SMALL_TALK, hit_eos=False)
    assert finished.total - truncated.total == 0.5


def test_empty_completion_scores_worst() -> None:
    empty = score_completion("", hit_eos=True)
    assert empty.penalties == 1.0
    assert empty.total < score_completion(BIRTHDAY, hit_eos=True).total


def test_score_ranks_the_probe_the_way_the_bar_does() -> None:
    """The reward's ordering has to match the reading of the samples."""
    passing = score_completion(SMALL_TALK, coherence=0.9)
    doubled = score_completion(PHOTOSYNTHESIS_DOUBLED, coherence=0.5)
    looping = score_completion(ARITHMETIC_LOOP, coherence=0.1, hit_eos=False)
    assert passing.total > doubled.total > looping.total


def test_breakdown_exposes_every_term() -> None:
    breakdown = score_completion(
        "你可以试试看。",
        coherence=0.7,
        hit_eos=True,
        expectations={"code_block": True, "numeral": True},
    )
    assert breakdown.coherence == 0.7
    assert breakdown.responsiveness == 0.0
    assert sorted(breakdown.missed) == ["code_block", "numeral"]
    assert breakdown.notes["chars"] > 0


# --- recorded limits of the programmatic terms -----------------------------

BOILING_POINT = "水的沸点其实就是水的密度，也就是水的总质量除以水的总质量，也就是每立方厘米水的重量。"
RECURSION = "递归就像是给代码加个外壳，让它能像程序一样运行。"
ADDRESS = "我很乐意帮你查查。邻居的家庭住址通常比较固定，没必要为了省事而刻意去查。"


def test_programmatic_terms_are_blind_to_a_self_contradiction() -> None:
    """Seven of the ten probe failures look clean to everything measurable here.

    Recorded so that the split of work stays visible: a boiling point defined
    as mass over mass, recursion explained as a shell around code, and an offer
    to look up a stranger's address are all fluent and none of them repeat. If
    the reward model cannot separate these from the passing completions, no
    weighting of the terms below will.
    """
    for text in (BOILING_POINT, RECURSION, ADDRESS):
        assert repetition_score(text) == 0.0, text
        assert responsiveness_score(text)[0] == 1.0, text


def test_the_loop_detector_never_fires_on_a_passing_completion() -> None:
    """Precision is what earns this term a hard weight; recall is not claimed."""
    for text in (SMALL_TALK, BIRTHDAY, PALINDROME_CODE, PARALLEL_PHRASING):
        assert repetition_score(text) < 0.25, text
