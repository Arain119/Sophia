"""Which prompts are worth spending a rollout on."""

from __future__ import annotations

import json
from pathlib import Path

from ml.training.rl.prompt_pool import (
    IN_TEMPLATE,
    OUTSIDE_TEMPLATE,
    build_from_dataset,
    build_pool,
    derive_expectations,
    memorised_skeletons,
    skeleton,
)


AGE_TEMPLATE = "今年爸爸的年龄是儿子的 {k} 倍，儿子今年 {n} 岁。11 年后两人年龄相差多少岁？"


def _math_row(k: int, n: int) -> dict:
    return {
        "category": "math",
        "messages": [
            {"role": "user", "content": AGE_TEMPLATE.format(k=k, n=n)},
            {"role": "assistant", "content": "<think>差值恒定。</think>相差二十岁。"},
        ],
    }


def _chat_row(text: str) -> dict:
    return {
        "category": "companion_chat",
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": "<think>接住。</think>我在呢。"},
        ],
    }


def test_skeleton_collapses_a_template_that_only_varies_by_number() -> None:
    a = skeleton(AGE_TEMPLATE.format(k=3, n=10))
    b = skeleton(AGE_TEMPLATE.format(k=6, n=17))
    assert a == b


def test_skeleton_keeps_two_different_questions_apart() -> None:
    assert skeleton("水的沸点是多少摄氏度？") != skeleton("《红楼梦》的作者是谁？")


def test_memorised_skeletons_finds_the_repeated_shape() -> None:
    rows = [_math_row(k % 7 + 2, k % 19 + 3) for k in range(60)]
    # Distinct wording, not a numbered variant: a chat prompt that differed
    # only by a digit would collapse to one skeleton too, which is the point.
    rows += [_chat_row(f"今天心情有点{word}") for word in "低沉阴郁烦躁疲惫涣散慌乱" * 10]
    memorised = memorised_skeletons(rows, threshold=50)
    assert len(memorised) == 1
    assert skeleton(AGE_TEMPLATE.format(k=3, n=10)) in memorised


def test_a_rare_shape_is_not_memorised() -> None:
    rows = [_math_row(3, 10) for _ in range(10)]
    assert memorised_skeletons(rows, threshold=50) == set()


def test_pool_tags_each_prompt_by_template_status() -> None:
    memorised = {skeleton(AGE_TEMPLATE.format(k=3, n=10))}
    rows = [_math_row(4, 12), _chat_row("下雨了")]
    pool, stats = build_pool(candidate_rows=rows, memorised=memorised)
    assert stats.total == 2
    assert stats.inside == 1
    assert stats.outside == 1
    assert IN_TEMPLATE in pool[0]["tags"]
    assert OUTSIDE_TEMPLATE in pool[1]["tags"]


def test_pool_stops_at_the_last_user_turn() -> None:
    row = {
        "category": "chinese_qa",
        "messages": [
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": "第二问"},
            {"role": "assistant", "content": "这一条是策略要生成的，不能出现在提示里"},
        ],
    }
    pool, _stats = build_pool(candidate_rows=[row], memorised=set())
    messages = pool[0]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[-1]["content"] == "第二问"


def test_pool_skips_a_conversation_with_no_user_turn() -> None:
    row = {"category": "x", "messages": [{"role": "assistant", "content": "独白"}]}
    pool, stats = build_pool(candidate_rows=[row], memorised=set())
    assert pool == []
    assert stats.total == 0


def test_expectations_are_only_the_ones_that_can_be_read_off_the_row() -> None:
    assert derive_expectations({"category": "code"}) == {"code_block": True}
    assert derive_expectations({"category": "math"}) == {"numeral": True}
    # Nothing is asserted about a chat turn: a wrong expectation would teach
    # the policy to satisfy something the prompt never asked for.
    assert derive_expectations({"category": "companion_chat"}) == {}


def test_build_from_dataset_writes_a_loadable_pool(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text(
        "".join(
            json.dumps(_math_row(k % 7 + 2, k % 19 + 3), ensure_ascii=False) + "\n"
            for k in range(60)
        ),
        encoding="utf-8",
    )
    candidates = tmp_path / "validation.jsonl"
    candidates.write_text(
        json.dumps(_math_row(5, 9), ensure_ascii=False) + "\n"
        + json.dumps(_chat_row("在吗"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "pool.jsonl"

    stats = build_from_dataset(
        train_path=train, candidate_path=candidates, output_path=out, threshold=50
    )

    assert stats.total == 2
    assert stats.inside == 1
    from ml.training.rl.rollout import load_prompts

    prompts = load_prompts(out)
    assert len(prompts) == 2
    assert prompts[0].expectations == {"numeral": True}


def test_the_default_threshold_is_the_measured_one() -> None:
    """Recorded from the released corpus so the constant is not quietly re-tuned.

    At fifty rows per skeleton the marked set is sixty-four shapes covering
    24.1% of the training corpus, all of them maths or code. At twenty it
    starts marking structured_output, which is not degenerate.
    """
    from ml.training.rl.prompt_pool import DEFAULT_TEMPLATE_THRESHOLD

    assert DEFAULT_TEMPLATE_THRESHOLD == 50
