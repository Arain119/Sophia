"""Grouping, parsing and round-tripping of rollouts, without a model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.training.rl.rollout import (
    Completion,
    RolloutPrompt,
    group_has_signal,
    load_prompts,
    read_groups,
    rollout_group,
    split_think,
    write_groups,
)


PROMPT = RolloutPrompt(
    prompt_id="p1",
    messages=[{"role": "user", "content": "用一句话解释递归。"}],
    expectations={"min_chars": 8},
)


def _fake_generator(rows):
    def generate_fn(messages, group_size):
        assert isinstance(messages, list)
        return rows[:group_size]

    return generate_fn


def test_split_think_separates_the_block_from_the_answer() -> None:
    raw = "<think>对方在问概念，说清楚就好。</think>递归就是函数调用自己。"
    think, answer = split_think(raw)
    assert think == "对方在问概念，说清楚就好。"
    assert answer == "递归就是函数调用自己。"


def test_split_think_leaves_an_unthought_completion_whole() -> None:
    think, answer = split_think("直接回答。")
    assert think == ""
    assert answer == "直接回答。"


def test_split_think_drops_the_end_of_sentence_marker() -> None:
    think, answer = split_think("<think>短。</think>好的。<eos>", eos="<eos>")
    assert answer == "好的。"


def test_rollout_group_parses_every_row() -> None:
    rows = [
        ("<think>a</think>答案一", 12, True),
        ("<think>b</think>答案二", 9, False),
    ]
    completions = rollout_group(
        PROMPT, generate_fn=_fake_generator(rows), group_size=2
    )
    assert [c.answer for c in completions] == ["答案一", "答案二"]
    assert [c.think for c in completions] == ["a", "b"]
    assert [c.hit_eos for c in completions] == [True, False]
    assert [c.index for c in completions] == [0, 1]


def test_a_group_of_one_cannot_be_compared() -> None:
    with pytest.raises(ValueError):
        rollout_group(PROMPT, generate_fn=_fake_generator([("x", 1, True)]), group_size=1)


def test_a_short_generator_result_is_an_error_not_a_silent_group() -> None:
    def generate_fn(messages, group_size):
        return [("only one", 3, True)]

    with pytest.raises(RuntimeError):
        rollout_group(PROMPT, generate_fn=generate_fn, group_size=4)


def test_group_has_signal_rejects_a_group_that_agrees() -> None:
    # Every completion earning the same reward means a zero advantage for the
    # whole group; sampling more of these spends the step on nothing.
    assert not group_has_signal([1.0, 1.0, 1.0, 1.0])
    assert not group_has_signal([0.80, 0.81, 0.80, 0.79])
    assert group_has_signal([1.4, 0.2, 1.1, 0.3])


def test_group_has_signal_needs_two_completions() -> None:
    assert not group_has_signal([0.5])
    assert not group_has_signal([])


def test_groups_round_trip_through_the_file(tmp_path: Path) -> None:
    completions = rollout_group(
        PROMPT,
        generate_fn=_fake_generator(
            [("<think>a</think>一", 5, True), ("<think>b</think>二", 6, True)]
        ),
        group_size=2,
    )
    path = tmp_path / "groups.jsonl"
    assert write_groups(path, [(PROMPT, completions)]) == 1

    restored = list(read_groups(path))
    assert len(restored) == 1
    prompt, rows = restored[0]
    assert prompt.prompt_id == "p1"
    assert prompt.expectations == {"min_chars": 8}
    assert [row.answer for row in rows] == ["一", "二"]
    assert [row.hit_eos for row in rows] == [True, True]


def test_load_prompts_reads_expectations(tmp_path: Path) -> None:
    path = tmp_path / "pool.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt_id": "code_1",
                "messages": [{"role": "user", "content": "写个函数"}],
                "expectations": {"code_block": True},
                "tags": ["code", "outside_template"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    prompts = load_prompts(path)
    assert len(prompts) == 1
    assert prompts[0].expectations == {"code_block": True}
    assert prompts[0].tags == ("code", "outside_template")


def test_load_prompts_rejects_an_empty_pool(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompts(path)


def test_load_prompts_rejects_a_row_without_messages(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"prompt_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompts(path)


def test_completion_payload_keeps_the_raw_text() -> None:
    """The raw decode is what lets a later pass re-split or re-score."""
    completion = Completion(
        prompt_id="p", index=0, raw="<think>t</think>a", think="t", answer="a",
        new_tokens=4, hit_eos=True,
    )
    assert completion.to_payload()["raw"] == "<think>t</think>a"
