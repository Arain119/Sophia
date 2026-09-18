from __future__ import annotations

import json

import pytest
import torch

from ml.tooling.scripts.build_dpo_pairs import build_pairs
from ml.tooling.scripts.sample_rl_rollouts import select_prompts
from ml.tooling.scripts.train_dpo import _encode
from ml.training.rl.dpo import (
    dpo_accuracy,
    dpo_loss,
    read_preference_pairs,
    sequence_logprob,
    write_preference_pairs,
)
from ml.training.rl.grpo import group_advantages, grpo_loss
from ml.training.rl.rollout import RolloutPrompt, build_model_generator


def test_sequence_logprob_matches_causal_shift_and_masks_prompt() -> None:
    # Token 1 predicts 2, token 2 predicts 3, token 3 predicts 4.  Only the
    # last two positions are assistant tokens in the SFT label convention.
    logits = torch.tensor(
        [[[0.0, 0.0, 4.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 4.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 4.0],
          [4.0, 0.0, 0.0, 0.0, 0.0]]]
    )
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 3, 4]])
    score, count = sequence_logprob(logits, ids, labels)
    expected = torch.log_softmax(logits[0, 1:3], dim=-1)[[0, 1], [3, 4]].mean()
    assert count.tolist() == [2]
    assert torch.allclose(score, expected.unsqueeze(0))


def test_dpo_loss_is_lower_when_policy_prefers_chosen() -> None:
    ref_c = torch.tensor([-2.0, -2.0])
    ref_r = torch.tensor([-2.0, -2.0])
    worse = dpo_loss(torch.tensor([-2.0, -2.0]), torch.tensor([-1.0, -1.0]), ref_c, ref_r)
    better = dpo_loss(torch.tensor([-1.0, -1.0]), torch.tensor([-2.0, -2.0]), ref_c, ref_r)
    assert better < worse
    assert dpo_accuracy(torch.tensor([-1.0, -1.0]), torch.tensor([-2.0, -2.0]), ref_c, ref_r) == 1


def test_build_pairs_selects_margin_and_round_trips(tmp_path) -> None:
    groups = tmp_path / "groups.jsonl"
    groups.write_text(
        json.dumps(
            {
                "prompt_id": "p1",
                "messages": [{"role": "user", "content": "2+2?"}],
                "completions": [
                    {"index": 0, "raw": "4", "score": 1.0},
                    {"index": 1, "raw": "5", "score": 0.0},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "pairs.jsonl"
    report = build_pairs(groups_path=groups, output_path=output)
    assert report["pairs"] == 1
    rows = list(read_preference_pairs(output))
    assert rows[0].chosen == "4"
    assert rows[0].rejected == "5"


def test_preference_writer_rejects_empty(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_preference_pairs(tmp_path / "empty.jsonl", [])


def test_dpo_masks_assistant_history() -> None:
    class Tokenizer:
        eos_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            if "old" in text:
                return [101, 102]
            if "new" in text:
                return [201, 202]
            return [1]

    encoded = _encode(
        Tokenizer(),
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "old"},
            {"role": "user", "content": "follow up"},
        ],
        "new",
        128,
    )
    old_positions = torch.isin(encoded["input_ids"], torch.tensor([101, 102]))
    new_positions = torch.isin(encoded["input_ids"], torch.tensor([201, 202]))
    assert not bool(encoded["labels"][old_positions].ne(-100).any())
    assert bool(encoded["labels"][new_positions].ne(-100).all())


def test_grpo_advantages_require_group_signal() -> None:
    rewards = torch.tensor([[1.0, 0.0, 0.0], [3.0, 1.0, 2.0]])
    advantages = group_advantages(rewards)
    assert torch.allclose(advantages.mean(dim=1), torch.zeros(2), atol=1e-6)
    with pytest.raises(ValueError):
        group_advantages(torch.ones(1, 3))


def test_grpo_loss_is_finite_and_prefers_positive_advantage() -> None:
    old = torch.zeros(1, 2, 3)
    ref = torch.zeros(1, 2, 3)
    mask = torch.ones_like(old)
    advantages = torch.tensor([[1.0, -1.0]])
    neutral = grpo_loss(old, old, ref, advantages, mask)
    improved = grpo_loss(torch.tensor([[[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]]]), old, ref, advantages, mask)
    assert torch.isfinite(neutral)
    assert improved < neutral


def test_rl_prompt_selection_excludes_templates_and_balances_categories() -> None:
    prompts = [
        RolloutPrompt(
            prompt_id="qa_single",
            messages=[{"role": "user", "content": "q"}],
            tags=("chinese_qa", "outside_template_set"),
        ),
        RolloutPrompt(
            prompt_id="qa_multi",
            messages=[
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ],
            tags=("chinese_qa", "outside_template_set"),
        ),
        RolloutPrompt(
            prompt_id="identity",
            messages=[{"role": "user", "content": "who"}],
            tags=("identity", "outside_template_set"),
        ),
        RolloutPrompt(
            prompt_id="memorised",
            messages=[{"role": "user", "content": "2+2"}],
            tags=("math", "in_template_set"),
        ),
    ]
    selected = select_prompts(prompts, max_prompts=3, seed=42)
    assert [prompt.prompt_id for prompt in selected] == [
        "identity",
        "qa_multi",
        "qa_single",
    ]

def test_rollout_generator_removes_only_terminal_eos() -> None:
    class Tokenizer:
        eos_token_id = 0

        def decode(self, tokens, *, skip_special_tokens=False):
            del skip_special_tokens
            return "".join(
                "<eos>" if int(token) in {0, 3} else chr(96 + int(token))
                for token in tokens
            )

    class Model:
        pass

    import ml.runtime.inference.eval.generation_runtime as generation_runtime
    import ml.runtime.inference.eval.runtime as runtime

    original_build = runtime.build_inputs_from_conversation
    original_generate = generation_runtime.generate_with_kv_cache
    try:
        runtime.build_inputs_from_conversation = lambda **_: {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }
        generation_runtime.generate_with_kv_cache = lambda **_: torch.tensor([[1, 2, 1, 3, 0]])
        generator = build_model_generator(
            model=Model(), tokenizer=Tokenizer(), device="cpu", max_new_tokens=4
        )
        rows = generator([{"role": "user", "content": "q"}], 1)
        assert rows == [("a<eos>", 3, True)]
    finally:
        runtime.build_inputs_from_conversation = original_build
        generation_runtime.generate_with_kv_cache = original_generate
