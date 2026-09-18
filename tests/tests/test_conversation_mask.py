from __future__ import annotations

import pytest

from ml.modeling.text.conversation import (
    assistant_is_masked,
    encode_conversation,
    render_conversation_segments,
)


class _CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) % 251 for ch in str(text)]


def _supervised(segments) -> str:
    return "".join(segment.text for segment in segments if segment.supervise)


def test_assistant_without_mask_is_supervised() -> None:
    segments = render_conversation_segments(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        add_generation_prompt=False,
    )
    assert "yo" in _supervised(segments)


def test_assistant_mask_removes_supervision() -> None:
    segments = render_conversation_segments(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo", "mask": True},
        ],
        add_generation_prompt=False,
    )
    assert _supervised(segments) == ""


def test_assistant_mask_rejects_non_boolean() -> None:
    with pytest.raises(TypeError, match="assistant.mask must be a boolean"):
        render_conversation_segments(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo", "mask": 1},
            ],
            add_generation_prompt=False,
        )


def test_masked_prior_turns_leave_only_final_turn_supervised() -> None:
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "mask": True},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "<think>t</think>a2"},
    ]
    segments = render_conversation_segments(messages, add_generation_prompt=False)
    supervised = _supervised(segments)
    assert "a1" not in supervised
    assert "<think>t</think>a2" in supervised


def test_masked_turn_labels_are_ignore_index() -> None:
    tokenizer = _CharTokenizer()
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "mask": True},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    encoding = encode_conversation(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=4096,
        add_generation_prompt=False,
    )
    labels = encoding.labels.tolist()
    supervised_ids = [value for value in labels if value != -100]
    assert supervised_ids
    masked_only = encode_conversation(
        tokenizer=tokenizer,
        messages=messages[:2],
        max_seq_len=4096,
        add_generation_prompt=False,
    )
    assert set(masked_only.labels.tolist()) == {-100}


def test_assistant_is_masked_helper() -> None:
    assert assistant_is_masked({"role": "assistant", "mask": True}) is True
    assert assistant_is_masked({"role": "assistant"}) is False
    with pytest.raises(TypeError):
        assistant_is_masked({"role": "assistant", "mask": "yes"})
