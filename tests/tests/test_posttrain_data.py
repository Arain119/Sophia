from __future__ import annotations

import json

import pytest
import torch

from ml.modeling.text.conversation import encode_conversation
from ml.training.posttrain.data import (
    ChatExample,
    ShuffledBatchIterator,
    SupervisedBatchIterator,
    _normalize_message,
    collate_supervised_examples,
    encode_completion_example,
    encode_supervised_example,
    load_prompt_examples,
    load_supervised_examples,
)


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 3

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [ord(ch) + 4 for ch in str(text)]


def _text_from_ids(ids: torch.Tensor) -> str:
    return "".join(chr(int(token) - 4) for token in ids.tolist())


def test_encode_supervised_example_masks_non_assistant_tokens() -> None:
    tokenizer = _DummyTokenizer()
    example = ChatExample(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ],
        metadata={},
    )

    encoded = encode_supervised_example(
        tokenizer=tokenizer,
        example=example,
        max_seq_len=512,
    )

    assert encoded.input_ids.shape == encoded.labels.shape
    assert int((encoded.labels != -100).sum().item()) > 0
    assert int((encoded.labels == -100).sum().item()) > 0


def test_encode_supervised_example_preserve_final_turn_drops_partial_prior_turns() -> None:
    tokenizer = _DummyTokenizer()
    example = ChatExample(
        messages=[
            {"role": "user", "content": "intro"},
            {"role": "assistant", "content": "P" * 80},
            {"role": "user", "content": "final-question"},
            {"role": "assistant", "content": "final-answer"},
        ],
        metadata={},
    )

    tail_encoded = encode_supervised_example(
        tokenizer=tokenizer,
        example=example,
        max_seq_len=90,
        crop_policy="tail_tokens",
    )
    preserved_encoded = encode_supervised_example(
        tokenizer=tokenizer,
        example=example,
        max_seq_len=90,
        crop_policy="preserve_final_turn",
    )

    assert "P" in _text_from_ids(tail_encoded.input_ids)
    preserved_text = _text_from_ids(preserved_encoded.input_ids)
    assert "P" not in preserved_text
    assert "final-question" in preserved_text
    assert "final-answer" in preserved_text


def test_collate_supervised_examples_pads_labels_with_ignore_index() -> None:
    rows = [
        type(
            "Enc",
            (),
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([-100, 2]),
            },
        )(),
        type(
            "Enc",
            (),
            {
                "input_ids": torch.tensor([3]),
                "attention_mask": torch.tensor([1]),
                "labels": torch.tensor([3]),
            },
        )(),
    ]
    batch = collate_supervised_examples(rows, pad_token_id=0, pad_to_multiple_of=4)

    assert tuple(batch["input_ids"].shape) == (2, 4)
    assert tuple(batch["labels"].shape) == (2, 4)
    assert int(batch["labels"][1, 1].item()) == -100
    assert batch["input_ids"].dtype == torch.int32
    assert batch["labels"].dtype == torch.int32
    assert batch["attention_mask"].dtype == torch.bool


def test_load_supervised_examples_maps_final_field(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "", "final": "A"},
                ]
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    [example] = load_supervised_examples(str(path))
    assert example.messages[1]["content"] == "A"


def test_normalize_message_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="unsupported message role"):
        _normalize_message({"role": "observer", "content": "unused"})


def test_normalize_message_rejects_response_format_on_user() -> None:
    with pytest.raises(ValueError, match="message.response_format is only supported"):
        _normalize_message(
            {
                "role": "user",
                "content": "Q",
                "response_format": {"type": "object", "properties": {}},
            }
        )


def test_load_prompt_examples_trims_trailing_assistant(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ]
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    [example] = load_prompt_examples(str(path))
    assert [m["role"] for m in example.messages] == ["user"]


def test_supervised_batch_iterator_roundtrip_state() -> None:
    tokenizer = _DummyTokenizer()
    dataset = [
        ChatExample(messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], metadata={}),
        ChatExample(messages=[{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}], metadata={}),
        ChatExample(messages=[{"role": "user", "content": "e"}, {"role": "assistant", "content": "f"}], metadata={}),
    ]
    it1 = SupervisedBatchIterator(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=2,
        max_seq_len=128,
        pad_to_multiple_of=4,
        seed=123,
        shuffle=True,
        repeat=True,
        drop_last=True,
    )
    _ = next(it1)
    state = it1.state_dict()

    it2 = SupervisedBatchIterator(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=2,
        max_seq_len=128,
        pad_to_multiple_of=4,
        seed=123,
        shuffle=True,
        repeat=True,
        drop_last=True,
    )
    it2.load_state_dict(state)

    batch1 = next(it1)
    batch2 = next(it2)
    torch.testing.assert_close(batch1["input_ids"], batch2["input_ids"])
    torch.testing.assert_close(batch1["labels"], batch2["labels"])


def test_encode_completion_example_masks_prompt_assistant_history() -> None:
    encoded = encode_completion_example(
        tokenizer=_DummyTokenizer(),
        prompt_messages=[
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
        completion_messages=[{"role": "assistant", "content": "a2"}],
        max_seq_len=1024,
    )

    labeled_positions = (encoded.labels != -100).nonzero(as_tuple=False).squeeze(-1)
    assert int(labeled_positions.numel()) > 0
    assert int(labeled_positions.min().item()) > 0


def test_shuffled_batch_iterator_rejects_impossible_drop_last() -> None:
    with pytest.raises(ValueError, match="dataset_size >= batch_size"):
        ShuffledBatchIterator(
            dataset=[1],
            batch_size=2,
            seed=1,
            shuffle=True,
            repeat=True,
            drop_last=True,
        )


def test_encode_conversation_preserves_visible_assistant_text() -> None:
    encoded = encode_conversation(
        tokenizer=_DummyTokenizer(),
        messages=[
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ],
        max_seq_len=4096,
        add_generation_prompt=False,
    )
    text = _text_from_ids(encoded.input_ids)
    assert "a1" in text
    assert "a2" in text
    assert "<think>" not in text


def test_encode_conversation_ignores_task_metadata_on_consecutive_user_messages() -> None:
    messages_with_task = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second", "task": "ignored"},
    ]
    messages_without_task = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]
    encoded_with_task = encode_conversation(
        tokenizer=_DummyTokenizer(),
        messages=messages_with_task,
        max_seq_len=4096,
        add_generation_prompt=True,
    )
    encoded_without_task = encode_conversation(
        tokenizer=_DummyTokenizer(),
        messages=messages_without_task,
        max_seq_len=4096,
        add_generation_prompt=True,
    )
    text = _text_from_ids(encoded_with_task.input_ids)
    assert encoded_with_task.input_ids.tolist() == encoded_without_task.input_ids.tolist()
    assert "<｜Assistant｜>" in text
