from __future__ import annotations

from collections.abc import Callable

import torch

from ml.modeling.text.conversation import (
    ConversationMessage,
    encode_conversation,
    encode_rendered_segments,
    normalize_conversation_messages,
    render_conversation_segments,
)
from ml.data.jsonl_stream import iter_jsonl_objects
from ml.training.posttrain.contracts import SupportsPosttrainTokenizer
from ml.training.posttrain.data_iterators import (
    ShuffledBatchIterator,
    SupervisedBatchIteratorCore,
)
from ml.training.posttrain.data_types import ChatExample, EncodedChatExample


def _stringify_content(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    raise TypeError(f"message content must be a string or null, got {type(value)!r}")


def _normalize_user_content_blocks(raw: object) -> list[ConversationMessage]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("user.content_blocks must be a list when provided")
    out: list[ConversationMessage] = []
    for block in raw:
        if not isinstance(block, dict):
            raise TypeError("user.content_blocks entries must be objects")
        block_type = str(block.get("type", "")).strip().lower()
        if block_type == "text":
            out.append({"type": "text", "text": _stringify_content(block.get("text"))})
            continue
        raise ValueError(f"unsupported content block type: {block_type!r}")
    return out


def _copy_message_controls(
    raw: dict[str, object],
    out: ConversationMessage,
) -> ConversationMessage:
    role = str(out.get("role", "")).strip()
    task = raw.get("task")
    if task is not None:
        if not isinstance(task, str) or not task.strip():
            raise TypeError("message.task must be a non-empty string when provided")
        out["task"] = task.strip()
    if "wo_eos" in raw:
        out["wo_eos"] = bool(raw.get("wo_eos"))
    if "mask" in raw:
        out["mask"] = raw.get("mask")
    response_format = raw.get("response_format")
    if response_format is not None:
        if role not in {"system", "developer"}:
            raise ValueError(
                "message.response_format is only supported on system/developer messages, "
                f"got {role!r}"
            )
        if not isinstance(response_format, dict):
            raise TypeError("message.response_format must be an object when provided")
        out["response_format"] = dict(response_format)
    return out


def _normalize_message(raw: object) -> ConversationMessage:
    if not isinstance(raw, dict):
        raise TypeError(f"message must be an object, got {type(raw)!r}")
    role = raw.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("message role must be a non-empty string")
    normalized_role = str(role).strip()
    if normalized_role not in {
        "system",
        "user",
        "assistant",
        "developer",
        "latest_reminder",
    }:
        raise ValueError(f"unsupported message role: {normalized_role!r}")

    if normalized_role == "assistant":
        content = _stringify_content(raw.get("content"))
        final = raw.get("final")
        out: ConversationMessage = {
            "role": "assistant",
            "content": _stringify_content(final) if (final is not None and not content) else content,
        }
        return _copy_message_controls(raw, out)

    if normalized_role in {"user", "developer"}:
        out = {"role": "user"}
        if normalized_role == "developer":
            out["role"] = "developer"
        blocks = _normalize_user_content_blocks(raw.get("content_blocks"))
        if blocks:
            out["content_blocks"] = blocks
        content = _stringify_content(raw.get("content"))
        if content:
            out["content"] = content
        return _copy_message_controls(raw, out)

    return _copy_message_controls(
        raw,
        {
            "role": normalized_role,
            "content": _stringify_content(raw.get("content")),
        },
    )


def _load_examples(
    path: str,
    *,
    message_keys: tuple[str, ...],
    normalize_message_fn: Callable[[object], ConversationMessage],
) -> list[ChatExample]:
    out: list[ChatExample] = []
    for obj in iter_jsonl_objects(path):
        if not isinstance(obj, dict):
            continue
        raw_messages = None
        for key in message_keys:
            candidate = obj.get(key)
            if candidate is not None:
                raw_messages = candidate
                break
        if not isinstance(raw_messages, list):
            raise ValueError(
                f"dataset object is missing message list keys: {message_keys!r}"
            )
        messages = [normalize_message_fn(message) for message in raw_messages]
        messages = normalize_conversation_messages(messages)
        if not messages:
            raise ValueError("conversation must contain at least one message")
        metadata = dict(obj.get("metadata") or {})
        for passthrough_key in (
            "reference_answer",
            "answers",
            "required_substrings",
            "forbidden_substrings",
        ):
            if passthrough_key in obj and passthrough_key not in metadata:
                metadata[passthrough_key] = obj[passthrough_key]
        out.append(ChatExample(messages=messages, metadata=metadata))
    if not out:
        raise ValueError(f"no chat examples loaded from: {path}")
    return out


def load_supervised_examples(path: str) -> list[ChatExample]:
    return _load_examples(
        path,
        message_keys=("conversations", "messages"),
        normalize_message_fn=_normalize_message,
    )


def load_prompt_examples(path: str) -> list[ChatExample]:
    examples = _load_examples(
        path,
        message_keys=("prompt", "prompt_messages", "conversations", "messages"),
        normalize_message_fn=_normalize_message,
    )
    out: list[ChatExample] = []
    for example in examples:
        messages = list(example.messages)
        if messages and str(messages[-1].get("role")) == "assistant":
            messages = messages[:-1]
        if not messages:
            raise ValueError(
                "prompt example is empty after trimming trailing assistant message"
            )
        out.append(
            ChatExample(
                messages=messages,
                metadata=dict(example.metadata),
            )
        )
    return out

def _message_token_length(
    *,
    tokenizer: SupportsPosttrainTokenizer,
    messages: list[ConversationMessage],
) -> int:
    encoded = encode_conversation(
        tokenizer=tokenizer,
        messages=list(messages),
        max_seq_len=0,
        add_generation_prompt=False,
    )
    return int(encoded.input_ids.numel())


def _final_turn_start_index(messages: list[ConversationMessage]) -> int:
    total = len(messages)
    if total <= 1:
        return 0
    if str(messages[-1].get("role", "")).strip() != "assistant":
        return max(total - 1, 0)
    for idx in range(total - 2, -1, -1):
        message = messages[idx]
        if str(message.get("role", "")).strip() != "user":
            continue
        return int(idx)
    return 0


def _select_message_suffix_preserving_final_turn(
    *,
    tokenizer: SupportsPosttrainTokenizer,
    example: ChatExample,
    max_seq_len: int,
) -> list[ConversationMessage]:
    messages = list(example.messages)
    if int(max_seq_len) <= 0 or not messages:
        return messages
    if _message_token_length(
        tokenizer=tokenizer,
        messages=messages,
    ) <= int(max_seq_len):
        return messages

    final_turn_start = _final_turn_start_index(messages)
    selected_start = int(final_turn_start)
    for probe_start in range(int(final_turn_start) - 1, -1, -1):
        candidate = messages[probe_start:]
        candidate_len = _message_token_length(
            tokenizer=tokenizer,
            messages=candidate,
        )
        if int(candidate_len) > int(max_seq_len):
            break
        selected_start = int(probe_start)
    return messages[selected_start:]


def encode_supervised_example(
    *,
    tokenizer: SupportsPosttrainTokenizer,
    example: ChatExample,
    max_seq_len: int,
    crop_policy: str = "tail_tokens",
) -> EncodedChatExample:
    crop_policy_name = str(crop_policy or "tail_tokens").strip()
    messages = list(example.messages)
    if crop_policy_name == "preserve_final_turn":
        messages = _select_message_suffix_preserving_final_turn(
            tokenizer=tokenizer,
            example=example,
            max_seq_len=int(max_seq_len),
        )
    encoded = encode_conversation(
        tokenizer=tokenizer,
        messages=messages,
        max_seq_len=int(max_seq_len),
        add_generation_prompt=False,
    )
    return EncodedChatExample(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        labels=encoded.labels,
    )


def encode_completion_example(
    *,
    tokenizer: SupportsPosttrainTokenizer,
    prompt_messages: list[ConversationMessage],
    completion_messages: list[ConversationMessage],
    max_seq_len: int,
) -> EncodedChatExample:
    if not completion_messages:
        raise ValueError("completion_messages must be non-empty")
    full_segments = render_conversation_segments(
        list(prompt_messages) + list(completion_messages),
        add_generation_prompt=False,
    )
    prompt_segments = render_conversation_segments(
        list(prompt_messages),
        add_generation_prompt=True,
    )
    full_untruncated = encode_rendered_segments(
        tokenizer=tokenizer,
        segments=full_segments,
        max_seq_len=0,
    )
    prompt_untruncated = encode_rendered_segments(
        tokenizer=tokenizer,
        segments=prompt_segments,
        max_seq_len=0,
    )
    prompt_prefix_len = int(prompt_untruncated.input_ids.numel())
    if prompt_prefix_len > int(full_untruncated.input_ids.numel()):
        raise ValueError("prompt encoding is longer than full completion encoding")
    if prompt_prefix_len > 0 and not torch.equal(
        full_untruncated.input_ids[:prompt_prefix_len],
        prompt_untruncated.input_ids,
    ):
        raise ValueError("prompt encoding is not an exact prefix of the completion example")

    input_ids = full_untruncated.input_ids
    attention_mask = full_untruncated.attention_mask
    labels = full_untruncated.labels.clone()
    labels[:prompt_prefix_len] = -100

    keep = int(max_seq_len)
    if keep > 0 and int(input_ids.numel()) > keep:
        input_ids = input_ids[-keep:]
        attention_mask = attention_mask[-keep:]
        labels = labels[-keep:]
    return EncodedChatExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


def collate_supervised_examples(
    examples: list[EncodedChatExample],
    *,
    pad_token_id: int,
    pad_to_multiple_of: int = 1,
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty example list")
    seq_len = max(int(example.input_ids.numel()) for example in examples)
    multiple = max(int(pad_to_multiple_of), 1)
    if seq_len % multiple != 0:
        seq_len = ((seq_len + multiple - 1) // multiple) * multiple

    batch_size = len(examples)
    input_ids = torch.full((batch_size, seq_len), int(pad_token_id), dtype=torch.int32)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.int32)
    for row, example in enumerate(examples):
        length = int(example.input_ids.numel())
        input_ids[row, :length] = example.input_ids
        attention_mask[row, :length] = example.attention_mask
        labels[row, :length] = example.labels
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def resolve_batch_pad_token_id(tokenizer: SupportsPosttrainTokenizer) -> int:
    from ml.training.posttrain.tokenizer_runtime import (
        resolve_tokenizer_pad_token_id,
    )

    return resolve_tokenizer_pad_token_id(tokenizer)


def encode_supervised_batch(
    *,
    tokenizer: SupportsPosttrainTokenizer,
    examples: list[ChatExample],
    max_seq_len: int,
    crop_policy: str,
    pad_to_multiple_of: int,
) -> dict[str, torch.Tensor]:
    encoded = [
        encode_supervised_example(
            tokenizer=tokenizer,
            example=example,
            max_seq_len=int(max_seq_len),
            crop_policy=str(crop_policy),
        )
        for example in examples
    ]
    return collate_supervised_examples(
        encoded,
        pad_token_id=resolve_batch_pad_token_id(tokenizer),
        pad_to_multiple_of=int(pad_to_multiple_of),
    )


class SupervisedBatchIterator(SupervisedBatchIteratorCore):
    def __init__(
        self,
        *,
        dataset: list[ChatExample],
        tokenizer: SupportsPosttrainTokenizer,
        batch_size: int,
        max_seq_len: int,
        pad_to_multiple_of: int,
        seed: int,
        shuffle: bool,
        repeat: bool,
        drop_last: bool,
        crop_policy: str = "tail_tokens",
    ) -> None:
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            crop_policy=str(crop_policy),
            pad_to_multiple_of=pad_to_multiple_of,
            seed=seed,
            shuffle=shuffle,
            repeat=repeat,
            drop_last=drop_last,
            encode_supervised_batch_fn=encode_supervised_batch,
        )


__all__ = [
    "ChatExample",
    "EncodedChatExample",
    "ShuffledBatchIterator",
    "SupervisedBatchIterator",
    "_normalize_message",
    "collate_supervised_examples",
    "encode_completion_example",
    "encode_supervised_batch",
    "encode_supervised_example",
    "load_prompt_examples",
    "load_supervised_examples",
]
