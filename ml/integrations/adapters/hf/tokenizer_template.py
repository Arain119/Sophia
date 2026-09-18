from __future__ import annotations

from typing import Protocol

import torch
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding

from ml.modeling.text.conversation import ConversationMessage
from ml.modeling.text.conversation import render_conversation_segments

ChatConversation = list[ConversationMessage]
ChatConversationBatch = list[ChatConversation]
TokenizerChatInput = ChatConversation | ChatConversationBatch
TokenizerTemplateOutput = (
    str | list[str] | list[int] | list[list[int]] | BatchEncoding
)


class SupportsApplyChatTemplate(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


def ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> None:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return

    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_tok = getattr(tokenizer, "eos_token", None)
    if isinstance(eos_tok, str) and eos_tok:
        try:
            tokenizer.pad_token = eos_tok
        except (AttributeError, ValueError):
            pass
    if getattr(tokenizer, "pad_token_id", None) is None and isinstance(eos_id, int):
        try:
            tokenizer.pad_token_id = int(eos_id)
        except (AttributeError, ValueError):
            pass


def set_model_max_length(
    tokenizer: PreTrainedTokenizerBase,
    *,
    model_max_length: int,
) -> None:
    max_len = int(model_max_length)
    if max_len <= 0:
        return
    tokenizer.model_max_length = int(max_len)
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        init_kwargs["model_max_length"] = int(max_len)


def truncate_sequence(
    values: list[int],
    *,
    max_length: int,
    side: str,
) -> list[int]:
    limit = int(max_length)
    if limit <= 0 or len(values) <= limit:
        return list(values)
    if str(side).strip().lower() == "left":
        return list(values[-limit:])
    return list(values[:limit])


def tensorize_chat_field(
    rows: list[list[int]],
    *,
    return_tensors: str,
) -> torch.Tensor:
    tensor_type = str(return_tensors).strip().lower()
    if tensor_type == "pt":
        return torch.tensor(rows, dtype=torch.long)
    raise ValueError(f"unsupported return_tensors: {return_tensors!r}")


def delegate_chat_template(
    *,
    tokenizer: PreTrainedTokenizerBase,
    conversation: TokenizerChatInput,
    chat_template: str | None,
    add_generation_prompt: bool,
    continue_final_message: bool,
    tokenize: bool,
    padding: bool | str,
    truncation: bool,
    max_length: int | None,
    return_tensors: str | None,
    return_dict: bool,
    return_assistant_tokens_mask: bool,
    **kwargs: object,
) -> TokenizerTemplateOutput:
    original_apply = getattr(tokenizer, "apply_chat_template", None)
    if original_apply is None or not callable(original_apply):
        raise RuntimeError("tokenizer does not expose apply_chat_template")
    return original_apply(
        conversation,
        documents=None,
        chat_template=chat_template,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        tokenize=tokenize,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors,
        return_dict=return_dict,
        return_assistant_tokens_mask=return_assistant_tokens_mask,
        **kwargs,
    )


def apply_runtime_chat_template(
    *,
    tokenizer: PreTrainedTokenizerBase,
    default_template: str | None,
    conversation: TokenizerChatInput,
    documents: object,
    chat_template: str | None,
    add_generation_prompt: bool,
    continue_final_message: bool,
    tokenize: bool,
    padding: bool | str,
    truncation: bool,
    max_length: int | None,
    return_tensors: str | None,
    return_dict: bool,
    return_assistant_tokens_mask: bool,
    tokenizer_kwargs: dict[str, object] | None,
    **kwargs: object,
) -> TokenizerTemplateOutput:
    del documents, tokenizer_kwargs
    should_delegate = bool(
        continue_final_message
        or (chat_template is not None and chat_template != default_template)
        or kwargs
    )
    if should_delegate:
        return delegate_chat_template(
            tokenizer=tokenizer,
            conversation=conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_dict=return_dict,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            **kwargs,
        )

    is_batched = bool(
        isinstance(conversation, list) and conversation and isinstance(conversation[0], list)
    )
    conversations = list(conversation) if is_batched else [conversation]

    rendered_texts: list[str] = []
    rendered_ids: list[list[int]] = []
    rendered_masks: list[list[int]] = []

    for messages in conversations:
        if not isinstance(messages, list):
            raise TypeError(
                "conversation must be a list of messages or a list of conversations"
            )
        segments = render_conversation_segments(
            messages,
            add_generation_prompt=bool(add_generation_prompt),
        )
        rendered_texts.append("".join(segment.text for segment in segments))
        ids: list[int] = []
        assistant_mask: list[int] = []
        for segment in segments:
            token_ids = tokenizer.encode(str(segment.text), add_special_tokens=False)
            ids.extend(int(token_id) for token_id in token_ids)
            if bool(return_assistant_tokens_mask):
                assistant_mask.extend([1 if segment.supervise else 0] * len(token_ids))
        if bool(truncation):
            target_len = int(max_length or getattr(tokenizer, "model_max_length", 0) or 0)
            if target_len > 0:
                trunc_side = str(
                    getattr(tokenizer, "truncation_side", "right") or "right"
                )
                ids = truncate_sequence(ids, max_length=target_len, side=trunc_side)
                if bool(return_assistant_tokens_mask):
                    assistant_mask = truncate_sequence(
                        assistant_mask,
                        max_length=target_len,
                        side=trunc_side,
                    )
        rendered_ids.append(ids)
        if bool(return_assistant_tokens_mask):
            rendered_masks.append(assistant_mask)

    if not bool(tokenize):
        return rendered_texts if is_batched else rendered_texts[0]

    if bool(padding):
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if not isinstance(pad_id, int):
            raise ValueError("tokenizer must define pad_token_id for padding")
        pad_side = str(getattr(tokenizer, "padding_side", "right") or "right").lower()
        max_len = max((len(row) for row in rendered_ids), default=0)
        for idx, row in enumerate(rendered_ids):
            pad_width = max_len - len(row)
            if pad_width <= 0:
                continue
            pad_tokens = [int(pad_id)] * pad_width
            pad_mask = [0] * pad_width
            if pad_side == "left":
                rendered_ids[idx] = pad_tokens + row
                if bool(return_assistant_tokens_mask):
                    rendered_masks[idx] = pad_mask + rendered_masks[idx]
            else:
                rendered_ids[idx] = row + pad_tokens
                if bool(return_assistant_tokens_mask):
                    rendered_masks[idx] = rendered_masks[idx] + pad_mask

    attention_rows = [[1] * len(row) for row in rendered_ids]
    if bool(padding):
        attention_rows = [
            [
                0 if token_id == getattr(tokenizer, "pad_token_id", None) else 1
                for token_id in row
            ]
            for row in rendered_ids
        ]

    if return_tensors is not None:
        encoded: dict[str, object] = {
            "input_ids": tensorize_chat_field(
                rendered_ids,
                return_tensors=return_tensors,
            ),
            "attention_mask": tensorize_chat_field(
                attention_rows,
                return_tensors=return_tensors,
            ),
        }
        if bool(return_assistant_tokens_mask):
            encoded["assistant_masks"] = tensorize_chat_field(
                rendered_masks,
                return_tensors=return_tensors,
            )
        return BatchEncoding(encoded)

    if bool(return_dict):
        encoded = {
            "input_ids": rendered_ids if is_batched else rendered_ids[0],
            "attention_mask": attention_rows if is_batched else attention_rows[0],
        }
        if bool(return_assistant_tokens_mask):
            encoded["assistant_masks"] = (
                rendered_masks if is_batched else rendered_masks[0]
            )
        return BatchEncoding(encoded)

    return rendered_ids if is_batched else rendered_ids[0]


class RuntimeTokenizer:
    """Explicit Sophia tokenizer API that owns chat-template semantics."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(
            self,
            "_default_chat_template",
            getattr(tokenizer, "chat_template", None),
        )

    @property
    def base_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def __getattr__(self, name: str) -> object:
        return getattr(self._tokenizer, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        setattr(self._tokenizer, name, value)

    def __len__(self) -> int:
        return len(self._tokenizer)

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self._tokenizer(*args, **kwargs)

    def apply_chat_template(
        self,
        conversation: TokenizerChatInput,
        documents: object = None,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: bool = True,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: dict[str, object] | None = None,
        **kwargs: object,
    ) -> TokenizerTemplateOutput:
        return apply_runtime_chat_template(
            tokenizer=self._tokenizer,
            default_template=self._default_chat_template,
            conversation=conversation,
            documents=documents,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_dict=return_dict,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )


__all__ = [
    "ChatConversation",
    "ChatConversationBatch",
    "RuntimeTokenizer",
    "SupportsApplyChatTemplate",
    "TokenizerChatInput",
    "TokenizerTemplateOutput",
    "apply_runtime_chat_template",
    "delegate_chat_template",
    "ensure_pad_token",
    "set_model_max_length",
    "tensorize_chat_field",
    "truncate_sequence",
]
