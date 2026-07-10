from __future__ import annotations

import torch

from ml.modeling.text.conversation_renderer import render_conversation_segments
from ml.modeling.text.conversation_types import (
    ConversationEncoding,
    ConversationMessage,
    RenderedSegment,
    SupportsEncodeText,
)


def encode_rendered_segments(
    *,
    tokenizer: SupportsEncodeText,
    segments: list[RenderedSegment],
    max_seq_len: int,
) -> ConversationEncoding:
    input_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    for segment in segments:
        token_ids = tokenizer.encode(str(segment.text), add_special_tokens=False)
        if not token_ids:
            continue
        ids = torch.tensor(token_ids, dtype=torch.long)
        input_chunks.append(ids)
        if segment.supervise:
            label_chunks.append(ids.clone())
        else:
            label_chunks.append(torch.full_like(ids, -100))
    if not input_chunks:
        raise ValueError("rendered conversation produced zero tokens")
    input_ids = torch.cat(input_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    if int(max_seq_len) > 0 and int(input_ids.numel()) > int(max_seq_len):
        keep = int(max_seq_len)
        input_ids = input_ids[-keep:]
        labels = labels[-keep:]
    attention_mask = torch.ones_like(input_ids)
    return ConversationEncoding(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


def encode_conversation(
    *,
    tokenizer: SupportsEncodeText,
    messages: list[ConversationMessage],
    max_seq_len: int,
    add_generation_prompt: bool,
) -> ConversationEncoding:
    segments = render_conversation_segments(
        list(messages),
        add_generation_prompt=bool(add_generation_prompt),
    )
    return encode_rendered_segments(
        tokenizer=tokenizer,
        segments=segments,
        max_seq_len=int(max_seq_len),
    )


__all__ = ["encode_conversation", "encode_rendered_segments"]
