from __future__ import annotations

import torch
from torch import nn

from ml.runtime.generation.sampler import apply_top_k_top_p


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float = 1.0,
) -> torch.Tensor:
    if float(temperature) == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / float(temperature)
    logits = apply_top_k_top_p(logits, top_k=int(top_k), top_p=float(top_p))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def sample_token_from_last_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float = 1.0,
) -> torch.Tensor:
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    if logits.dim() != 2:
        raise ValueError(
            f"logits must be [B,V] or [B,T,V], got shape {tuple(logits.shape)}"
        )
    return sample_next_token(
        logits,
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
    )


def update_finished_state(
    *,
    next_token: torch.Tensor,
    finished: torch.Tensor | None,
    stop_lens: torch.Tensor | None,
    eos_id: int | None,
    stop_len_value: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if finished is None or stop_lens is None or eos_id is None:
        return next_token, finished, stop_lens
    eos_fill = torch.full_like(next_token, int(eos_id))
    next_token = torch.where(finished.unsqueeze(1), eos_fill, next_token)
    newly_finished = (~finished) & (next_token.squeeze(1) == int(eos_id))
    stop_lens = torch.where(
        newly_finished,
        torch.full_like(stop_lens, int(stop_len_value)),
        stop_lens,
    )
    finished = finished | newly_finished
    return next_token, finished, stop_lens


def forward_decode_step(
    model: nn.Module,
    token: torch.Tensor,
    *,
    start_pos: int,
) -> torch.Tensor:
    logits, _ = model.replay_with_cache(
        token,
        start_pos=int(start_pos),
        return_all_logits=False,
    )
    return logits


def _write_generated_token(
    generated: torch.Tensor,
    *,
    position: int,
    token: torch.Tensor,
) -> None:
    if int(token.dim()) != 2 or int(token.size(1)) != 1:
        raise ValueError(f"generated token must be [B,1], got {tuple(token.shape)}")
    generated[:, position : position + 1].copy_(token)


def trim_generated_sequences(
    generated: torch.Tensor,
    *,
    total_len: int,
    stop_lens: torch.Tensor | None,
) -> list[list[int]]:
    rows = generated[:, : int(total_len)].detach().cpu().tolist()
    if stop_lens is None:
        return rows
    trimmed: list[list[int]] = []
    for idx, row in enumerate(rows):
        stop_len = int(stop_lens[idx].item())
        trimmed.append(row if stop_len <= 0 else row[:stop_len])
    return trimmed


def decode_from_prefill(
    *,
    model: nn.Module,
    input_ids: torch.Tensor,
    prefill_logits: torch.Tensor,
    model_device: torch.device,
    prompt_len: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_id: int | None,
) -> list[list[int]]:
    total_capacity = int(prompt_len) + int(max_new_tokens)
    generated = input_ids.new_empty((int(batch_size), total_capacity))
    generated[:, : int(prompt_len)].copy_(input_ids)
    current_pos = int(prompt_len)
    finished: torch.Tensor | None = None
    stop_lens: torch.Tensor | None = None
    if eos_id is not None:
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=model_device)
        stop_lens = torch.zeros((batch_size,), dtype=torch.int64, device=model_device)

    if int(max_new_tokens) <= 0:
        return trim_generated_sequences(
            generated,
            total_len=int(prompt_len),
            stop_lens=stop_lens,
        )

    next_token = sample_token_from_last_logits(
        prefill_logits,
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
    )
    next_token, finished, stop_lens = update_finished_state(
        next_token=next_token,
        finished=finished,
        stop_lens=stop_lens,
        eos_id=eos_id,
        stop_len_value=int(prompt_len) + 1,
    )
    _write_generated_token(generated, position=int(current_pos), token=next_token)
    current_pos += 1
    if finished is not None and bool(finished.all().item()):
        return trim_generated_sequences(
            generated,
            total_len=int(current_pos),
            stop_lens=stop_lens,
        )

    generated_tokens = 1
    while generated_tokens < int(max_new_tokens):
        logits = forward_decode_step(
            model,
            generated[:, int(current_pos) - 1 : int(current_pos)],
            start_pos=int(current_pos),
        )
        accepted_tokens = [
            sample_token_from_last_logits(
                logits,
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
            )
        ]

        for token in accepted_tokens:
            if generated_tokens >= int(max_new_tokens):
                break
            token, finished, stop_lens = update_finished_state(
                next_token=token,
                finished=finished,
                stop_lens=stop_lens,
                eos_id=eos_id,
                stop_len_value=int(prompt_len) + generated_tokens + 1,
            )
            _write_generated_token(
                generated,
                position=int(current_pos),
                token=token,
            )
            current_pos += 1
            generated_tokens += 1
            if finished is not None and bool(finished.all().item()):
                return trim_generated_sequences(
                    generated,
                    total_len=int(current_pos),
                    stop_lens=stop_lens,
                )

    return trim_generated_sequences(
        generated,
        total_len=int(current_pos),
        stop_lens=stop_lens,
    )


__all__ = [
    "decode_from_prefill",
    "forward_decode_step",
    "sample_next_token",
    "sample_token_from_last_logits",
    "torch",
    "trim_generated_sequences",
    "update_finished_state",
]
