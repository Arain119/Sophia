from __future__ import annotations

from typing import Protocol

import torch


class PrefillRuntime(Protocol):
    def reset_runtime_cache(self) -> None: ...

    def runtime_max_seq_len(self) -> int: ...

    def replay_with_cache(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int = 0,
        return_all_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


def maybe_reset_model_cache(model: PrefillRuntime) -> None:
    model.reset_runtime_cache()


def resolve_prefill_chunk_size(
    model: PrefillRuntime,
    prompt_len: int,
    prefill_chunk_size: int | None,
) -> int:
    if prefill_chunk_size is not None:
        return int(prefill_chunk_size)
    max_seq_len = int(model.runtime_max_seq_len())
    return max(1, min(max_seq_len, int(prompt_len)))


def replay_prompt_range(
    model: PrefillRuntime,
    *,
    input_ids: torch.Tensor,
    start_pos: int,
    end_pos: int,
    prefill_chunk_size: int | None,
) -> torch.Tensor:
    if int(end_pos) <= int(start_pos):
        raise ValueError(
            f"end_pos must be > start_pos, got start_pos={start_pos}, end_pos={end_pos}"
        )
    if prefill_chunk_size is None:
        logits, _ = model.replay_with_cache(
            input_ids[:, int(start_pos) : int(end_pos)],
            start_pos=int(start_pos),
            return_all_logits=False,
        )
        return logits

    chunk_size = resolve_prefill_chunk_size(
        model,
        int(end_pos) - int(start_pos),
        prefill_chunk_size,
    )
    logits: torch.Tensor | None = None
    for chunk_start in range(int(start_pos), int(end_pos), int(chunk_size)):
        chunk_end = min(chunk_start + int(chunk_size), int(end_pos))
        logits, _ = model.replay_with_cache(
            input_ids[:, chunk_start:chunk_end],
            start_pos=int(chunk_start),
            return_all_logits=False,
        )
    if logits is None:
        raise RuntimeError("failed to replay prompt range")
    return logits


def prefill_prompt(
    model: PrefillRuntime,
    *,
    input_ids: torch.Tensor,
    input_ids_list: list[list[int]] | None,
    model_device: torch.device,
    prefill_chunk_size: int | None,
) -> torch.Tensor:
    del input_ids_list, model_device
    return replay_prompt_range(
        model,
        input_ids=input_ids,
        start_pos=0,
        end_pos=int(input_ids.size(1)),
        prefill_chunk_size=prefill_chunk_size,
    )


__all__ = [
    "PrefillRuntime",
    "maybe_reset_model_cache",
    "prefill_prompt",
    "replay_prompt_range",
    "resolve_prefill_chunk_size",
]
