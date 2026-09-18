from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NormalizedGenerationInputs:
    input_ids: torch.Tensor
    input_ids_list: list[list[int]] | None
    batch_size: int
    prompt_len: int


def validate_generate_request(
    *,
    model: nn.Module,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    eos_id: int | None,
    prefill_chunk_size: int | None,
) -> None:
    del model, eos_id
    if prefill_chunk_size is not None and int(prefill_chunk_size) <= 0:
        raise ValueError(f"prefill_chunk_size must be > 0 when provided, got {prefill_chunk_size}")
    if not torch.isfinite(torch.tensor(float(temperature))):
        raise ValueError(f"temperature must be finite, got {temperature}")
    if float(temperature) < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if int(top_k) < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")
    if int(max_new_tokens) < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")


def normalize_generation_inputs(
    *,
    model: nn.Module,
    input_ids: torch.Tensor | list[list[int]],
    max_new_tokens: int,
    model_device: torch.device,
) -> NormalizedGenerationInputs:
    if not hasattr(model, "runtime_max_seq_len"):
        raise TypeError(
            "generate requires a runtime model exposing runtime_max_seq_len()."
        )
    if isinstance(input_ids, list):
        if not input_ids:
            raise ValueError("input_ids must contain at least one sequence")
        seq_lens = {len(row) for row in input_ids}
        if 0 in seq_lens:
            raise ValueError("input_ids must contain at least one token per sequence")
        if len(seq_lens) != 1:
            raise ValueError(
                "list input_ids must be rectangular; pad to a tensor before calling generate"
            )
        input_ids_list = input_ids
        tensor_input_ids = torch.tensor(input_ids, dtype=torch.long, device=model_device)
    else:
        tensor_input_ids = input_ids.to(device=model_device, dtype=torch.long)
        input_ids_list = None

    if tensor_input_ids.dim() != 2:
        raise ValueError(
            f"input_ids must be rank 2 [B, T], got shape {tuple(tensor_input_ids.shape)}"
        )
    batch_size = int(tensor_input_ids.size(0))
    prompt_len = int(tensor_input_ids.size(1))
    if batch_size <= 0:
        raise ValueError("input_ids must contain at least one sequence")
    if prompt_len <= 0:
        raise ValueError("input_ids must contain at least one token")
    max_seq_len = int(model.runtime_max_seq_len())
    if prompt_len + int(max_new_tokens) > max_seq_len:
        raise ValueError(
            "prompt plus generation exceeds max_seq_len: "
            f"prompt_len={prompt_len}, max_new_tokens={int(max_new_tokens)}, max_seq_len={max_seq_len}"
        )

    return NormalizedGenerationInputs(
        input_ids=tensor_input_ids,
        input_ids_list=input_ids_list,
        batch_size=int(batch_size),
        prompt_len=int(prompt_len),
    )
