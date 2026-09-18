from __future__ import annotations

from ml.runtime.inference.eval.common import torch
from ml.modeling.input_mask import is_all_ones_mask, slice_valid_tokens
from ml.runtime.generation import generate as runtime_generate


def pad_generated_rows(
    rows: list[list[int]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    pad_token_id: int,
) -> torch.Tensor:
    if not rows:
        return torch.empty((0, 0), device=device, dtype=dtype)
    width = max(len(row) for row in rows)
    padded = torch.full(
        (len(rows), int(width)),
        int(pad_token_id),
        device=device,
        dtype=dtype,
    )
    for index, row in enumerate(rows):
        if not row:
            continue
        padded[index, : len(row)] = torch.tensor(
            row,
            device=device,
            dtype=dtype,
        )
    return padded


def generate_with_kv_cache(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    eos_token_id: int | None,
    max_cache_len: int,
    runtime_generate_fn=runtime_generate,
) -> torch.Tensor:
    del max_cache_len
    if int(max_new_tokens) <= 0:
        return input_ids
    if int(input_ids.size(0)) <= 0:
        return input_ids

    def _row_to_tokens(row: torch.Tensor) -> list[int]:
        return [int(token) for token in row.detach().cpu().tolist()]

    if attention_mask is None or is_all_ones_mask(attention_mask):
        prompt_rows = [_row_to_tokens(row) for row in input_ids]
        generated_rows = runtime_generate_fn(
            model,
            prompt_rows,
            max_new_tokens=int(max_new_tokens),
            temperature=(float(temperature) if bool(do_sample) else 0.0),
            top_k=int(top_k),
            top_p=float(top_p),
            eos_id=eos_token_id,
        )
        return pad_generated_rows(
            generated_rows,
            device=input_ids.device,
            dtype=input_ids.dtype,
            pad_token_id=(0 if eos_token_id is None else int(eos_token_id)),
        )

    rows = slice_valid_tokens(input_ids, attention_mask)
    generated_rows: list[list[int]] = []
    for _start, _end, row_tokens in rows:
        generated_rows.extend(
            runtime_generate_fn(
                model,
                [_row_to_tokens(row_tokens.squeeze(0))],
                max_new_tokens=int(max_new_tokens),
                temperature=(float(temperature) if bool(do_sample) else 0.0),
                top_k=int(top_k),
                top_p=float(top_p),
                eos_id=eos_token_id,
            )
        )
    return pad_generated_rows(
        generated_rows,
        device=input_ids.device,
        dtype=input_ids.dtype,
        pad_token_id=(0 if eos_token_id is None else int(eos_token_id)),
    )


__all__ = [
    "generate_with_kv_cache",
    "pad_generated_rows",
]
