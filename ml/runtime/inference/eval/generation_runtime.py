from __future__ import annotations

from ml.runtime.inference.eval.common import torch
from ml.modeling.input_mask import is_all_ones_mask, slice_valid_tokens
from ml.runtime.controls import (
    maybe_refresh_runtime_state_buffers,
    maybe_reset_runtime_cache,
)
from ml.runtime.generation import generate as runtime_generate
from ml.runtime.generation.decode import (
    sample_token_from_last_logits as runtime_sample_token_from_last_logits,
)


def sample_next_token(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    runtime_temperature = float(temperature) if bool(do_sample) else 0.0
    if bool(do_sample) and not (runtime_temperature > 0.0):
        raise ValueError("temperature must be > 0 when sampling")
    return runtime_sample_token_from_last_logits(
        logits,
        temperature=float(runtime_temperature),
        top_k=int(top_k),
        top_p=float(top_p),
    ).squeeze(-1)


def reset_runtime_state_if_available(model: torch.nn.Module) -> None:
    if maybe_refresh_runtime_state_buffers(model):
        return
    maybe_reset_runtime_cache(model)


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


def naive_generate_no_cache(
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
    sample_next_token_fn=sample_next_token,
    reset_runtime_state_if_available_fn=reset_runtime_state_if_available,
) -> torch.Tensor:
    if int(max_new_tokens) <= 0:
        return input_ids

    device = input_ids.device
    bsz = int(input_ids.size(0))
    prompt_len = int(input_ids.size(1))
    total_len = int(prompt_len) + int(max_new_tokens)
    out_ids = torch.empty((bsz, total_len), device=device, dtype=input_ids.dtype)
    out_ids[:, :prompt_len] = input_ids

    out_mask = None
    one_mask = None
    if attention_mask is not None:
        one_mask = torch.ones((bsz, 1), device=device, dtype=attention_mask.dtype)
        out_mask = torch.empty(
            (bsz, total_len), device=device, dtype=attention_mask.dtype
        )
        out_mask[:, :prompt_len] = attention_mask

    finished = None
    eos_id = None
    if eos_token_id is not None:
        eos_id = int(eos_token_id)
        finished = torch.zeros((bsz,), device=device, dtype=torch.bool)

    cur_len = int(prompt_len)
    for _ in range(int(max_new_tokens)):
        reset_runtime_state_if_available_fn(model)
        out = model(
            input_ids=out_ids[:, :cur_len],
            attention_mask=(out_mask[:, :cur_len] if out_mask is not None else None),
            use_cache=False,
            logits_to_keep=1,
        )
        logits = out.logits
        if logits is None:
            raise RuntimeError("Model returned logits=None for generation step.")
        next_id = sample_next_token_fn(
            logits[:, -1, :],
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
        )

        if finished is not None and eos_id is not None:
            eos_fill = torch.full_like(next_id, int(eos_id))
            next_id = torch.where(finished, eos_fill, next_id)
            finished = finished | (next_id == int(eos_id))

        if next_id.dtype != out_ids.dtype:
            next_id = next_id.to(dtype=out_ids.dtype)
        if int(cur_len) >= int(total_len):
            break
        out_ids[:, cur_len] = next_id
        if out_mask is not None and one_mask is not None:
            out_mask[:, cur_len] = one_mask.squeeze(1)
        cur_len += 1

        if finished is not None and bool(finished.all().item()):
            break

    return out_ids[:, :cur_len]


__all__ = [
    "generate_with_kv_cache",
    "naive_generate_no_cache",
    "pad_generated_rows",
    "reset_runtime_state_if_available",
    "sample_next_token",
]
