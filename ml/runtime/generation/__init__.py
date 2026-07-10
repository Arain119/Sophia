"""Generation helpers for the active Sophia runtime route."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ml.runtime.generation.decode import decode_from_prefill, sample_next_token
from ml.runtime.generation.prefill import (
    model_cache_namespace,
    maybe_reset_model_cache,
    prefill_prompt,
    prefix_cache_path,
    save_snapshot_to_disk,
)
from ml.runtime.generation.prefix_cache import PrefixCache, PrefixSnapshot
from ml.runtime.generation.request import (
    normalize_generation_inputs,
    validate_generate_request,
)
from ml.runtime.api import resolve_runtime_lock


def generate(
    model: nn.Module,
    input_ids: torch.Tensor | list[list[int]],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    eos_id: int | None = None,
    prefix_cache: PrefixCache | None = None,
    prefix_cache_min_tokens: int = 1,
    prefix_cache_dir: str | Path | None = None,
    prefill_chunk_size: int | None = None,
) -> list[list[int]]:
    validate_generate_request(
        model=model,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        eos_id=eos_id,
        prefix_cache=prefix_cache,
        prefix_cache_min_tokens=int(prefix_cache_min_tokens),
        prefix_cache_dir=prefix_cache_dir,
        prefill_chunk_size=prefill_chunk_size,
    )

    model_device = next(model.parameters()).device
    runtime_lock = resolve_runtime_lock(model)

    def _run_generation() -> list[list[int]]:
        was_training = bool(model.training)
        model.eval()
        try:
            return _generate_impl(
                model=model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_id=eos_id,
                prefix_cache=prefix_cache,
                prefix_cache_min_tokens=prefix_cache_min_tokens,
                prefix_cache_dir=prefix_cache_dir,
                prefill_chunk_size=prefill_chunk_size,
                model_device=model_device,
            )
        finally:
            if was_training:
                model.train()

    if runtime_lock is None:
        return _run_generation()
    with runtime_lock:
        return _run_generation()


def _generate_impl(
    *,
    model: nn.Module,
    input_ids: torch.Tensor | list[list[int]],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_id: int | None,
    prefix_cache: PrefixCache | None,
    prefix_cache_min_tokens: int,
    prefix_cache_dir: str | Path | None,
    prefill_chunk_size: int | None,
    model_device: torch.device,
) -> list[list[int]]:
    request = normalize_generation_inputs(
        model=model,
        input_ids=input_ids,
        max_new_tokens=int(max_new_tokens),
        prefix_cache=prefix_cache,
        prefix_cache_dir=prefix_cache_dir,
        model_device=model_device,
    )
    input_ids = request.input_ids
    input_ids_list = request.input_ids_list
    batch_size = int(request.batch_size)
    prompt_len = int(request.prompt_len)

    maybe_reset_model_cache(model)

    with torch.no_grad():
        prefill_logits = prefill_prompt(
            model,
            input_ids=input_ids,
            input_ids_list=input_ids_list,
            model_device=model_device,
            prefix_cache=prefix_cache,
            prefix_cache_min_tokens=int(prefix_cache_min_tokens),
            prefix_cache_dir=prefix_cache_dir,
            prefill_chunk_size=prefill_chunk_size,
        )
        return decode_from_prefill(
            model=model,
            input_ids=input_ids,
            prefill_logits=prefill_logits,
            model_device=model_device,
            prompt_len=int(prompt_len),
            batch_size=int(batch_size),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            eos_id=eos_id,
        )


__all__ = [
    "PrefixCache",
    "PrefixSnapshot",
    "generate",
    "model_cache_namespace",
    "prefix_cache_path",
    "sample_next_token",
    "save_snapshot_to_disk",
]
