from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml.runtime.generation.prefix_cache import PrefixCache
from ml.runtime.generation.request import (
    normalize_generation_inputs,
    validate_generate_request,
)
from ml.runtime.model.transformer import ModelArgs, Transformer


def _tiny_model() -> Transformer:
    args = ModelArgs(
        max_batch_size=1,
        max_seq_len=32,
        vocab_size=64,
        dim=64,
        n_layers=2,
        n_heads=4,
        head_dim=16,
        num_key_value_heads=2,
        rope_head_dim=8,
        ffn_hidden=192,
    )
    return Transformer(args).eval()


def test_validate_generate_request_accepts_prefix_cache_directory() -> None:
    model = _tiny_model()

    validate_generate_request(
        model=model,
        max_new_tokens=2,
        temperature=0.0,
        top_k=0,
        eos_id=None,
        prefix_cache=PrefixCache(max_entries=4),
        prefix_cache_min_tokens=1,
        prefix_cache_dir=None,
        prefill_chunk_size=None,
    )


def test_validate_generate_request_rejects_negative_temperature() -> None:
    model = _tiny_model()

    with pytest.raises(ValueError, match="temperature must be >= 0"):
        validate_generate_request(
            model=model,
            max_new_tokens=2,
            temperature=-0.1,
            top_k=0,
            eos_id=None,
            prefix_cache=None,
            prefix_cache_min_tokens=1,
            prefix_cache_dir=None,
            prefill_chunk_size=None,
        )


def test_normalize_generation_inputs_rejects_batch_prefix_cache() -> None:
    model = _tiny_model()

    with pytest.raises(ValueError, match="prefix caching only supports batch_size == 1"):
        normalize_generation_inputs(
            model=model,
            input_ids=[[1, 2], [3, 4]],
            max_new_tokens=1,
            prefix_cache=PrefixCache(max_entries=4),
            prefix_cache_dir=None,
            model_device=torch.device("cpu"),
        )


def test_normalize_generation_inputs_materializes_list_input() -> None:
    model = _tiny_model()

    normalized = normalize_generation_inputs(
        model=model,
        input_ids=[[1, 2, 3]],
        max_new_tokens=2,
        prefix_cache=None,
        prefix_cache_dir=Path("/tmp/prefix-cache"),
        model_device=torch.device("cpu"),
    )

    assert tuple(normalized.input_ids.shape) == (1, 3)
    assert normalized.input_ids_list == [[1, 2, 3]]
    assert normalized.batch_size == 1
    assert normalized.prompt_len == 3
    assert normalized.prefix_cache_enabled is True
