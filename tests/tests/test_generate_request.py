from __future__ import annotations

import pytest
import torch

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
        n_layers=4,
        num_heads=4,
        head_dim=16,
        ffn_hidden=192,
        kda_decay_rank=16,
        kda_output_gate_rank=16,
        mla_q_rank=16,
        mla_kv_rank=16,
    )
    return Transformer(args).eval()


def test_validate_generate_request_rejects_negative_temperature() -> None:
    model = _tiny_model()

    with pytest.raises(ValueError, match="temperature must be >= 0"):
        validate_generate_request(
            model=model,
            max_new_tokens=2,
            temperature=-0.1,
            top_k=0,
            eos_id=None,
            prefill_chunk_size=None,
        )


def test_normalize_generation_inputs_materializes_list_input() -> None:
    model = _tiny_model()

    normalized = normalize_generation_inputs(
        model=model,
        input_ids=[[1, 2, 3]],
        max_new_tokens=2,
        model_device=torch.device("cpu"),
    )

    assert tuple(normalized.input_ids.shape) == (1, 3)
    assert normalized.input_ids_list == [[1, 2, 3]]
    assert normalized.batch_size == 1
    assert normalized.prompt_len == 3
