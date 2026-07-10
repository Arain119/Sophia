"""Hugging Face model helpers."""

from __future__ import annotations

import os

import torch
from transformers import PreTrainedModel

from ml.integrations.adapters.hf.remote_code import (
    HF_CAUSAL_LM_CLASS,
    HF_CONFIG_CLASS,
    HF_MODELING_MODULE,
    HF_RUNTIME_FILENAME,
    HF_SUPPORT_FILENAME,
)
from ml.integrations.adapters.hf.support import (
    ensure_auto_map,
)
from ml.integrations.export.model_dir import (
    GradientCheckpointingFn,
    load_local_export_tokenizer,
    local_export_load_kwargs,
)
from ml.integrations.adapters.hf.tokenizer import tokenizer_vocab_size
from ml.modeling.cache_decode import (
    prefill_runtime_cache,
)
from ml.modeling.input_mask import (
    is_all_ones_mask,
    slice_valid_tokens,
)

local_model_load_kwargs = local_export_load_kwargs
model_load_kwargs = local_export_load_kwargs


def load_trainable_model(
    *,
    model_cls: type[PreTrainedModel],
    model_dir: str,
    device: torch.device,
    base_dtype: torch.dtype,
    gradient_checkpointing: bool,
    apply_gradient_checkpointing_fn: GradientCheckpointingFn,
) -> PreTrainedModel:
    resolved_model_dir = os.path.abspath(str(model_dir))
    model = model_cls.from_pretrained(
        resolved_model_dir,
        **local_model_load_kwargs(),
    )
    model = model.to(device=device, dtype=base_dtype)
    model.train(True)
    config = getattr(model, "config", None)
    if config is not None:
        config.return_logits_in_train = True
        config.use_cache = False
    apply_gradient_checkpointing_fn(model, enabled=bool(gradient_checkpointing))
    return model


__all__ = [
    "HF_CAUSAL_LM_CLASS",
    "HF_CONFIG_CLASS",
    "HF_MODELING_MODULE",
    "HF_RUNTIME_FILENAME",
    "HF_SUPPORT_FILENAME",
    "ensure_auto_map",
    "is_all_ones_mask",
    "local_model_load_kwargs",
    "load_local_export_tokenizer",
    "load_trainable_model",
    "model_load_kwargs",
    "prefill_runtime_cache",
    "slice_valid_tokens",
    "tokenizer_vocab_size",
]
