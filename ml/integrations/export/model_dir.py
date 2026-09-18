"""Local exported-model directory helpers."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable

import torch

from ml.modeling.sophia_decoder import SophiaDecoder
from ml.integrations.adapters.hf.tokenizer import load_local_tokenizer


GradientCheckpointingFn = Callable[..., None]


def local_export_load_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {
        "dtype": torch.float32,
        "local_files_only": True,
    }
    if importlib.util.find_spec("accelerate") is not None:
        kwargs["low_cpu_mem_usage"] = True
    return kwargs


load_local_export_tokenizer = load_local_tokenizer


def load_trainable_decoder_from_export_dir(
    *,
    export_dir: str,
    device: torch.device,
    base_dtype: torch.dtype,
    gradient_checkpointing: bool,
    gradient_checkpointing_exclude_first: int,
    gradient_checkpointing_exclude_last: int,
    apply_gradient_checkpointing_fn: GradientCheckpointingFn,
) -> SophiaDecoder:
    model = SophiaDecoder.from_pretrained(
        os.path.abspath(str(export_dir)),
        device=device,
        dtype=base_dtype,
    )
    model.train(True)
    config = getattr(model, "config", None)
    if config is not None:
        config.return_logits_in_train = True
        config.use_cache = False
    exclude_first = int(gradient_checkpointing_exclude_first)
    exclude_last = int(gradient_checkpointing_exclude_last)
    if min(exclude_first, exclude_last) < 0:
        raise ValueError("gradient checkpoint exclusion counts must be >= 0")
    if not bool(gradient_checkpointing) and (exclude_first or exclude_last):
        raise ValueError(
            "gradient checkpoint exclusions require gradient_checkpointing=1"
        )
    if bool(gradient_checkpointing) and exclude_first + exclude_last >= int(
        model.config.n_layers
    ):
        raise ValueError(
            "gradient checkpoint exclusions must leave at least one checkpointed layer"
        )
    model.apply_runtime_recipe_knobs(
        loss_chunk_size=int(model.config.loss_chunk_size),
        gradient_checkpointing_exclude_first=exclude_first,
        gradient_checkpointing_exclude_last=exclude_last,
    )
    apply_gradient_checkpointing_fn(model, enabled=bool(gradient_checkpointing))
    return model


__all__ = [
    "load_local_export_tokenizer",
    "load_trainable_decoder_from_export_dir",
    "local_export_load_kwargs",
]
