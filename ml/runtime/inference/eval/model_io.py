from __future__ import annotations

import os

from ml.integrations.export.model_dir import (
    load_local_export_tokenizer,
    load_trainable_decoder_from_export_dir as load_trainable_decoder,
)
from ml.errors import SophiaUsageError
from ml.modeling.config_introspection import (
    resolve_config_max_position_embeddings,
)
from ml.runtime.inference.contracts import (
    InferenceTokenizer,
    SupportsConfiguredModel,
)
from ml.runtime.inference.eval.common import (
    PRETRAIN_CHECK_DIRNAME,
    SOPHIA_DIM,
    torch,
)
from ml.runtime.inference.eval.runtime_config import EvalRuntimeConfig


def resolve_max_pos(
    model: SupportsConfiguredModel,
    tokenizer: InferenceTokenizer | None,
) -> int | None:
    config = getattr(model, "config", None)
    max_pos = resolve_config_max_position_embeddings(config)
    if max_pos is not None:
        return int(max_pos)
    max_pos = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_pos, int) and 0 < int(max_pos) < 1_000_000_000:
        return int(max_pos)
    return None


def init_model(config: EvalRuntimeConfig) -> tuple[torch.nn.Module, InferenceTokenizer]:
    export_dir = str(config.export_dir or "").strip()
    if not export_dir:
        candidates = (
            os.path.join(str(config.save_dir), PRETRAIN_CHECK_DIRNAME),
            os.path.join(str(config.save_dir), f"sophia_{int(SOPHIA_DIM)}_export"),
        )
        for candidate in candidates:
            if os.path.isdir(candidate) and os.path.exists(
                os.path.join(candidate, "config.json")
            ):
                export_dir = candidate
                break

    if not export_dir:
        raise SophiaUsageError(
            "Unable to locate a local exported Sophia model directory.\n"
            f"Prepare `out/{PRETRAIN_CHECK_DIRNAME}/` or `out/sophia_{int(SOPHIA_DIM)}_export/`, or pass `--export_dir` explicitly.\n"
            f"Example: python -m ml.cli.eval --export_dir out/{PRETRAIN_CHECK_DIRNAME}"
        )

    print(f"Using local Sophia export directory: {export_dir}")
    if not os.path.isdir(export_dir):
        raise SophiaUsageError(
            f"[ERR] export_dir must be a local export directory (remote downloads are disabled): {export_dir}"
        )

    target_device = torch.device(str(config.device))
    target_dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
    model = load_trainable_decoder(
        export_dir=export_dir,
        device=target_device,
        base_dtype=target_dtype,
        gradient_checkpointing=False,
        apply_gradient_checkpointing_fn=lambda *_args, **_kwargs: None,
    )

    max_ctx = resolve_max_pos(model, tokenizer=None)
    tokenizer = load_local_export_tokenizer(
        export_dir,
        padding_side="left",
        truncation_side="left",
        model_max_length=max_ctx,
    )
    model = model.eval().to(device=target_device, dtype=target_dtype)
    model.config.use_cache = bool(config.use_cache)
    return model, tokenizer


__all__ = ["init_model", "resolve_max_pos"]
