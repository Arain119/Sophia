"""Local export writers for Sophia and remote-code model bundles."""

from __future__ import annotations

import contextlib
import os
from typing import Protocol

import torch

from ml.integrations.adapters.hf.bundle import save_config_bundle
from ml.integrations.export.runtime_packager import write_remote_code_bundle
from ml.integrations.adapters.hf.loaders import ensure_auto_map


class SupportsApplyToModel(Protocol):
    def apply_to_model(self) -> contextlib.AbstractContextManager[object]: ...


def export_model_artifacts(
    *,
    model: torch.nn.Module,
    tokenizer: object,
    output_dir: str,
    safe_serialization: bool,
    ema: SupportsApplyToModel | None = None,
) -> str:
    export_dir = os.path.abspath(str(output_dir))
    os.makedirs(export_dir, exist_ok=True)
    ctx = ema.apply_to_model() if ema is not None else contextlib.nullcontext()
    with ctx:
        save_pretrained = getattr(model, "save_pretrained", None)
        if not callable(save_pretrained):
            raise RuntimeError("model export requires a save_pretrained method")
        if hasattr(getattr(model, "config", None), "architectures"):
            ensure_auto_map(model)
        save_pretrained(
            export_dir,
            safe_serialization=bool(safe_serialization),
        )
        config = getattr(model, "config", None)
        if config is not None:
            save_config_bundle(config, save_directory=export_dir)
    tokenizer_save_pretrained = getattr(tokenizer, "save_pretrained", None)
    if callable(tokenizer_save_pretrained):
        tokenizer_save_pretrained(export_dir)
    write_remote_code_bundle(output_dir=export_dir)
    return export_dir


__all__ = ["export_model_artifacts"]
