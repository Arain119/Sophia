from __future__ import annotations

import os

from transformers import GenerationConfig

from ml.integrations.adapters.hf.config import build_config
from ml.integrations.adapters.hf.support import apply_auto_map


def save_config_bundle(
    config: object,
    *,
    save_directory: str | os.PathLike,
) -> None:
    save_dir = os.path.abspath(str(save_directory))
    os.makedirs(save_dir, exist_ok=True)
    adapter_config = apply_auto_map(build_config(config))
    adapter_config.save_pretrained(save_dir)
    GenerationConfig.from_model_config(adapter_config).save_pretrained(save_dir)


__all__ = ["save_config_bundle"]
