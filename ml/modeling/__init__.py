"""
Sophia model package.

Canonical entrypoints:
- `ModelSpec`
- `SophiaModelConfig`
- `SophiaDecoderConfig`
- `SophiaDecoder`
- `create_model_from_spec`

This namespace is intentionally core-model-only. Hugging Face adapters live under
`ml.integrations.adapters.hf`; text assets live under `ml.modeling.text`.
"""

from __future__ import annotations

from ml.modeling.config import SophiaModelConfig
from ml.modeling.sophia_decoder import SophiaDecoderConfig, SophiaDecoder
from ml.core.spec import ModelSpec


def create_model_from_spec(
    spec: ModelSpec,
    *,
    runtime_max_seq_len: int | None = None,
):
    from ml.runtime.model.transformer import Transformer

    return Transformer(spec.to_model_args(runtime_max_seq_len=runtime_max_seq_len))


__all__ = [
    "ModelSpec",
    "SophiaModelConfig",
    "SophiaDecoderConfig",
    "SophiaDecoder",
    "create_model_from_spec",
]
