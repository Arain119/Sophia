from __future__ import annotations

from ml.runtime.stack import (
    disable_optional_ml_backends,
    ensure_standard_stack,
)
from ml.runtime.torch_env import require_cuda, setup_seed, setup_torch_backends

disable_optional_ml_backends()

import torch  # noqa: E402

from ml.integrations.export.model_dir import (  # noqa: E402
    load_local_export_tokenizer,
    local_export_load_kwargs as local_model_load_kwargs,
)
from ml.modeling import SophiaDecoder  # noqa: E402
from ml.modeling.text.conversation import encode_conversation  # noqa: E402
from ml.runtime.generation.sampler import apply_top_k_top_p  # noqa: E402
from ml.core.spec import ModelSpec  # noqa: E402

PRETRAIN_CHECK_DIRNAME = "pretrain_check"
SOPHIA_DIM = int(ModelSpec.default().dim)

__all__ = [
    "PRETRAIN_CHECK_DIRNAME",
    "SOPHIA_DIM",
    "SophiaDecoder",
    "apply_top_k_top_p",
    "encode_conversation",
    "ensure_standard_stack",
    "load_local_export_tokenizer",
    "local_model_load_kwargs",
    "require_cuda",
    "setup_seed",
    "setup_torch_backends",
    "torch",
]
