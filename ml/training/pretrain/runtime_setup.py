from __future__ import annotations

from dataclasses import dataclass

import torch

from ml.errors import SophiaUsageError
from ml.runtime.stack import ensure_standard_stack
from ml.runtime.torch_env import require_cuda, setup_seed, setup_torch_backends
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.value_semantics import coerce_int, normalized_str
from ml.training.pretrain.release_config import (
    RELEASE_PRETRAIN_PRECISION_POLICY,
)


@dataclass(frozen=True)
class ResolvedPretrainRuntime:
    device: torch.device
    base_dtype: torch.dtype
    runtime_metadata: dict[str, object]


def setup_train_runtime_and_device(
    args: PretrainRunConfig, *, run_kind: str
) -> ResolvedPretrainRuntime:
    """
    Shared training runtime setup.

    Enforces Sophia's pinned training stack: CUDA + BF16 base dtype.
    """
    run_kind = normalized_str(run_kind, default="run")
    ensure_standard_stack(mode="train", require_tensorboard=True)

    base_dtype = RELEASE_PRETRAIN_PRECISION_POLICY.resolve_dtype()
    if base_dtype != torch.bfloat16:
        raise SophiaUsageError(
            f"[ERR] Unsupported base_dtype={base_dtype}; Sophia {run_kind} is pinned to BF16 base dtype."
        )

    device_str = require_cuda(normalized_str(args.device, default="cuda:0"))
    device = torch.device(device_str)
    setup_torch_backends()
    setup_seed(coerce_int(args.seed, default=42))

    return ResolvedPretrainRuntime(
        device=device,
        base_dtype=base_dtype,
        runtime_metadata=RELEASE_PRETRAIN_PRECISION_POLICY.runtime_metadata(),
    )


__all__ = [
    "ResolvedPretrainRuntime",
    "setup_train_runtime_and_device",
]
