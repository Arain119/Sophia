from __future__ import annotations

from dataclasses import dataclass

import torch

from ml.errors import SophiaUsageError
from ml.runtime.stack import ensure_standard_stack
from ml.runtime.torch_env import require_cuda, setup_seed, setup_torch_backends
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.value_semantics import coerce_int, normalized_str
from ml.training.pretrain.release_config import (
    RELEASE_PRETRAIN_DTYPE,
    release_pretrain_runtime_metadata,
)


@dataclass(frozen=True)
class ResolvedPretrainRuntime:
    device: torch.device
    base_dtype: torch.dtype
    runtime_metadata: dict[str, object]


def _assert_sdpa_non_math_backend(device: torch.device) -> None:
    if device.type != "cuda":
        return
    probe: torch.Tensor | None = None
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        probe = torch.empty(
            (1, 16, 4096, 128),
            device=device,
            dtype=torch.bfloat16,
        )
        with sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        ):
            torch.nn.functional.scaled_dot_product_attention(
                probe,
                probe,
                probe,
                is_causal=True,
            )
        torch.cuda.synchronize(device)
    except Exception as exc:
        raise SophiaUsageError(
            "[ERR] scaled_dot_product_attention must support a flash or "
            "efficient backend for the 4K MLA training path."
        ) from exc
    finally:
        if probe is not None:
            del probe
        torch.cuda.empty_cache()


def setup_train_runtime_and_device(
    args: PretrainRunConfig, *, run_kind: str
) -> ResolvedPretrainRuntime:
    """
    Shared training runtime setup.

    Enforces Sophia's pinned training stack: CUDA + BF16 base dtype.
    """
    run_kind = normalized_str(run_kind, default="run")
    ensure_standard_stack(mode="train", require_tensorboard=True)

    base_dtype = RELEASE_PRETRAIN_DTYPE
    if base_dtype != torch.bfloat16:
        raise SophiaUsageError(
            f"[ERR] Unsupported base_dtype={base_dtype}; Sophia {run_kind} is pinned to BF16 base dtype."
        )

    device_str = require_cuda(normalized_str(args.device, default="cuda:0"))
    device = torch.device(device_str)
    setup_torch_backends()
    _assert_sdpa_non_math_backend(device)
    setup_seed(coerce_int(args.seed, default=42))

    return ResolvedPretrainRuntime(
        device=device,
        base_dtype=base_dtype,
        runtime_metadata=release_pretrain_runtime_metadata(),
    )


__all__ = [
    "ResolvedPretrainRuntime",
    "_assert_sdpa_non_math_backend",
    "setup_train_runtime_and_device",
]
