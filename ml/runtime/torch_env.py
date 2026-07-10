"""Shared PyTorch runtime helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from ml.runtime.stack import disable_optional_ml_backends

disable_optional_ml_backends()

PINNED_CUDA_ALLOC_CONF = "expandable_segments:True"


def require_cuda(device: str = "cuda:0") -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but not available. Install a CUDA-enabled PyTorch build and run on a GPU machine."
        )

    dev = str(device).strip()
    if dev == "cuda":
        dev = "cuda:0"
    if not dev.startswith("cuda"):
        raise ValueError(f"CUDA device is required in this project (got: {device!r}).")

    if dev.startswith("cuda:"):
        idx_s = dev.split(":", 1)[1]
        if idx_s.isdigit():
            idx = int(idx_s)
            n = torch.cuda.device_count()
            if idx < 0 or idx >= n:
                raise ValueError(
                    f"Invalid CUDA device index {idx}; available device_count={n}."
                )
            torch.cuda.set_device(idx)
    return dev


def setup_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


def setup_torch_backends() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", PINNED_CUDA_ALLOC_CONF)
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


__all__ = [
    "PINNED_CUDA_ALLOC_CONF",
    "require_cuda",
    "setup_seed",
    "setup_torch_backends",
    "torch",
]
