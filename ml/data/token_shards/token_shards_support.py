from __future__ import annotations

import warnings

import numpy as np
import torch


def np_dtype(dtype: str) -> np.dtype:
    if dtype == "int32":
        return np.dtype("<i4")
    raise ValueError(f"Unsupported token shard dtype: {dtype!r}. Expected 'int32'.")


def torch_from_numpy_readonly(arr: np.ndarray) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The given NumPy array is not writable"
        )
        return torch.from_numpy(arr)


__all__ = [
    "np_dtype",
    "torch_from_numpy_readonly",
]
