from __future__ import annotations

import math

import torch
from torch import Tensor

def get_scalar_value(x: float | Tensor) -> float:
    if torch.is_tensor(x):
        return float(x.item())
    return float(x)


def zeropower_via_newtonschulz5(
    g: Tensor,
    steps: int,
    eps: float,
    coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    output_dtype: torch.dtype | None = None,
) -> Tensor:
    if not torch.is_tensor(g):
        raise TypeError("zeropower_via_newtonschulz5 expects a Tensor")
    if g.ndim != 2:
        raise ValueError(f"zeropower expects a 2D tensor (got shape={tuple(g.shape)!r})")

    if len(coeffs) != 3:
        raise ValueError(
            "Newton-Schulz coefficients must contain exactly three values "
            f"(got {len(coeffs)})"
        )
    if int(steps) < 0:
        raise ValueError(f"Newton-Schulz steps must be non-negative (got {steps})")
    if float(eps) <= 0.0:
        raise ValueError(f"Newton-Schulz eps must be positive (got {eps})")

    a, b, c = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    if not all(math.isfinite(value) for value in (a, b, c)):
        raise ValueError(f"Newton-Schulz coefficients must be finite (got {coeffs!r})")
    eps_f = float(eps)

    x = g
    do_transpose = bool(g.size(0) > g.size(1))
    if do_transpose:
        x = x.T
    compute_dtype = torch.bfloat16 if x.is_cuda else torch.float32
    if x.dtype != compute_dtype:
        x = x.to(dtype=compute_dtype)
    else:
        x = x.clone()

    x.div_(x.norm() + eps_f)
    for _ in range(int(steps)):
        # The Muon polynomial is X <- aX + (bA + cA^2)X, A = XX^T.
        # Keeping the identity out of this expression is essential: b scales
        # the first Gram-matrix term, rather than an identity skip connection.
        gram = x @ x.T
        gram_squared = gram @ gram
        gram.mul_(b)
        gram_squared.mul_(c)
        gram.add_(gram_squared)
        del gram_squared
        next_x = gram @ x
        x.mul_(a)
        next_x.add_(x)
        x = next_x

    if do_transpose:
        x = x.T
    target_dtype = g.dtype if output_dtype is None else output_dtype
    return x.to(dtype=target_dtype) if x.dtype != target_dtype else x


def adjust_lr_for_muon(
    lr: float,
    param_shape: torch.Size,
) -> float:
    """Use Moonlight's RMS-matching scale for matrix updates."""
    lr_f = float(lr)
    if len(param_shape) != 2:
        return lr_f

    m = float(param_shape[0])
    n = float(param_shape[1])
    if m <= 0.0 or n <= 0.0:
        return lr_f

    return lr_f * 0.2 * math.sqrt(max(m, n))


__all__ = [
    "adjust_lr_for_muon",
    "get_scalar_value",
    "zeropower_via_newtonschulz5",
]
