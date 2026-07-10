from __future__ import annotations

import torch
from torch import Tensor

ORTHOGONALIZATION_EPS = 1e-16

_EYE_CACHE: dict[tuple[torch.device, torch.dtype, int], Tensor] = {}


def get_scalar_value(x: float | Tensor) -> float:
    if torch.is_tensor(x):
        return float(x.item())
    return float(x)


def _cached_eye(*, n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (device, dtype, int(n))
    eye = _EYE_CACHE.get(key)
    if eye is None:
        eye = torch.eye(int(n), device=device, dtype=dtype)
        _EYE_CACHE[key] = eye
    return eye


def zeropower_via_newtonschulz5(
    g: Tensor,
    steps: int,
    eps: float,
    coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
) -> Tensor:
    if not torch.is_tensor(g):
        raise TypeError("zeropower_via_newtonschulz5 expects a Tensor")
    if g.ndim != 2:
        raise ValueError(f"zeropower expects a 2D tensor (got shape={tuple(g.shape)!r})")

    a, b, c = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    eps_f = float(eps)

    x = g
    do_transpose = bool(g.size(0) > g.size(1))
    if do_transpose:
        x = x.T
    if x.dtype != torch.float32:
        x = x.float()

    x = x / (x.norm() + eps_f)
    eye = _cached_eye(n=int(x.size(0)), device=x.device, dtype=x.dtype)
    for _ in range(int(steps)):
        x1 = x @ x.T
        b_mat = torch.addmm(eye, x1, x1, beta=b, alpha=c)
        x = (a * x) + (b_mat @ x)

    if do_transpose:
        x = x.T
    return x.to(dtype=g.dtype)


def zeropower_via_newtonschulz5_batched(
    g: Tensor,
    steps: int,
    eps: float,
    coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
) -> Tensor:
    if not torch.is_tensor(g):
        raise TypeError("zeropower_via_newtonschulz5_batched expects a Tensor")
    if g.ndim != 3:
        raise ValueError(
            "zeropower_via_newtonschulz5_batched expects a 3D tensor "
            f"(got shape={tuple(g.shape)!r})"
        )

    a, b, c = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    eps_f = float(eps)

    x = g
    do_transpose = bool(g.size(-2) > g.size(-1))
    if do_transpose:
        x = x.transpose(-2, -1)
    if x.dtype != torch.float32:
        x = x.float()

    denom = x.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(eps_f)
    x = x / denom

    eye = _cached_eye(n=int(x.size(-2)), device=x.device, dtype=x.dtype).unsqueeze(0)
    for _ in range(int(steps)):
        x1 = torch.matmul(x, x.transpose(-2, -1))
        b_mat = (x1 @ x1) * c + eye * b
        x = (a * x) + torch.matmul(b_mat, x)

    if do_transpose:
        x = x.transpose(-2, -1)
    return x.to(dtype=g.dtype)


def adjust_lr_for_muon(
    lr: float,
    param_shape: torch.Size,
    adjust_lr_fn: str | None = "match_rms",
) -> float:
    lr_f = float(lr)
    if adjust_lr_fn is None:
        return lr_f

    fn = str(adjust_lr_fn)
    if fn not in ("match_rms", "match_rms_adamw"):
        raise ValueError(
            f"Invalid adjust_lr_fn: {adjust_lr_fn!r} (expected: match_rms/match_rms_adamw/None)"
        )
    if len(param_shape) != 2:
        return lr_f

    m = float(param_shape[0])
    n = float(param_shape[1])
    if m <= 0.0 or n <= 0.0:
        return lr_f

    scale = float(max(1.0, m / n, n / m))
    if fn == "match_rms":
        return lr_f * (scale**0.5)
    return lr_f * scale


__all__ = [
    "ORTHOGONALIZATION_EPS",
    "adjust_lr_for_muon",
    "get_scalar_value",
    "zeropower_via_newtonschulz5",
    "zeropower_via_newtonschulz5_batched",
]
