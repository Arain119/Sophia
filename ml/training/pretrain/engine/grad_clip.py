"""
Gradient scaling + clipping utilities shared by training loop and machine benchmarking.

Policy:
- The pinned stack uses optimizer-aware hybrid clipping for stability.
- Machine benchmarking must use the same gradient post-processing as real training.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

# Canonical source is grad_clip_semantics.py (dependency-free) so lightweight
# config modules like run_config.py can import the pinned values without
# pulling in this package's __init__ and its full engine execution chain.
# Re-exported here for existing consumers of this module.
from ml.training.pretrain.grad_clip_semantics import (
    PINNED_AGC_CLIP,
    PINNED_AGC_EPS,
    PINNED_AGC_EXCLUDE_BIAS_AND_NORM,
    PINNED_GRAD_CLIP_MODE,
)


def _foreach_mul_(
    tensors: list[torch.Tensor],
    scales: list[torch.Tensor] | torch.Tensor,
) -> None:
    fn = getattr(torch, "_foreach_mul_", None)
    if not callable(fn):
        raise RuntimeError("torch._foreach_mul_ is required for gradient scaling.")
    try:
        fn(tensors, scales)
    except RuntimeError as exc:
        raise RuntimeError("torch._foreach_mul_ failed during gradient scaling.") from exc


def _l2_norms(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    foreach_norm = getattr(torch, "_foreach_norm", None)
    if not callable(foreach_norm):
        raise RuntimeError("torch._foreach_norm is required for gradient norm computation.")
    try:
        return list(foreach_norm(tensors, ord=2, dtype=torch.float32))
    except RuntimeError as exc:
        raise RuntimeError("torch._foreach_norm failed during gradient norm computation.") from exc


@torch.no_grad()
def adaptive_clip_grad_(
    params,
    *,
    clip: float,
    eps: float = 1e-3,
    exclude_bias_and_norm: bool = True,
) -> torch.Tensor | None:
    """
    Adaptive Gradient Clipping (AGC), per-parameter-tensor.

    Clips each gradient so ``||g|| <= clip * ||p||``. Norms are computed in
    float32 for stability. Returns the pre-clip global gradient norm.
    """

    clip = float(clip)
    if not (clip > 0):
        return None
    eps = float(eps)
    if eps <= 0:
        eps = 1e-3

    grads: list[torch.Tensor] = []
    ps: list[torch.Tensor] = []
    for p in params:
        if not torch.is_tensor(p) or p.grad is None:
            continue
        if bool(exclude_bias_and_norm) and int(p.ndim) <= 1:
            continue
        grads.append(p.grad)
        ps.append(p.detach())

    if not grads:
        return None

    p_norms = _l2_norms(ps)
    g_norms = _l2_norms([g.detach() for g in grads])

    total_sq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    scales: list[torch.Tensor] = []
    for p_norm, g_norm, g in zip(p_norms, g_norms, grads, strict=True):
        total_sq.add_(g_norm * g_norm)
        max_norm = (p_norm * clip).clamp_min(eps)
        scale = (max_norm / (g_norm + eps)).clamp(max=1.0)
        scales.append(scale.to(dtype=g.dtype))

    _foreach_mul_(grads, scales)
    return torch.sqrt(total_sq)


def combine_grad_norms_(
    grad_norms: Iterable[torch.Tensor | None],
) -> torch.Tensor | None:
    total_sq: torch.Tensor | None = None
    for grad_norm in grad_norms:
        if grad_norm is None:
            continue
        grad_norm_f = grad_norm.detach().to(dtype=torch.float32)
        term = grad_norm_f * grad_norm_f
        if total_sq is None:
            total_sq = term
        else:
            total_sq = total_sq + term
    if total_sq is None:
        return None
    return torch.sqrt(total_sq)


@torch.no_grad()
def grad_global_norm_(params) -> torch.Tensor | None:
    grads = [p.grad.detach() for p in params if torch.is_tensor(p) and p.grad is not None]
    if not grads:
        return None
    norms = _l2_norms(grads)
    total_sq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for norm in norms:
        total_sq.add_(norm * norm)
    return torch.sqrt(total_sq)


@torch.no_grad()
def scale_grads_by_token_count_(
    *, model: torch.nn.Module, supervised_tokens: torch.Tensor
) -> None:
    denom = supervised_tokens.clamp_min(1).to(dtype=torch.float32)
    scale = 1.0 / denom
    grads = [p.grad for p in model.parameters() if torch.is_tensor(p.grad)]
    if grads:
        _foreach_mul_(grads, scale)


@torch.no_grad()
def clip_gradients_(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_mode: str,
    max_grad_norm: float,
    agc_clip: float,
    agc_eps: float,
    agc_exclude_bias_and_norm: bool,
) -> torch.Tensor | None:
    grad_norm = None
    clip_mode = str(grad_clip_mode or "norm").strip().lower()
    if clip_mode in ("hybrid_auto", "optimizer"):
        if optimizer is None or not hasattr(optimizer, "clip_gradients"):
            raise ValueError(
                "grad_clip_mode='hybrid_auto' requires an optimizer with clip_gradients()."
            )
        grad_norm = optimizer.clip_gradients(
            max_grad_norm=float(max_grad_norm),
            agc_clip=float(agc_clip),
            agc_eps=float(agc_eps),
            agc_exclude_bias_and_norm=bool(agc_exclude_bias_and_norm),
        )
    elif clip_mode in ("norm", "clip_norm", "global_norm", "gnorm"):
        if float(max_grad_norm) > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(max_grad_norm)
            )
    elif clip_mode in (
        "agc",
        "adaptive",
        "adaptive_grad_clip",
        "adaptive_gradient_clipping",
    ):
        grad_norm = adaptive_clip_grad_(
            model.parameters(),
            clip=float(agc_clip),
            eps=float(agc_eps),
            exclude_bias_and_norm=bool(agc_exclude_bias_and_norm),
        )
    elif clip_mode in ("none", "off", "0", ""):
        pass
    else:
        raise ValueError(f"Unsupported grad_clip_mode: {clip_mode!r}")
    return grad_norm


__all__ = [
    "PINNED_AGC_CLIP",
    "PINNED_AGC_EPS",
    "PINNED_AGC_EXCLUDE_BIAS_AND_NORM",
    "PINNED_GRAD_CLIP_MODE",
    "adaptive_clip_grad_",
    "clip_gradients_",
    "combine_grad_norms_",
    "grad_global_norm_",
    "scale_grads_by_token_count_",
]
