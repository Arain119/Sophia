from __future__ import annotations

from collections.abc import Sequence

import torch


def _l2_norms(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    foreach_norm = getattr(torch, "_foreach_norm", None)
    if not callable(foreach_norm):
        raise RuntimeError("torch._foreach_norm is required for gradient norm computation.")
    try:
        return list(foreach_norm(tensors, ord=2, dtype=torch.float32))
    except RuntimeError as exc:
        raise RuntimeError("torch._foreach_norm failed during gradient norm computation.") from exc


@torch.no_grad()
def global_grad_norm(
    params,
    *,
    gradients: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor | None:
    grads = (
        [gradient.detach() for gradient in gradients]
        if gradients is not None
        else [
            p.grad.detach()
            for p in params
            if torch.is_tensor(p) and p.grad is not None
        ]
    )
    if not grads:
        return None
    norms = _l2_norms(grads)
    total_sq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for norm in norms:
        total_sq.add_(norm * norm)
    return torch.sqrt(total_sq)


@torch.no_grad()
def scale_gradients_by_token_count(
    *,
    model: torch.nn.Module,
    supervised_tokens: torch.Tensor,
    gradients: Sequence[torch.Tensor] | None = None,
) -> None:
    scale = 1.0 / supervised_tokens.clamp_min(1).to(dtype=torch.float32)
    grads = (
        list(gradients)
        if gradients is not None
        else [p.grad for p in model.parameters() if torch.is_tensor(p.grad)]
    )
    if not grads:
        return
    foreach_mul = getattr(torch, "_foreach_mul_", None)
    if not callable(foreach_mul):
        raise RuntimeError("torch._foreach_mul_ is required for gradient scaling.")
    try:
        foreach_mul(grads, scale)
    except RuntimeError as exc:
        raise RuntimeError("torch._foreach_mul_ failed during gradient scaling.") from exc


__all__ = ["global_grad_norm", "scale_gradients_by_token_count"]
