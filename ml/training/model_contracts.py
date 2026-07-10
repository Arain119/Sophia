from __future__ import annotations

import inspect

import torch


def model_accepts_compute_loss(model: torch.nn.Module) -> bool:
    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return False
    if "compute_loss" in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def require_compute_loss(model: torch.nn.Module, *, context: str) -> None:
    if model_accepts_compute_loss(model):
        return
    raise TypeError(
        f"{context} requires a model forward(..., compute_loss=True) contract. "
        f"Got model type: {type(model)!r}"
    )


__all__ = ["model_accepts_compute_loss", "require_compute_loss"]
