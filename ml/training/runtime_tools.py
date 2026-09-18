"""Shared pretraining runtime helpers."""

from __future__ import annotations

import torch

from ml.training.pretrain.optimizer import create_torch_muon_optimizer

__all__ = [
    "apply_gradient_checkpointing",
    "create_optimizer",
    "move_optimizer_state_to_device",
]


def apply_gradient_checkpointing(model: torch.nn.Module, *, enabled: bool) -> None:
    enabled = bool(enabled)
    if enabled:
        model.gradient_checkpointing_enable()  # type: ignore[attr-defined]
        return
    model.gradient_checkpointing_disable()  # type: ignore[attr-defined]


def create_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    muon_ns_steps: int,
) -> torch.optim.Optimizer:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training optimizer creation. Install a CUDA-enabled PyTorch build."
        )

    try:
        import inspect

        sig = inspect.signature(torch.optim.AdamW)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Unable to inspect torch.optim.AdamW signature: {exc}"
        ) from exc
    if "fused" not in sig.parameters:
        raise RuntimeError(
            "This PyTorch build does not support fused AdamW (missing AdamW(..., fused=...)).\n"
            "Install a modern CUDA-enabled PyTorch build that includes fused optimizers."
        )
    return create_torch_muon_optimizer(
        model,
        lr=float(lr),
        weight_decay=float(weight_decay),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(eps),
        muon_ns_steps=muon_ns_steps,
    )


def _move_single_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    *,
    target: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(target)


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: str,
) -> None:
    target = torch.device(str(device))
    _move_single_optimizer_state_to_device(optimizer, target=target)

    for attr_name in ("_muon_opt", "_adamw_opt"):
        inner = getattr(optimizer, attr_name, None)
        if isinstance(inner, torch.optim.Optimizer):
            _move_single_optimizer_state_to_device(inner, target=target)
