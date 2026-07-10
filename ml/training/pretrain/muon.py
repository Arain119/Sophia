"""Muon optimizer for Sophia transformer matrix weights."""

from __future__ import annotations

import os
from collections import defaultdict

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ml.training.pretrain.muon_math import (
    ORTHOGONALIZATION_EPS,
    adjust_lr_for_muon,
    get_scalar_value,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_batched,
)

MUON_ORTHO_BATCH_MAX_ENV = "SOPHIA_MUON_ORTHO_BATCH_MAX"
MUON_ORTHO_BATCH_MAX_DEFAULT = 1


def _resolve_ortho_batch_max(value: object | None = None) -> int:
    raw = os.environ.get(MUON_ORTHO_BATCH_MAX_ENV, "") if value is None else value
    try:
        resolved = int(str(raw).strip())
    except (TypeError, ValueError):
        resolved = int(MUON_ORTHO_BATCH_MAX_DEFAULT)
    if resolved < 0:
        resolved = int(MUON_ORTHO_BATCH_MAX_DEFAULT)
    return int(resolved)


def _iter_chunks(
    params: list[Tensor],
    grads_eff: list[Tensor],
    *,
    max_batch: int,
):
    if len(params) != len(grads_eff):
        raise RuntimeError(
            "Muon internal error: parameter/update groups have mismatched lengths "
            f"({len(params)} != {len(grads_eff)})"
        )
    if int(max_batch) <= 0:
        yield params, grads_eff
        return
    for start in range(0, len(params), int(max_batch)):
        end = start + int(max_batch)
        yield params[start:end], grads_eff[start:end]


class Muon(Optimizer):
    """Muon optimizer for 2D transformer weights."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        eps: float = 1e-7,
        nesterov: bool = True,
        backend: str = "auto",
        ns_steps: int = 5,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        adjust_lr_fn: str | None = "match_rms",
        target_rms: float | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")
        if backend not in ("auto", "foreach"):
            raise ValueError(f"Invalid backend: {backend!r} (expected: auto/foreach)")
        if ns_steps < 0:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")

        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "eps": float(eps),
            "nesterov": bool(nesterov),
            "backend": str(backend),
            "ns_steps": int(ns_steps),
            "ns_coefficients": tuple(float(x) for x in ns_coefficients),
            "adjust_lr_fn": adjust_lr_fn,
            "target_rms": None if target_rms is None else float(target_rms),
        }
        super().__init__(params, defaults)

    @staticmethod
    def _apply_decoupled_weight_decay(
        params: list[Tensor],
        *,
        lr: float,
        weight_decay: float,
    ) -> None:
        if not params or float(weight_decay) == 0.0:
            return
        decay = 1.0 - (float(lr) * float(weight_decay))
        torch._foreach_mul_(params, float(decay))

    @staticmethod
    def _orthogonalized_updates(
        grads_eff: list[Tensor],
        *,
        ns_steps: int,
        eps: float,
        ns_coefficients: tuple[float, ...],
        target_rms: float | None,
    ) -> list[Tensor]:
        if len(grads_eff) == 1:
            updates = [
                zeropower_via_newtonschulz5(
                    grads_eff[0],
                    steps=int(ns_steps),
                    eps=float(eps),
                    coeffs=tuple(float(x) for x in ns_coefficients),
                )
            ]
            if target_rms is not None and float(target_rms) > 0.0:
                rms = updates[0].float().square().mean().sqrt().clamp_min(1e-12)
                updates[0] = updates[0] * (float(target_rms) / rms).to(dtype=updates[0].dtype)
            return updates

        max_batch = _resolve_ortho_batch_max()
        if max_batch > 0 and len(grads_eff) > int(max_batch):
            updates: list[Tensor] = []
            for start in range(0, len(grads_eff), int(max_batch)):
                updates.extend(
                    Muon._orthogonalized_updates(
                        grads_eff[start : start + int(max_batch)],
                        ns_steps=int(ns_steps),
                        eps=float(eps),
                        ns_coefficients=tuple(float(x) for x in ns_coefficients),
                        target_rms=target_rms,
                    )
                )
            return updates

        updates_batched = zeropower_via_newtonschulz5_batched(
            torch.stack(grads_eff, dim=0),
            steps=int(ns_steps),
            eps=float(eps),
            coeffs=tuple(float(x) for x in ns_coefficients),
        )
        if target_rms is not None and float(target_rms) > 0.0:
            rms = (
                updates_batched.float()
                .square()
                .mean(dim=(-2, -1), keepdim=True)
                .sqrt()
                .clamp_min(1e-12)
            )
            updates_batched = updates_batched * (
                float(target_rms) / rms
            ).to(dtype=updates_batched.dtype)
        return list(updates_batched.unbind(0))

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = get_scalar_value(group["lr"])
            momentum = get_scalar_value(group["momentum"])
            weight_decay = get_scalar_value(group["weight_decay"])
            eps = get_scalar_value(group["eps"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            ns_coefficients = tuple(group["ns_coefficients"])
            adjust_lr_fn = group.get("adjust_lr_fn", "match_rms")
            target_rms = group.get("target_rms")

            grouped: dict[
                tuple[torch.device, torch.dtype, int, int], tuple[list[Tensor], list[Tensor]]
            ] = defaultdict(lambda: ([], []))

            for p in group["params"]:
                if not torch.is_tensor(p) or p.grad is None:
                    continue
                if p.ndim != 2:
                    raise RuntimeError(
                        "Muon requires 2D parameters (matrix weights). "
                        f"Got shape={tuple(p.shape)!r}"
                    )
                grad = p.grad
                if not torch.is_tensor(grad):
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients.")
                key = (p.device, p.dtype, int(p.shape[0]), int(p.shape[1]))
                grouped[key][0].append(p)
                grouped[key][1].append(grad)

            for (_device, _dtype, _m, _n), (params, grads) in grouped.items():
                if not params:
                    continue

                buffers: list[Tensor] = []
                for p in params:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["momentum_buffer"] = buf
                    buffers.append(buf)

                torch._foreach_mul_(buffers, float(momentum))
                torch._foreach_add_(buffers, grads)
                grads_eff = grads
                if not nesterov:
                    grads_eff = buffers
                else:
                    torch._foreach_add_(grads, buffers, alpha=float(momentum))

                lr_adj = adjust_lr_for_muon(float(lr), params[0].shape, adjust_lr_fn)
                max_batch = _resolve_ortho_batch_max()
                for chunk_params, chunk_grads_eff in _iter_chunks(
                    params,
                    grads_eff,
                    max_batch=int(max_batch),
                ):
                    updates = self._orthogonalized_updates(
                        chunk_grads_eff,
                        ns_steps=int(ns_steps),
                        eps=float(eps),
                        ns_coefficients=tuple(float(x) for x in ns_coefficients),
                        target_rms=(
                            None if target_rms is None else float(target_rms)
                        ),
                    )
                    self._apply_decoupled_weight_decay(
                        chunk_params,
                        lr=float(lr_adj),
                        weight_decay=float(weight_decay),
                    )
                    torch._foreach_mul_(updates, float(lr_adj))
                    torch._foreach_add_(chunk_params, updates, alpha=-1.0)

        return loss


__all__ = [
    "MUON_ORTHO_BATCH_MAX_DEFAULT",
    "MUON_ORTHO_BATCH_MAX_ENV",
    "Muon",
    "ORTHOGONALIZATION_EPS",
    "adjust_lr_for_muon",
    "_resolve_ortho_batch_max",
    "zeropower_via_newtonschulz5",
    "zeropower_via_newtonschulz5_batched",
]
