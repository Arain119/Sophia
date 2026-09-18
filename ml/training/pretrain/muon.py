"""Muon optimizer for Sophia transformer matrix weights."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ml.training.pretrain.muon_math import (
    adjust_lr_for_muon,
    get_scalar_value,
    zeropower_via_newtonschulz5,
)
from ml.training.pretrain.optimizer_contracts import MUON_NS_COEFFICIENTS


class Muon(Optimizer):
    """Muon optimizer for complete transformer weight matrices."""

    def __init__(
        self,
        params,
        *,
        lr: float,
        weight_decay: float,
        eps: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        preserve_gradients: bool = True,
        update_dtype: torch.dtype | None = None,
        momentum_dtype: torch.dtype | None = None,
        ns_steps: int = 5,
        ns_coefficients: tuple[float, float, float] = MUON_NS_COEFFICIENTS,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")
        if ns_steps < 0:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")

        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "eps": float(eps),
            "nesterov": bool(nesterov),
            "preserve_gradients": bool(preserve_gradients),
            "update_dtype": update_dtype,
            "momentum_dtype": momentum_dtype,
            "ns_steps": int(ns_steps),
            "ns_coefficients": tuple(float(x) for x in ns_coefficients),
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

    def load_state_dict(self, state_dict) -> None:  # type: ignore[override]
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            momentum_dtype = group["momentum_dtype"]
            for p in group["params"]:
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if not torch.is_tensor(buf):
                    continue
                target_dtype = p.dtype if momentum_dtype is None else momentum_dtype
                if buf.dtype != target_dtype:
                    state["momentum_buffer"] = buf.to(dtype=target_dtype)

    @torch.no_grad()
    def initialize_state(self) -> None:
        """Materialize momentum buffers before a CUDA Graph is captured."""
        for group in self.param_groups:
            momentum_dtype = group["momentum_dtype"]
            target_dtype = momentum_dtype
            for parameter in group["params"]:
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        parameter,
                        dtype=(parameter.dtype if target_dtype is None else target_dtype),
                        memory_format=torch.preserve_format,
                    )

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
            preserve_gradients = bool(group["preserve_gradients"])
            update_dtype = group["update_dtype"]
            momentum_dtype = group["momentum_dtype"]
            ns_steps = int(group["ns_steps"])
            ns_coefficients = tuple(group["ns_coefficients"])

            grouped: dict[
                tuple[torch.device, torch.dtype, int, int],
                tuple[list[Tensor], list[Tensor]],
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
                key = (
                    p.device,
                    p.dtype,
                    int(p.shape[0]),
                    int(p.shape[1]),
                )
                grouped[key][0].append(p)
                grouped[key][1].append(grad)

            for (_device, _dtype, m, n), (params, grads) in grouped.items():
                if not params:
                    continue

                buffers: list[Tensor] = []
                for p in params:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(
                            p,
                            dtype=(p.dtype if momentum_dtype is None else momentum_dtype),
                            memory_format=torch.preserve_format,
                        )
                        state["momentum_buffer"] = buf
                    buffers.append(buf)

                torch._foreach_mul_(buffers, float(momentum))
                torch._foreach_add_(buffers, grads)
                grads_eff = grads
                if not nesterov:
                    grads_eff = buffers
                else:
                    if preserve_gradients:
                        grads_eff = torch._foreach_add(
                            grads,
                            buffers,
                            alpha=float(momentum),
                        )
                    else:
                        torch._foreach_add_(grads, buffers, alpha=float(momentum))

                self._apply_decoupled_weight_decay(
                    params,
                    lr=float(lr),
                    weight_decay=float(weight_decay),
                )
                lr_adj = adjust_lr_for_muon(float(lr), torch.Size((m, n)))
                for param, grad in zip(params, grads_eff, strict=True):
                    update = zeropower_via_newtonschulz5(
                        grad,
                        steps=int(ns_steps),
                        eps=float(eps),
                        coeffs=tuple(float(x) for x in ns_coefficients),
                        output_dtype=update_dtype,
                    )
                    update.mul_(float(lr_adj))
                    param.add_(update, alpha=-1.0)

        return loss


__all__ = [
    "Muon",
    "adjust_lr_for_muon",
    "zeropower_via_newtonschulz5",
]
