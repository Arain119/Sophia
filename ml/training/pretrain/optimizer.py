"""
Torch Muon hybrid optimizer for Sophia.

Policy:
- Apply Muon to transformer matrix weights (2D, excluding embeddings / lm_head).
- Apply fused AdamW to all remaining trainable parameters.

Master-weight mixed precision:
- The bf16 model parameters are used for the forward/backward.
- The optimizer keeps an fp32 master copy of every parameter; updates and optimizer
  state live in fp32, and the bf16 model weights are refreshed from the masters after
  each step. Without this, an update of magnitude < half a bf16 ULP (the Muon/AdamW
  regime at the pinned LR) rounds away on write-back and the weight never moves. This
  mirrors the fp32 EMA shadow (see ``ModelEMA``); the two now share one numerics story.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer

from ml.training.pretrain.muon import Muon
from ml.training.pretrain.optimizer_contracts import (
    MUON_EPS,
    MUON_NS_COEFFICIENTS,
    MUON_NS_STEPS,
    TorchMuonHybridState,
)
from ml.training.pretrain.optimizer_diagnostics import adamw_group_diagnostics
from ml.training.pretrain.optimizer_grouping import (
    _group_params_for_hybrid_layerwise,
    _split_params_for_hybrid,
    build_hybrid_param_groups,
)


def _checkpoint_state_value(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _checkpoint_state_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_checkpoint_state_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_state_value(item) for item in value)
    return value


class TorchMuonFusedAdamW(Optimizer):
    """
    Hybrid optimizer: Muon + fused `torch.optim.AdamW`, over fp32 master weights.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
        layerwise_lr_decay: float = 1.0,
        embedding_lr_scale: float = 1.0,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_adjust_lr_fn: str | None = None,
        muon_ns_coefficients: tuple[float, float, float] = MUON_NS_COEFFICIENTS,
        muon_eps: float = MUON_EPS,
        muon_ns_steps: int = MUON_NS_STEPS,
        muon_target_rms: float | None = None,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("TorchMuonFusedAdamW requires a torch.nn.Module")

        muon_param_groups, adamw_groups = build_hybrid_param_groups(
            model,
            lr=float(lr),
            weight_decay=float(weight_decay),
            layerwise_lr_decay=float(layerwise_lr_decay),
            embedding_lr_scale=float(embedding_lr_scale),
        )

        # Map every model parameter to a single fp32 master; the inner optimizers
        # operate on masters, while the model keeps its bf16 view for compute.
        self._master_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        master_of: dict[torch.Tensor, torch.Tensor] = {}

        def _master_payload(
            group, sink: list[torch.Tensor]
        ) -> dict[str, object]:
            model_params = [p for p in group.params if torch.is_tensor(p)]
            sink.extend(model_params)
            masters: list[torch.Tensor] = []
            for p in model_params:
                master = master_of.get(p)
                if master is None:
                    master = p.detach().to(dtype=torch.float32).clone()
                    master.requires_grad_(True)
                    master_of[p] = master
                    self._master_pairs.append((p, master))
                masters.append(master)
            payload: dict[str, object] = {
                "name": str(group.name),
                "params": masters,
            }
            if group.lr is not None:
                payload["lr"] = float(group.lr)
            if group.weight_decay is not None:
                payload["weight_decay"] = float(group.weight_decay)
            return payload

        self._model_muon_params: list[torch.Tensor] = []
        self._muon_opt = Muon(
            [_master_payload(group, self._model_muon_params) for group in muon_param_groups],
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=float(muon_momentum),
            nesterov=bool(muon_nesterov),
            ns_coefficients=tuple(float(x) for x in muon_ns_coefficients),
            eps=float(muon_eps),
            ns_steps=int(muon_ns_steps),
            adjust_lr_fn=(
                None
                if muon_adjust_lr_fn is None
                else str(muon_adjust_lr_fn)
            ),
            target_rms=(
                None
                if muon_target_rms is None
                else float(muon_target_rms)
            ),
        )

        self._model_adamw_params: list[torch.Tensor] = []
        self._adamw_opt: torch.optim.Optimizer | None = None
        if adamw_groups:
            self._adamw_opt = torch.optim.AdamW(
                [_master_payload(group, self._model_adamw_params) for group in adamw_groups],
                lr=float(lr),
                betas=(float(betas[0]), float(betas[1])),
                eps=float(eps),
                fused=True,
            )

        defaults = {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "betas": (float(betas[0]), float(betas[1])),
            "eps": float(eps),
        }
        super().__init__(list(self._muon_opt.param_groups), defaults)

        self.param_groups = list(self._muon_opt.param_groups)
        if self._adamw_opt is not None:
            self.param_groups.extend(list(self._adamw_opt.param_groups))

    def _sync_grads_to_masters(self) -> None:
        for model_param, master in self._master_pairs:
            grad = model_param.grad
            if grad is None:
                master.grad = None
                continue
            existing = master.grad
            if existing is None or existing.shape != grad.shape:
                master.grad = grad.detach().to(dtype=torch.float32)
            else:
                existing.copy_(grad)

    def _sync_masters_to_model(self) -> None:
        for model_param, master in self._master_pairs:
            model_param.data.copy_(master.data)

    def _muon_parameters(self) -> list[torch.Tensor]:
        return list(self._model_muon_params)

    def _adamw_parameters(self) -> list[torch.Tensor]:
        return list(self._model_adamw_params)

    @torch.no_grad()
    def clip_gradients(
        self,
        *,
        max_grad_norm: float,
        agc_clip: float,
        agc_eps: float,
        agc_exclude_bias_and_norm: bool,
    ) -> torch.Tensor | None:
        from ml.training.pretrain.engine.grad_clip import (
            adaptive_clip_grad_,
            combine_grad_norms_,
        )

        muon_grad_norm = None
        muon_params = self._muon_parameters()
        if muon_params and float(max_grad_norm) > 0:
            muon_grad_norm = torch.nn.utils.clip_grad_norm_(
                muon_params,
                float(max_grad_norm),
            )

        adamw_grad_norm = None
        adamw_params = self._adamw_parameters()
        if adamw_params and float(agc_clip) > 0:
            adamw_grad_norm = adaptive_clip_grad_(
                adamw_params,
                clip=float(agc_clip),
                eps=float(agc_eps),
                exclude_bias_and_norm=bool(agc_exclude_bias_and_norm),
            )
        return combine_grad_norms_((muon_grad_norm, adamw_grad_norm))

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._sync_grads_to_masters()
        self._muon_opt.step()
        if self._adamw_opt is not None:
            self._adamw_opt.step()
        self._sync_masters_to_model()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        # Gradients are produced on the bf16 model parameters by backward; the master
        # grads are transient buffers refreshed each step. Clear both.
        for model_param, master in self._master_pairs:
            for tensor in (model_param, master):
                grad = tensor.grad
                if grad is None:
                    continue
                if bool(set_to_none):
                    tensor.grad = None
                else:
                    grad.detach_()
                    grad.zero_()

    def state_dict(self) -> dict[str, object]:  # type: ignore[override]
        return TorchMuonHybridState(
            muon=dict(_checkpoint_state_value(self._muon_opt.state_dict())),
            adamw=(
                None
                if self._adamw_opt is None
                else dict(_checkpoint_state_value(self._adamw_opt.state_dict()))
            ),
            masters=[
                master.detach().to(device="cpu", dtype=torch.float32, copy=True)
                for _, master in self._master_pairs
            ],
        ).to_payload()

    def load_state_dict(self, state_dict: dict[str, object]) -> None:  # type: ignore[override]
        if not isinstance(state_dict, dict):
            raise RuntimeError("TorchMuonFusedAdamW expected a dict state_dict.")
        payload = TorchMuonHybridState.from_payload(state_dict)
        self._muon_opt.load_state_dict(payload.muon)
        if self._adamw_opt is not None:
            if payload.adamw is None:
                raise RuntimeError("Missing AdamW state in torch_muon_hybrid checkpoint.")
            self._adamw_opt.load_state_dict(payload.adamw)

        masters = payload.masters
        if len(masters) != len(self._master_pairs):
            raise RuntimeError(
                "Master weights in torch_muon_hybrid checkpoint do not match optimizer parameters."
            )
        for (model_param, master), saved in zip(
            self._master_pairs, masters, strict=True
        ):
            master.data.copy_(saved.to(device=master.device, dtype=torch.float32))
            model_param.data.copy_(master.data)

        self.param_groups = list(self._muon_opt.param_groups)
        if self._adamw_opt is not None:
            self.param_groups.extend(list(self._adamw_opt.param_groups))

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        payload = {
            "optimizer/muon/group_count": float(len(self._muon_opt.param_groups)),
            "optimizer/muon/param_count": float(len(self._muon_parameters())),
        }
        if self._adamw_opt is None:
            payload["optimizer/adamw/group_count"] = 0.0
            payload["optimizer/adamw/param_count"] = 0.0
            return payload

        payload["optimizer/adamw/group_count"] = float(len(self._adamw_opt.param_groups))
        payload["optimizer/adamw/param_count"] = float(len(self._adamw_parameters()))
        payload.update(
            adamw_group_diagnostics(
                param_groups=self._adamw_opt.param_groups,
                state=self._adamw_opt.state,
            )
        )
        return payload


def create_torch_muon_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    layerwise_lr_decay: float = 1.0,
    embedding_lr_scale: float = 1.0,
    muon_ns_steps: int | None = None,
    muon_target_rms: float | None = None,
) -> Optimizer:
    ns_steps: int | None = None
    if muon_ns_steps is not None:
        try:
            ns_steps = int(muon_ns_steps)
        except (TypeError, ValueError):
            ns_steps = None
    if ns_steps is not None and ns_steps <= 0:
        ns_steps = None
    return TorchMuonFusedAdamW(
        model,
        lr=float(lr),
        weight_decay=float(weight_decay),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(eps),
        layerwise_lr_decay=float(layerwise_lr_decay),
        embedding_lr_scale=float(embedding_lr_scale),
        muon_ns_steps=int(ns_steps) if ns_steps is not None else MUON_NS_STEPS,
        muon_target_rms=(
            None
            if muon_target_rms is None
            else float(muon_target_rms)
        ),
    )


__all__ = [
    "TorchMuonFusedAdamW",
    "_group_params_for_hybrid_layerwise",
    "_split_params_for_hybrid",
    "create_torch_muon_optimizer",
]
