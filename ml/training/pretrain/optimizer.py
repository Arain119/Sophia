"""
Torch Muon hybrid optimizer for Sophia.

Policy:
- Apply Muon to transformer matrix weights (2D, excluding embeddings / lm_head).
- Apply fused AdamW to all remaining trainable parameters.

Master-weight mixed precision:
- The bf16 model parameters are used for the forward/backward.
- The optimizer keeps an fp32 master copy of every parameter; updates are applied to
  the masters and the bf16 model weights are refreshed after each step. Muon momentum
  follows the bf16 compute precision, while AdamW moments remain fp32. Without fp32
  masters, an update below half a bf16 ULP can round away on every write-back.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.optim.optimizer import Optimizer

from ml.training.pretrain.muon import Muon
from ml.training.pretrain.optimizer_contracts import (
    MUON_EPS,
    MUON_MOMENTUM,
    MUON_NESTEROV,
    MUON_NS_COEFFICIENTS,
    MUON_NS_STEPS,
    TorchMuonHybridState,
)
from ml.training.pretrain.optimizer_diagnostics import adamw_group_diagnostics
from ml.training.pretrain.optimizer_grouping import (
    _split_params_for_hybrid,
    build_hybrid_param_groups,
)

TORCH_MUON_HYBRID = "torch_muon_hybrid"
SUPPORTED_PRETRAIN_OPTIMIZERS = frozenset({TORCH_MUON_HYBRID})


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


def _iter_tensors(value: object) -> Iterator[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _constant_memory_isfinite(tensor: torch.Tensor) -> bool:
    pending = [tensor]
    while pending:
        chunk = pending.pop()
        if chunk.numel() <= 1_048_576:
            if not bool(torch.isfinite(chunk).all().item()):
                return False
            continue
        split_dim = max(range(chunk.ndim), key=lambda index: chunk.shape[index])
        split_at = int(chunk.shape[split_dim]) // 2
        pending.append(
            chunk.narrow(split_dim, split_at, int(chunk.shape[split_dim]) - split_at)
        )
        pending.append(chunk.narrow(split_dim, 0, split_at))
    return True


def _first_nonfinite_entry(
    entries: list[tuple[str, torch.Tensor]],
) -> str | None:
    finite_by_device: dict[torch.device, torch.Tensor] = {}
    for _name, tensor in entries:
        check_tensor = torch.view_as_real(tensor) if tensor.is_complex() else tensor
        if check_tensor.numel() == 0:
            continue
        minimum, maximum = torch.aminmax(check_tensor)
        tensor_finite = torch.isfinite(minimum) & torch.isfinite(maximum)
        device_finite = finite_by_device.get(check_tensor.device)
        if device_finite is None:
            finite_by_device[check_tensor.device] = tensor_finite
        else:
            device_finite.logical_and_(tensor_finite)

    for device, device_finite in finite_by_device.items():
        if not bool(device_finite.item()):
            for name, tensor in entries:
                if tensor.device == device and not _constant_memory_isfinite(tensor):
                    return name
            return "state"
    return None


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
        muon_momentum: float = MUON_MOMENTUM,
        muon_nesterov: bool = MUON_NESTEROV,
        muon_ns_coefficients: tuple[float, float, float] = MUON_NS_COEFFICIENTS,
        muon_eps: float = MUON_EPS,
        muon_ns_steps: int = MUON_NS_STEPS,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("TorchMuonFusedAdamW requires a torch.nn.Module")

        muon_param_groups, adamw_groups = build_hybrid_param_groups(
            model,
            weight_decay=float(weight_decay),
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
            preserve_gradients=False,
            update_dtype=torch.bfloat16,
            momentum_dtype=torch.bfloat16,
            ns_coefficients=tuple(float(x) for x in muon_ns_coefficients),
            eps=float(muon_eps),
            ns_steps=int(muon_ns_steps),
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

        pair_by_model_id = {
            id(model_param): (model_param, master)
            for model_param, master in self._master_pairs
        }
        self._muon_master_pairs = [
            pair_by_model_id[id(model_param)]
            for model_param in self._model_muon_params
        ]
        self._adamw_master_pairs = [
            pair_by_model_id[id(model_param)]
            for model_param in self._model_adamw_params
        ]
        self._gradient_buffers: dict[int, torch.Tensor] = {}
        self._master_by_model_id = {
            id(model_parameter): master
            for model_parameter, master in self._master_pairs
        }

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

    def set_gradient_buffers(
        self,
        pairs: list[tuple[torch.Tensor, torch.Tensor]]
        | tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> None:
        expected = {id(model_param) for model_param, _master in self._master_pairs}
        received = {id(model_param) for model_param, _gradient in pairs}
        if received != expected or len(received) != len(pairs):
            raise RuntimeError("gradient buffers do not match optimizer parameters")
        buffers: dict[int, torch.Tensor] = {}
        for model_param, gradient in pairs:
            if not torch.is_tensor(gradient):
                raise TypeError("gradient buffers must contain tensors")
            if gradient.dtype != torch.float32:
                raise TypeError("gradient buffers must use float32")
            if gradient.shape != model_param.shape or gradient.device != model_param.device:
                raise ValueError("gradient buffer shape/device does not match its parameter")
            buffers[id(model_param)] = gradient
        self._gradient_buffers = buffers

    @torch.no_grad()
    def initialize_state(self) -> None:
        """Materialize all lazy optimizer tensors before graph capture."""
        self._muon_opt.initialize_state()
        if self._adamw_opt is None:
            return
        for group in self._adamw_opt.param_groups:
            for parameter in group["params"]:
                state = self._adamw_opt.state[parameter]
                state.setdefault(
                    "step",
                    torch.zeros((), dtype=torch.float32, device=parameter.device),
                )
                state.setdefault("exp_avg", torch.zeros_like(parameter))
                state.setdefault("exp_avg_sq", torch.zeros_like(parameter))

    @torch.no_grad()
    def apply_mla_qk_clip(
        self,
        mla_modules: tuple[torch.nn.Module, ...],
        max_logits: torch.Tensor,
        *,
        threshold: float,
    ) -> None:
        if max_logits.ndim != 2 or int(max_logits.size(0)) != len(mla_modules):
            raise RuntimeError("MLA QK-Clip received invalid per-head logits")
        scale = torch.sqrt(float(threshold) / max_logits.clamp_min(float(threshold)))
        for module_index, module in enumerate(mla_modules):
            num_heads = int(module.num_heads)
            head_dim = int(module.head_dim)
            if int(max_logits.size(1)) != num_heads:
                raise RuntimeError("MLA QK-Clip head count mismatch")
            for name in ("q_up", "k_up"):
                parameter = getattr(module, name).weight
                head_scale = scale[module_index].view(num_heads, 1, 1)
                parameter.view(num_heads, head_dim, -1).mul_(head_scale)
                master = self._master_by_model_id[id(parameter)]
                master.view(num_heads, head_dim, -1).mul_(head_scale)

    def _sync_grad_pairs(
        self,
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        for model_param, master in pairs:
            grad = self._gradient_buffers.get(id(model_param), model_param.grad)
            if grad is None:
                master.grad = None
                continue
            if grad.dtype == torch.float32 and grad.device == master.device:
                master.grad = grad
            else:
                existing = master.grad
                if existing is None or existing.shape != grad.shape:
                    master.grad = grad.detach().to(dtype=torch.float32)
                else:
                    existing.copy_(grad)

    @staticmethod
    def _clear_master_grad_pairs(
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        for _model_param, master in pairs:
            master.grad = None

    def _sync_grads_to_masters(self) -> None:
        self._sync_grad_pairs(self._master_pairs)

    def _sync_masters_to_model(self) -> None:
        for model_param, master in self._master_pairs:
            model_param.data.copy_(master.data)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Muon and AdamW own disjoint parameter sets. Synchronize and update them
        # one at a time so fp32 master gradients for the full 1B model are never
        # simultaneously resident with all bf16 model gradients. This is
        # mathematically identical to syncing both sets before either optimizer,
        # while removing the resume-time peak that can exceed a 16 GB GPU once all
        # optimizer state is mature.
        self._sync_grad_pairs(self._muon_master_pairs)
        try:
            self._muon_opt.step()
        finally:
            self._clear_master_grad_pairs(self._muon_master_pairs)
        if self._adamw_opt is not None:
            self._sync_grad_pairs(self._adamw_master_pairs)
            try:
                self._adamw_opt.step()
            finally:
                self._clear_master_grad_pairs(self._adamw_master_pairs)
        self._sync_masters_to_model()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        dense_gradients: list[torch.Tensor] = []
        for model_param, master in self._master_pairs:
            for tensor in (model_param, master):
                grad = tensor.grad
                if grad is None:
                    continue
                if bool(set_to_none):
                    tensor.grad = None
                else:
                    grad.detach_()
                    dense_gradients.append(grad)
        if dense_gradients:
            torch._foreach_zero_(dense_gradients)
        gradient_buffers = list(self._gradient_buffers.values())
        if gradient_buffers:
            torch._foreach_zero_(gradient_buffers)

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
            "optimizer/muon/param_count": float(len(self._model_muon_params)),
        }
        if self._adamw_opt is None:
            payload["optimizer/adamw/group_count"] = 0.0
            payload["optimizer/adamw/param_count"] = 0.0
            return payload

        payload["optimizer/adamw/group_count"] = float(len(self._adamw_opt.param_groups))
        payload["optimizer/adamw/param_count"] = float(len(self._model_adamw_params))
        payload.update(
            adamw_group_diagnostics(
                param_groups=self._adamw_opt.param_groups,
                state=self._adamw_opt.state,
            )
        )
        return payload

    @torch.no_grad()
    def check_finite_state(self) -> None:
        entries: list[tuple[str, torch.Tensor]] = []
        for index, (model_param, master) in enumerate(self._master_pairs):
            entries.append((f"parameter[{index}]", model_param.data))
            entries.append((f"master[{index}]", master))
            gradient = self._gradient_buffers.get(id(model_param))
            if gradient is not None:
                entries.append((f"gradient[{index}]", gradient))
        for optimizer_name, inner_optimizer in (
            ("muon", self._muon_opt),
            ("adamw", self._adamw_opt),
        ):
            if inner_optimizer is None:
                continue
            for parameter_index, state in enumerate(inner_optimizer.state.values()):
                for tensor_index, value in enumerate(_iter_tensors(state)):
                    if value.is_floating_point() or value.is_complex():
                        entries.append(
                            (
                                f"{optimizer_name}.state[{parameter_index}]"
                                f".tensor[{tensor_index}]",
                                value,
                            )
                        )
        if not entries:
            return
        nonfinite_name = _first_nonfinite_entry(entries)
        if nonfinite_name is not None:
            raise RuntimeError(f"non-finite optimizer state: {nonfinite_name}")


def create_torch_muon_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    muon_ns_steps: int | None = None,
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
        muon_ns_steps=int(ns_steps) if ns_steps is not None else MUON_NS_STEPS,
    )


__all__ = [
    "SUPPORTED_PRETRAIN_OPTIMIZERS",
    "TORCH_MUON_HYBRID",
    "TorchMuonFusedAdamW",
    "_split_params_for_hybrid",
    "create_torch_muon_optimizer",
]
