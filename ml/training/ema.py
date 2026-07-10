from __future__ import annotations

import contextlib
from typing import Any

import torch


class ModelEMA:
    """
    Exponential Moving Average (EMA) of model parameters.

    Keep a shadow copy of parameters and optionally swap them into the model for
    eval / final model export.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        decay: float,
        update_interval: int = 1,
    ) -> None:
        self.decay = float(decay)
        self.update_interval = max(int(update_interval), 1)
        self._update_step = 0

        if not (0.0 < float(self.decay) < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1) (got {decay}).")

        named: list[tuple[str, torch.nn.Parameter]] = []
        for name, p in model.named_parameters():
            if not torch.is_tensor(p):
                continue
            named.append((str(name), p))
        if not named:
            raise ValueError("ModelEMA requires at least one parameter.")

        self._named_params = named
        self._shadow: list[torch.Tensor] = []
        for _name, p in self._named_params:
            t = p.detach().clone().float()
            t.requires_grad_(False)
            self._shadow.append(t)

        self._backup: list[torch.Tensor] | None = None

    @property
    def enabled(self) -> bool:
        return True

    @torch.no_grad()
    def update(self) -> None:
        self._update_step += 1
        if int(self._update_step) % int(self.update_interval) != 0:
            return

        decay = float(self.decay)
        one_minus = 1.0 - decay
        for (_name, p), shadow in zip(self._named_params, self._shadow, strict=True):
            if not torch.is_tensor(p):
                continue
            param = p.detach()
            if not bool(p.requires_grad):
                shadow.copy_(param)
                continue
            shadow.mul_(decay).add_(param, alpha=one_minus)

    @contextlib.contextmanager
    def apply_to_model(self) -> Any:
        if self._backup is not None:
            raise RuntimeError("ModelEMA.apply_to_model() does not support nesting.")

        backup: list[torch.Tensor] = []
        for (_name, p), shadow in zip(self._named_params, self._shadow, strict=True):
            backup.append(p.data)
            p.data = shadow.to(dtype=p.data.dtype)
        self._backup = backup

        try:
            yield
        finally:
            for (_name, p), original in zip(self._named_params, backup, strict=True):
                p.data = original
            self._backup = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": float(self.decay),
            "update_interval": int(self.update_interval),
            "update_step": int(self._update_step),
            "names": [name for name, _ in self._named_params],
            "shadow": [tensor.detach() for tensor in self._shadow],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("ModelEMA.load_state_dict expects a dict.")

        required = ("decay", "update_interval", "update_step", "names", "shadow")
        missing = [key for key in required if key not in state]
        if missing:
            raise ValueError(
                "ModelEMA state is missing required fields: " + ", ".join(missing)
            )

        names = state.get("names")
        shadow = state.get("shadow")
        if not isinstance(names, list) or not isinstance(shadow, list):
            raise TypeError("ModelEMA state fields 'names'/'shadow' must be lists.")
        if len(names) != len(self._named_params) or len(shadow) != len(self._shadow):
            raise ValueError("ModelEMA state size mismatch.")

        for idx, (expected, got) in enumerate(
            zip((name for name, _ in self._named_params), names, strict=True)
        ):
            if str(got) != str(expected):
                raise ValueError(
                    "ModelEMA parameter name mismatch at index "
                    f"{idx}: expected={expected!r} got={got!r}"
                )

        self.decay = float(state.get("decay", self.decay))
        self.update_interval = max(
            int(state.get("update_interval", self.update_interval)),
            1,
        )
        self._update_step = int(state.get("update_step", self._update_step))

        for idx, ((_name, p), tensor) in enumerate(
            zip(self._named_params, shadow, strict=True)
        ):
            if not torch.is_tensor(tensor):
                raise TypeError(f"ModelEMA shadow[{idx}] is not a tensor.")
            if tensor.size() != p.size():
                raise ValueError(
                    f"ModelEMA shadow[{idx}] shape mismatch: {tuple(tensor.size())} vs {tuple(p.size())}"
                )
            self._shadow[idx] = tensor.to(device=p.device, dtype=torch.float32).detach()
            self._shadow[idx].requires_grad_(False)
