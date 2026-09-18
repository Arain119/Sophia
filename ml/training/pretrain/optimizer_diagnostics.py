from __future__ import annotations

from collections.abc import Iterable

import torch


def _diag_key(group_name: str, suffix: str) -> str:
    normalized = str(group_name or "group").strip().lower().replace(" ", "_")
    return f"optimizer/{normalized}/{suffix}"


@torch.no_grad()
def adamw_group_diagnostics(
    *,
    param_groups: Iterable[dict[str, object]],
    state: dict[torch.Tensor, dict[object, object]],
) -> dict[str, float]:
    payload: dict[str, float] = {}
    for group in param_groups:
        params = group.get("params", ())
        if not isinstance(params, (list, tuple)):
            continue
        group_name = str(group.get("name", "adamw"))
        lr = float(group.get("lr", 0.0) or 0.0)
        eps = float(group.get("eps", 0.0) or 0.0)
        denom_min = float("inf")
        inv_denom_max = 0.0
        tracked_params = 0

        for param in params:
            if not torch.is_tensor(param):
                continue
            param_state = state.get(param)
            if not isinstance(param_state, dict):
                continue
            exp_avg_sq = param_state.get("exp_avg_sq")
            if not torch.is_tensor(exp_avg_sq) or exp_avg_sq.numel() <= 0:
                continue
            denom_floor = (
                exp_avg_sq.detach()
                .amin()
                .to(dtype=torch.float32)
                .clamp_min(0.0)
                .sqrt()
            )
            denom_floor_value = float(denom_floor.item()) + float(eps)
            if denom_floor_value <= 0.0:
                continue
            tracked_params += 1
            denom_min = min(float(denom_min), float(denom_floor_value))
            inv_denom_max = max(float(inv_denom_max), 1.0 / float(denom_floor_value))

        if tracked_params <= 0:
            continue
        payload[_diag_key(group_name, "tracked_params")] = float(tracked_params)
        payload[_diag_key(group_name, "denom_min")] = float(denom_min)
        payload[_diag_key(group_name, "effective_lr_max")] = float(lr) * float(
            inv_denom_max
        )
    return payload


__all__ = ["adamw_group_diagnostics"]
