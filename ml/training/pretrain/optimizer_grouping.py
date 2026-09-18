from __future__ import annotations

import torch
from torch import Tensor

from ml.training.pretrain.optimizer_contracts import OptimizerParamGroupSpec


def _is_embedding_or_lm_head(name: str) -> bool:
    normalized = str(name)
    return normalized.endswith((
        "embed_tokens.weight",
        "tok_embeddings.weight",
        "lm_head.weight",
        "output.weight",
    ))


def _split_params_for_hybrid(
    model: torch.nn.Module,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    muon_params: list[Tensor] = []
    embed_decay: list[Tensor] = []
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []

    for name, param in model.named_parameters():
        if not torch.is_tensor(param) or not param.requires_grad:
            continue

        is_embed = _is_embedding_or_lm_head(str(name))
        if param.ndim == 2 and not is_embed:
            muon_params.append(param)
            continue

        if bool(getattr(param, "_no_weight_decay", False)):
            no_decay.append(param)
        elif is_embed and param.ndim >= 2:
            embed_decay.append(param)
        else:
            decay.append(param)

    return muon_params, embed_decay, decay, no_decay


def _shape_key_2d(param: Tensor) -> tuple[int, int]:
    if param.ndim != 2:
        raise ValueError(f"Expected a 2D tensor (got shape={tuple(param.shape)!r})")
    return int(param.shape[0]), int(param.shape[1])


def build_hybrid_param_groups(
    model: torch.nn.Module,
    *,
    weight_decay: float,
) -> tuple[list[OptimizerParamGroupSpec], list[OptimizerParamGroupSpec]]:
    if float(weight_decay) < 0.0:
        raise ValueError(f"weight decay must be >= 0, got {float(weight_decay)}")

    muon_params, adamw_embed, adamw_decay, adamw_no_decay = _split_params_for_hybrid(
        model
    )
    if not muon_params:
        raise RuntimeError("No Muon parameters found (expected 2D transformer weights).")

    muon_groups: dict[tuple[int, int], list[torch.Tensor]] = {}
    for param in muon_params:
        muon_groups.setdefault(_shape_key_2d(param), []).append(param)
    muon_param_groups = [
        OptimizerParamGroupSpec(
            name=f"muon_{int(m)}x{int(n)}",
            params=params,
        )
        for (m, n), params in sorted(muon_groups.items())
        if params
    ]

    adamw_groups: list[OptimizerParamGroupSpec] = []
    if adamw_embed:
        adamw_groups.append(
            OptimizerParamGroupSpec(
                name="adamw_embed",
                params=adamw_embed,
                weight_decay=float(weight_decay),
            )
        )
    if adamw_decay:
        adamw_groups.append(
            OptimizerParamGroupSpec(
                name="adamw_decay",
                params=adamw_decay,
                weight_decay=float(weight_decay),
            )
        )
    if adamw_no_decay:
        adamw_groups.append(
            OptimizerParamGroupSpec(
                name="adamw_no_decay",
                params=adamw_no_decay,
                weight_decay=0.0,
            )
        )
    return muon_param_groups, adamw_groups


__all__ = [
    "_split_params_for_hybrid",
    "build_hybrid_param_groups",
]
