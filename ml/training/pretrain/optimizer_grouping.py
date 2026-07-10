from __future__ import annotations

import re

import torch
from torch import Tensor

from ml.modeling.config_introspection import resolve_config_num_hidden_layers
from ml.training.pretrain.optimizer_contracts import OptimizerParamGroupSpec


def _is_embedding_or_lm_head(name: str) -> bool:
    normalized = str(name)
    return normalized.endswith((
        "embed_tokens.weight",
        "tok_embeddings.weight",
        "lm_head.weight",
        "output.weight",
    ))


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def _infer_num_layers(model: torch.nn.Module) -> int:
    cfg = getattr(model, "config", None)
    n = resolve_config_num_hidden_layers(cfg)
    if isinstance(n, int) and n > 0:
        return int(n)

    max_layer = -1
    for name, _param in model.named_parameters():
        match = _LAYER_RE.match(str(name))
        if match is None:
            continue
        idx = int(match.group(1))
        if idx > max_layer:
            max_layer = idx
    return int(max_layer + 1) if max_layer >= 0 else 0


def _param_depth(name: str, *, num_layers: int) -> int:
    normalized = str(name)
    if normalized.startswith(("model.embed_tokens.", "model.tok_embeddings.")):
        return 0

    match = _LAYER_RE.match(normalized)
    if match is not None:
        return 1 + int(match.group(1))

    if normalized.startswith(("model.norm.", "lm_head.", "model.output.")):
        return int(num_layers + 1)
    return int(num_layers + 1)


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

        if is_embed and param.ndim >= 2:
            embed_decay.append(param)
        elif str(name).endswith(".bias") or param.ndim < 2:
            no_decay.append(param)
        else:
            decay.append(param)

    return muon_params, embed_decay, decay, no_decay


def _shape_key_2d(param: Tensor) -> tuple[int, int]:
    if param.ndim != 2:
        raise ValueError(f"Expected a 2D tensor (got shape={tuple(param.shape)!r})")
    return int(param.shape[0]), int(param.shape[1])


def _group_params_for_hybrid_layerwise(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    layerwise_lr_decay: float,
    embedding_lr_scale: float = 1.0,
) -> tuple[list[OptimizerParamGroupSpec], list[OptimizerParamGroupSpec]]:
    base_lr = float(lr)
    if base_lr <= 0.0:
        raise ValueError(f"learning rate must be > 0, got {base_lr}")
    resolved_weight_decay = float(weight_decay)
    if resolved_weight_decay < 0.0:
        raise ValueError(f"weight decay must be >= 0, got {resolved_weight_decay}")
    llrd = float(layerwise_lr_decay)
    if not (0.0 < llrd <= 1.0):
        raise ValueError(f"layerwise_lr_decay must be in (0, 1], got {llrd}")
    if float(embedding_lr_scale) <= 0.0:
        raise ValueError(
            f"embedding_lr_scale must be > 0, got {float(embedding_lr_scale)}"
        )

    num_layers = _infer_num_layers(model)
    max_depth = int(num_layers + 1)

    def _lr_for_depth(depth: int) -> float:
        if llrd == 1.0:
            return float(base_lr)
        resolved_depth = min(max(int(depth), 0), max_depth)
        exponent = int(max_depth - resolved_depth)
        return float(base_lr) * float(llrd ** exponent)

    muon_groups: dict[tuple[int, tuple[int, int]], list[Tensor]] = {}
    adamw_embed_groups: dict[int, list[Tensor]] = {}
    adamw_decay_groups: dict[int, list[Tensor]] = {}
    adamw_no_decay_groups: dict[int, list[Tensor]] = {}

    for name, param in model.named_parameters():
        if not torch.is_tensor(param) or not param.requires_grad:
            continue

        depth = _param_depth(str(name), num_layers=int(num_layers))
        is_embed = _is_embedding_or_lm_head(str(name))
        if param.ndim == 2 and not is_embed:
            key = (int(depth), _shape_key_2d(param))
            muon_groups.setdefault(key, []).append(param)
            continue

        if is_embed and param.ndim >= 2:
            adamw_embed_groups.setdefault(int(depth), []).append(param)
        elif str(name).endswith(".bias") or param.ndim < 2:
            adamw_no_decay_groups.setdefault(int(depth), []).append(param)
        else:
            adamw_decay_groups.setdefault(int(depth), []).append(param)

    if not muon_groups:
        raise RuntimeError("No Muon parameters found (expected 2D transformer weights).")

    muon_param_groups: list[OptimizerParamGroupSpec] = []
    for (depth, shape), params in sorted(muon_groups.items()):
        if not params:
            continue
        muon_param_groups.append(
            OptimizerParamGroupSpec(
                name=f"muon_d{int(depth)}_{int(shape[0])}x{int(shape[1])}",
                params=params,
                lr=float(_lr_for_depth(depth)),
                weight_decay=float(resolved_weight_decay),
            )
        )

    adamw_param_groups: list[OptimizerParamGroupSpec] = []
    for depth in sorted(
        set(adamw_embed_groups) | set(adamw_decay_groups) | set(adamw_no_decay_groups)
    ):
        embed_params = adamw_embed_groups.get(depth) or []
        if embed_params:
            adamw_param_groups.append(
                OptimizerParamGroupSpec(
                    name=f"adamw_embed_d{int(depth)}",
                    params=embed_params,
                    lr=float(_lr_for_depth(depth)) * float(embedding_lr_scale),
                    weight_decay=0.0,
                )
            )
        decay_params = adamw_decay_groups.get(depth) or []
        if decay_params:
            adamw_param_groups.append(
                OptimizerParamGroupSpec(
                    name=f"adamw_decay_d{int(depth)}",
                    params=decay_params,
                    lr=float(_lr_for_depth(depth)),
                    weight_decay=float(resolved_weight_decay),
                )
            )
        no_decay_params = adamw_no_decay_groups.get(depth) or []
        if no_decay_params:
            adamw_param_groups.append(
                OptimizerParamGroupSpec(
                    name=f"adamw_no_decay_d{int(depth)}",
                    params=no_decay_params,
                    lr=float(_lr_for_depth(depth)),
                    weight_decay=0.0,
                )
            )

    return muon_param_groups, adamw_param_groups


def build_hybrid_param_groups(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    layerwise_lr_decay: float,
    embedding_lr_scale: float = 1.0,
) -> tuple[list[OptimizerParamGroupSpec], list[OptimizerParamGroupSpec]]:
    llrd = float(layerwise_lr_decay)
    if not (0.0 < llrd <= 1.0):
        raise ValueError(f"layerwise_lr_decay must be in (0, 1], got {llrd}")
    if float(lr) <= 0.0:
        raise ValueError(f"learning rate must be > 0, got {float(lr)}")
    if float(weight_decay) < 0.0:
        raise ValueError(f"weight decay must be >= 0, got {float(weight_decay)}")
    if float(embedding_lr_scale) <= 0.0:
        raise ValueError(
            f"embedding_lr_scale must be > 0, got {float(embedding_lr_scale)}"
        )
    if 0.0 < llrd < 1.0:
        return _group_params_for_hybrid_layerwise(
            model,
            lr=float(lr),
            weight_decay=float(weight_decay),
            layerwise_lr_decay=float(llrd),
            embedding_lr_scale=float(embedding_lr_scale),
        )

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
    embed_lr = float(lr) * float(embedding_lr_scale)
    if adamw_embed:
        adamw_groups.append(
            OptimizerParamGroupSpec(
                name="adamw_embed",
                params=adamw_embed,
                lr=float(embed_lr),
                # No weight decay on the (tied) token embeddings: the output-side
                # regularization is already supplied by z-loss, while
                # L2 decay here only shrinks rare-token rows that rarely receive a
                # gradient. Harmless under bf16 (the decay rounds away), but real
                # once fp32 master weights make small steps land -- so pin it to 0.
                weight_decay=0.0,
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
    "_group_params_for_hybrid_layerwise",
    "_split_params_for_hybrid",
    "build_hybrid_param_groups",
]
