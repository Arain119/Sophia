from __future__ import annotations

from ml.runtime.api import (
    SupportsRuntimeRecipeControl,
    resolve_runtime_control,
)


RuntimeRecipeTarget = SupportsRuntimeRecipeControl


def require_runtime_recipe_control(
    model: RuntimeRecipeTarget,
) -> SupportsRuntimeRecipeControl:
    control = resolve_runtime_control(model, SupportsRuntimeRecipeControl)
    if control is None:
        model_type = type(model).__name__
        raise TypeError(
            "pretrain runtime recipe controls require a runtime-backed model "
            f"implementing SupportsRuntimeRecipeControl (got {model_type})"
        )
    return control


def sanitize_runtime_recipe_knobs(
    model: RuntimeRecipeTarget,
    *,
    loss_chunk_size: int,
    gradient_checkpointing_exclude_first: int,
    gradient_checkpointing_exclude_last: int,
) -> tuple[int, int, int]:
    control = require_runtime_recipe_control(model)
    chunk_size = int(loss_chunk_size)
    exclude_first = int(gradient_checkpointing_exclude_first)
    exclude_last = int(gradient_checkpointing_exclude_last)
    if not control.supports_loss_chunk_size():
        chunk_size = 0
    if not control.supports_checkpoint_excludes():
        exclude_first = 0
        exclude_last = 0
    return int(chunk_size), int(exclude_first), int(exclude_last)


__all__ = [
    "require_runtime_recipe_control",
    "sanitize_runtime_recipe_knobs",
]
