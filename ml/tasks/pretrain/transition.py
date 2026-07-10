from __future__ import annotations

from ml.data.token_shards import shape_changed_resume_incompatibility
from ml.training.pretrain.resources import PretrainManifestLike
from ml.training.pretrain.train_state import PretrainTrainState
from ml.tasks.pretrain.planner import (
    StagePlan,
    RuntimeRecipe,
)


def stage_recipe_block_tokens(recipe: RuntimeRecipe) -> int:
    if int(recipe.batch_size) <= 0:
        return int(recipe.seq_len)
    return int(recipe.seq_len) * int(recipe.batch_size)


def stage_resume_state_requires_transition_step(
    *,
    manifest: PretrainManifestLike,
    resume_train_state: PretrainTrainState | dict[str, object] | None,
    next_recipe: RuntimeRecipe,
) -> bool:
    state = PretrainTrainState.resolve(resume_train_state)
    data_state = state.data_iter_state
    if data_state is None:
        return False
    shards = tuple(getattr(manifest, "shards", ()) or ())
    if not shards or int(data_state.order_pos) >= len(data_state.order):
        return False
    next_block_tokens = int(stage_recipe_block_tokens(next_recipe))
    try:
        incompatibility = shape_changed_resume_incompatibility(
            shard_tokens=[int(getattr(shard, "tokens", 0) or 0) for shard in shards],
            order=[int(x) for x in data_state.order],
            order_pos=int(data_state.order_pos),
            current_pos=data_state.current_pos,
            block_tokens=int(next_block_tokens),
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from None
    return incompatibility is not None


def pending_stage_boundary_index(
    *,
    stage_plans: list[StagePlan],
    current_step: int,
    current_train_state: PretrainTrainState | dict[str, object] | None = None,
    manifest: PretrainManifestLike | None = None,
) -> int | None:
    for idx, plan in enumerate(stage_plans[:-1]):
        next_plan = stage_plans[int(idx) + 1]
        if int(current_step) == int(plan.end_step):
            return int(idx)
        if int(current_step) <= int(plan.end_step):
            continue
        if manifest is None:
            continue
        if stage_resume_state_requires_transition_step(
            manifest=manifest,
            resume_train_state=current_train_state,
            next_recipe=next_plan.recipe,
        ):
            return int(idx)
    return None
