from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

import ml.tasks.pretrain.transition as transition
from ml.data.token_shards.shard_manifest import (
    load_manifest,
    validate_tokenizer_fingerprint,
)
from ml.training.pretrain.resources import PretrainManifestLike
from ml.tasks.pretrain.deps import PretrainDeps
from ml.tasks.pretrain.planner import StagePlan, RuntimeRecipe
from ml.tasks.pretrain.session import (
    attach_pretrain_session,
    ensure_pretrain_session,
)
from ml.training.ema import ModelEMA
from ml.training.pretrain import manifest_policy
from ml.training.pretrain.train_state import PretrainTrainState

if TYPE_CHECKING:
    from ml.tasks.pretrain.pipeline import PretrainPipeline
    from ml.tasks.pretrain.session import PretrainSession


@dataclass
class PretrainStageCursor:
    current_step: int
    train_state: PretrainTrainState
    fresh_start_stage_indices: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class StageBoundaryOutcome:
    cursor: PretrainStageCursor
    manifest_changed: bool
    made_progress: bool


def _reset_stage_compilation_state(
    *,
    session: PretrainSession,
    from_seq_len: int,
    to_seq_len: int,
) -> None:
    session.runner = None
    gc.collect()
    print(
        "[STAGE] runtime reset | "
        f"from_seq_len={int(from_seq_len)} to_seq_len={int(to_seq_len)}",
        flush=True,
    )
    torch._dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_runtime_stage(
    *,
    pipeline: PretrainPipeline,
    deps: PretrainDeps,
    recipe: RuntimeRecipe,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> None:
    session = ensure_pretrain_session(pipeline)
    previous_seq_len = int(getattr(session, "seq_len", 0) or 0)
    next_seq_len = int(recipe.seq_len)
    seq_len_changed = next_seq_len != previous_seq_len
    args = deps.apply_stage_recipe(
        args=pipeline.args,
        model=session.model,
        recipe=recipe,
        optimizer=optimizer,
        scheduler=scheduler,
        runtime_state=session.runtime_state,
    )
    session.seq_len = int(next_seq_len)
    if seq_len_changed:
        _reset_stage_compilation_state(
            session=session,
            from_seq_len=int(previous_seq_len),
            to_seq_len=int(next_seq_len),
        )
    session.runner = deps.build_runner(
        args=args,
        session=session,
    )
    session.absorb_run_config(args)
    attach_pretrain_session(pipeline, session, args=args)


def run_stage_transition_step(
    *,
    pipeline: PretrainPipeline,
    deps: PretrainDeps,
    manifest: PretrainManifestLike,
    manifest_path: str,
    recipe: RuntimeRecipe,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ema: ModelEMA | None,
    current_step: int,
    current_train_state: PretrainTrainState,
    from_seq_len: int,
    to_seq_len: int,
    prefix: str,
) -> tuple[int, PretrainTrainState]:
    session = ensure_pretrain_session(pipeline)
    end_step = int(min(int(current_step) + 1, int(session.max_steps)))
    print(
        f"{prefix} | from_seq_len={int(from_seq_len)} "
        f"to_seq_len={int(to_seq_len)} | step={int(current_step)}->{int(current_step) + 1}",
        flush=True,
    )
    transition_result = deps.run_train_loop(
        args=pipeline.args,
        model=session.model,
        optimizer=optimizer,
        scheduler=scheduler,
        runner=session.runner,
        tokenizer=session.tokenizer,
        output_dir=str(session.output_dir),
        manifest=manifest,
        manifest_path=str(manifest_path),
        device=session.device,
        base_dtype=session.base_dtype,
        seq_len=int(recipe.seq_len),
        start_step=int(current_step),
        end_step=int(end_step),
        resume_train_state=current_train_state,
        allow_fresh_data_iter_start=False,
        safe_serialization=bool(int(pipeline.args.safe_serialization)),
        ema=ema,
        total_max_steps=int(session.max_steps),
        final_stage=bool(int(end_step) >= int(session.max_steps)),
    )
    return int(transition_result.last_step), PretrainTrainState.from_payload(
        transition_result.train_state
    )


def resolve_stage_manifest(
    *,
    pipeline: PretrainPipeline,
    plan: StagePlan,
) -> tuple[PretrainManifestLike, str]:
    """Resolve the manifest a stage trains on (per-stage dataset override aware)."""
    session = ensure_pretrain_session(pipeline)
    data_path = str(getattr(plan, "data_path", "") or "").strip()
    if not data_path:
        return session.manifest, str(session.manifest_path)
    manifest_path = manifest_policy.resolve_manifest_path(str(data_path))
    if manifest_policy.paths_same(str(manifest_path), str(session.manifest_path)):
        return session.manifest, str(session.manifest_path)
    manifest = load_manifest(str(manifest_path))
    validate_tokenizer_fingerprint(
        tokenizer_dir=str(pipeline.args.tokenizer_path),
        expected_sha1=str(manifest.tokenizer_sha1),
    )
    return manifest, str(manifest_path)


def stage_start_index(
    *,
    stage_plans: list[StagePlan],
    current_step: int,
) -> int:
    for idx, plan in enumerate(stage_plans):
        if int(plan.end_step) > int(current_step):
            return int(idx)
    return 0


def advance_stage_boundary(
    *,
    pipeline: PretrainPipeline,
    deps: PretrainDeps,
    boundary_manifest: PretrainManifestLike,
    boundary_manifest_path: str,
    boundary_recipe: RuntimeRecipe,
    boundary_stage_index: int,
    next_recipe: RuntimeRecipe,
    next_stage_index: int,
    next_manifest_path: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ema: ModelEMA | None,
    cursor: PretrainStageCursor,
    boundary_prefix: str | None = None,
    transition_prefix: str = "[INFO] stage transition",
) -> StageBoundaryOutcome:
    session = ensure_pretrain_session(pipeline)
    manifest_changed = not manifest_policy.paths_same(
        boundary_manifest_path,
        next_manifest_path,
    )
    boundary_step = int(cursor.current_step)
    if boundary_prefix is not None:
        print(
            f"{boundary_prefix} | from_idx={int(boundary_stage_index)} "
            f"to_idx={int(next_stage_index)} | step={int(cursor.current_step)} "
            f"| manifest_changed={int(bool(manifest_changed))}",
            flush=True,
        )
    while (
        int(cursor.current_step) < int(session.max_steps)
        and not bool(manifest_changed)
        and transition.stage_resume_state_requires_transition_step(
            manifest=boundary_manifest,
            resume_train_state=cursor.train_state,
            next_recipe=next_recipe,
        )
    ):
        cursor.current_step, cursor.train_state = run_stage_transition_step(
            pipeline=pipeline,
            deps=deps,
            manifest=boundary_manifest,
            manifest_path=str(boundary_manifest_path),
            recipe=boundary_recipe,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            current_step=int(cursor.current_step),
            current_train_state=cursor.train_state,
            from_seq_len=int(boundary_recipe.seq_len),
            to_seq_len=int(next_recipe.seq_len),
            prefix=transition_prefix,
        )
    if int(cursor.current_step) < int(session.max_steps):
        if bool(manifest_changed):
            cursor.fresh_start_stage_indices.add(int(next_stage_index))
        cursor.train_state = cursor.train_state.cleared_for_stage_transition(
            clear_data_iter_state=bool(manifest_changed)
        )
    return StageBoundaryOutcome(
        cursor=cursor,
        manifest_changed=bool(manifest_changed),
        made_progress=int(cursor.current_step) != int(boundary_step),
    )


def run_resume_boundaries(
    *,
    pipeline: PretrainPipeline,
    deps: PretrainDeps,
    stage_plans: list[StagePlan],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ema: ModelEMA | None,
    cursor: PretrainStageCursor,
) -> PretrainStageCursor:
    session = ensure_pretrain_session(pipeline)

    def _cursor_plan_manifest() -> PretrainManifestLike:
        idx = stage_start_index(
            stage_plans=stage_plans,
            current_step=int(cursor.current_step),
        )
        manifest, _ = resolve_stage_manifest(
            pipeline=pipeline,
            plan=stage_plans[int(idx)],
        )
        return manifest

    boundary_idx = transition.pending_stage_boundary_index(
        stage_plans=stage_plans,
        current_step=int(cursor.current_step),
        current_train_state=cursor.train_state,
        manifest=_cursor_plan_manifest(),
    )
    while boundary_idx is not None and int(cursor.current_step) < int(session.max_steps):
        boundary_plan = stage_plans[int(boundary_idx)]
        next_plan = stage_plans[int(boundary_idx) + 1]
        boundary_manifest, boundary_manifest_path = resolve_stage_manifest(
            pipeline=pipeline,
            plan=boundary_plan,
        )
        _, next_manifest_path = resolve_stage_manifest(
            pipeline=pipeline,
            plan=next_plan,
        )
        apply_runtime_stage(
            pipeline=pipeline,
            deps=deps,
            recipe=boundary_plan.recipe,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        outcome = advance_stage_boundary(
            pipeline=pipeline,
            deps=deps,
            boundary_manifest=boundary_manifest,
            boundary_manifest_path=str(boundary_manifest_path),
            boundary_recipe=boundary_plan.recipe,
            boundary_stage_index=int(boundary_plan.stage.stage_index),
            next_recipe=next_plan.recipe,
            next_stage_index=int(next_plan.stage.stage_index),
            next_manifest_path=str(next_manifest_path),
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            cursor=cursor,
            boundary_prefix="[INFO] resume stage boundary",
            transition_prefix="[INFO] resume stage transition",
        )
        cursor = outcome.cursor
        if int(cursor.current_step) >= int(session.max_steps):
            break
        if not bool(outcome.made_progress):
            break
        boundary_idx = transition.pending_stage_boundary_index(
            stage_plans=stage_plans,
            current_step=int(cursor.current_step),
            current_train_state=cursor.train_state,
            manifest=_cursor_plan_manifest(),
        )
    return cursor


def run_planned_stages(
    *,
    pipeline: PretrainPipeline,
    deps: PretrainDeps,
    stage_plans: list[StagePlan],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ema: ModelEMA | None,
    cursor: PretrainStageCursor,
) -> PretrainStageCursor:
    session = ensure_pretrain_session(pipeline)
    stage_start_idx = stage_start_index(
        stage_plans=stage_plans,
        current_step=int(cursor.current_step),
    )
    for idx, plan in enumerate(stage_plans[stage_start_idx:], start=stage_start_idx):
        if int(cursor.current_step) >= int(plan.end_step):
            continue
        plan = deps.realize_stage_plan_machine_recipe(
            args=pipeline.args,
            model=session.model,
            device=session.device,
            output_dir=str(session.output_dir),
            vocab_size=int(session.vocab_size),
            base_dtype=session.base_dtype,
            plan=plan,
        )
        stage_plans[int(idx)] = plan
        session.runtime_plan.stage_execution_plans = list(stage_plans)
        allow_fresh_data_iter_start = (
            int(plan.stage.stage_index) in cursor.fresh_start_stage_indices
        )
        apply_runtime_stage(
            pipeline=pipeline,
            deps=deps,
            recipe=plan.recipe,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        plan_manifest, plan_manifest_path = resolve_stage_manifest(
            pipeline=pipeline,
            plan=plan,
        )
        print(
            f"[INFO] stage | idx={int(plan.stage.stage_index)} seq_len={int(plan.recipe.seq_len)} "
            f"tokens={int(plan.stage.start_tokens):,}->{int(plan.stage.end_tokens):,} "
            f"steps={int(max(cursor.current_step, plan.start_step))}->{int(plan.end_step)} "
            f"bs={int(plan.recipe.batch_size)} accum={int(plan.recipe.accumulation_steps)} "
            f"tokens/update={int(plan.recipe.tokens_per_update)} "
            f"data={str(plan.data_path) or 'primary'}",
            flush=True,
        )
        result = deps.run_train_loop(
            args=pipeline.args,
            model=session.model,
            optimizer=optimizer,
            scheduler=scheduler,
            runner=session.runner,
            tokenizer=session.tokenizer,
            output_dir=str(session.output_dir),
            manifest=plan_manifest,
            manifest_path=str(plan_manifest_path),
            device=session.device,
            base_dtype=session.base_dtype,
            seq_len=int(plan.recipe.seq_len),
            start_step=int(max(cursor.current_step, plan.start_step)),
            end_step=int(plan.end_step),
            resume_train_state=cursor.train_state,
            allow_fresh_data_iter_start=bool(allow_fresh_data_iter_start),
            safe_serialization=bool(int(pipeline.args.safe_serialization)),
            ema=ema,
            total_max_steps=int(session.max_steps),
            final_stage=bool(idx == len(stage_plans) - 1),
        )
        cursor.current_step = int(result.last_step)
        cursor.train_state = PretrainTrainState.from_payload(result.train_state)
        if int(cursor.current_step) >= int(session.max_steps):
            break
        if idx >= len(stage_plans) - 1:
            continue
        next_plan = stage_plans[int(idx) + 1]
        _, next_manifest_path = resolve_stage_manifest(
            pipeline=pipeline,
            plan=next_plan,
        )
        outcome = advance_stage_boundary(
            pipeline=pipeline,
            deps=deps,
            boundary_manifest=plan_manifest,
            boundary_manifest_path=str(plan_manifest_path),
            boundary_recipe=plan.recipe,
            boundary_stage_index=int(plan.stage.stage_index),
            next_recipe=next_plan.recipe,
            next_stage_index=int(next_plan.stage.stage_index),
            next_manifest_path=str(next_manifest_path),
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            cursor=cursor,
            transition_prefix="[INFO] stage transition",
        )
        cursor = outcome.cursor
        if int(cursor.current_step) >= int(session.max_steps):
            break
    return cursor


__all__ = [
    "PretrainStageCursor",
    "StageBoundaryOutcome",
    "advance_stage_boundary",
    "apply_runtime_stage",
    "resolve_stage_manifest",
    "run_planned_stages",
    "run_resume_boundaries",
    "run_stage_transition_step",
    "stage_start_index",
]
