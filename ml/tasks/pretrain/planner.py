from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ml.core.spec import build_release_pretrain_schedule_spec

if TYPE_CHECKING:
    from ml.training.pretrain.run_config import PretrainRunConfig
    from ml.training.pretrain.runtime_state import PretrainRuntimeState
    from ml.tasks.pretrain.session import PretrainSession

_RELEASE_SCHEDULE = build_release_pretrain_schedule_spec()


@dataclass(frozen=True)
class CurriculumStage:
    stage_index: int
    seq_len: int
    start_tokens: int
    end_tokens: int

    @property
    def token_budget(self) -> int:
        return max(int(self.end_tokens) - int(self.start_tokens), 0)


@dataclass(frozen=True)
class RuntimeRecipe:
    seq_len: int
    batch_size: int
    accumulation_steps: int
    tokens_per_update: int
    gradient_checkpointing: int = 0
    gradient_checkpointing_exclude_first: int = 0
    gradient_checkpointing_exclude_last: int = 0
    loss_chunk_size: int = 0
    machine_recipe_tuned: int = 0


@dataclass(frozen=True)
class StagePlan:
    stage: CurriculumStage
    recipe: RuntimeRecipe
    start_step: int
    end_step: int
    # Optional per-stage dataset override (dataset root or train split path).
    # Empty string means the stage trains on the session's primary dataset.
    data_path: str = ""


@dataclass(frozen=True)
class TokenBudget:
    target_tokens_per_update: int
    target_tokens_per_microbatch: int


def scaled_token_schedule(
    *,
    total_tokens: int,
    source_total_tokens: int,
    source_value: int,
) -> int:
    total = max(int(total_tokens), 1)
    source_total = max(int(source_total_tokens), 1)
    source_value = max(int(source_value), 0)
    if source_value <= 0:
        return 0
    scaled = int(round(float(source_value) * (float(total) / float(source_total))))
    return max(scaled, 1)


def allocate_weighted_token_budgets(total_tokens: int, weights: list[int]) -> list[int]:
    total = max(int(total_tokens), 0)
    if total <= 0 or not weights:
        return [0 for _ in weights]
    norm = [max(int(weight), 0) for weight in weights]
    weight_sum = sum(norm)
    if weight_sum <= 0:
        base = total // len(norm)
        out = [base for _ in norm]
        for idx in range(total - sum(out)):
            out[idx % len(out)] += 1
        return out
    raw = [(float(total) * float(weight)) / float(weight_sum) for weight in norm]
    out = [int(math.floor(value)) for value in raw]
    remainder = int(total - sum(out))
    order = sorted(
        range(len(raw)),
        key=lambda idx: (raw[idx] - float(out[idx]), norm[idx]),
        reverse=True,
    )
    for idx in order[:remainder]:
        out[int(idx)] += 1
    return out


def build_length_curriculum(
    *,
    total_tokens: int,
    max_seq_len: int,
    train_seq_stages: list[int] | tuple[int, ...] | None = None,
    stage_token_weights: list[int] | tuple[int, ...] | None = None,
) -> list[CurriculumStage]:
    resolved_seq_stages = (
        [int(stage) for stage in train_seq_stages]
        if train_seq_stages is not None
        else [int(stage) for stage in _RELEASE_SCHEDULE.train_seq_stages]
    )
    resolved_stage_weights = (
        [int(weight) for weight in stage_token_weights]
        if stage_token_weights is not None
        else [int(weight) for weight in _RELEASE_SCHEDULE.stage_token_weights]
    )
    if len(resolved_seq_stages) != len(resolved_stage_weights):
        raise RuntimeError("stage_token_weights must align with train_seq_stages")
    seq_stages = sorted(
        {
            int(stage)
            for stage in list(resolved_seq_stages)
            if 0 < int(stage) <= int(max_seq_len)
        }
    )
    max_seq_len = int(max_seq_len)
    if max_seq_len > 0 and max_seq_len not in seq_stages:
        seq_stages.append(int(max_seq_len))
        seq_stages.sort()
    if not seq_stages:
        raise RuntimeError("length curriculum resolved to an empty stage list")

    total_tokens = max(int(total_tokens), 1)
    stage_weight_map = {
        int(seq_len): int(weight)
        for seq_len, weight in zip(
            resolved_seq_stages,
            resolved_stage_weights,
            strict=True,
        )
    }
    stage_budgets = allocate_weighted_token_budgets(
        int(total_tokens),
        [int(stage_weight_map.get(int(stage), int(stage))) for stage in seq_stages],
    )

    stages: list[CurriculumStage] = []
    seen_tokens = 0
    for idx, (seq_len, budget) in enumerate(zip(seq_stages, stage_budgets, strict=True)):
        budget = max(int(budget), 0)
        end_tokens = int(seen_tokens) + int(budget)
        stages.append(
            CurriculumStage(
                stage_index=int(idx),
                seq_len=int(seq_len),
                start_tokens=int(seen_tokens),
                end_tokens=int(end_tokens),
            )
        )
        seen_tokens = int(end_tokens)
    if stages:
        stages[-1] = CurriculumStage(
            stage_index=int(stages[-1].stage_index),
            seq_len=int(stages[-1].seq_len),
            start_tokens=int(stages[-1].start_tokens),
            end_tokens=int(total_tokens),
        )
    return stages


def derive_stage_runtime_recipe(
    *,
    seq_len: int,
    target_tokens_per_microbatch: int,
    target_tokens_per_update: int,
    max_batch_size: int,
) -> RuntimeRecipe:
    seq_len = max(int(seq_len), 1)
    target = max(int(target_tokens_per_update), int(seq_len))
    max_batch = max(int(max_batch_size), 1)
    micro_target = max(int(target_tokens_per_microbatch), int(seq_len))
    batch_size = max(min(int(micro_target // seq_len), int(max_batch)), 1)
    accumulation_steps = max(int(math.ceil(float(target) / float(batch_size * seq_len))), 1)
    tokens_per_update = int(seq_len) * int(batch_size) * int(accumulation_steps)
    gradient_checkpointing = 1 if int(seq_len) > int(_RELEASE_SCHEDULE.train_seq_len) else 0
    return RuntimeRecipe(
        seq_len=int(seq_len),
        batch_size=int(batch_size),
        accumulation_steps=int(accumulation_steps),
        tokens_per_update=int(tokens_per_update),
        gradient_checkpointing=int(gradient_checkpointing),
    )


def build_stage_execution_plan(
    *,
    stages: list[CurriculumStage],
    target_tokens_per_microbatch: int,
    target_tokens_per_update: int,
    max_batch_size: int,
) -> list[StagePlan]:
    plans: list[StagePlan] = []
    start_step = 0
    for stage in stages:
        if int(stage.token_budget) <= 0:
            continue
        recipe = derive_stage_runtime_recipe(
            seq_len=int(stage.seq_len),
            target_tokens_per_microbatch=int(target_tokens_per_microbatch),
            target_tokens_per_update=int(target_tokens_per_update),
            max_batch_size=int(max_batch_size),
        )
        stage_steps = max(
            int(math.ceil(float(stage.token_budget) / float(recipe.tokens_per_update))),
            1,
        )
        end_step = int(start_step) + int(stage_steps)
        plans.append(
            StagePlan(
                stage=stage,
                recipe=recipe,
                start_step=int(start_step),
                end_step=int(end_step),
            )
        )
        start_step = int(end_step)
    return plans


def resolve_stage_token_budget(
    *,
    seq_len: int,
    runtime_state: PretrainRuntimeState,
) -> TokenBudget:
    target_tokens_per_update = int(runtime_state.target_tokens_per_update)
    if target_tokens_per_update <= 0:
        target_tokens_per_update = (
            int(runtime_state.batch_size)
            * int(seq_len)
            * max(int(runtime_state.accumulation_steps), 1)
        )
    target_tokens_per_microbatch = int(runtime_state.target_tokens_per_microbatch)
    if target_tokens_per_microbatch <= 0:
        target_tokens_per_microbatch = int(runtime_state.batch_size) * int(seq_len)
    if target_tokens_per_update <= 0:
        raise RuntimeError("invalid base stage tokens/update after machine recipe resolution")
    if target_tokens_per_microbatch <= 0:
        raise RuntimeError("invalid base stage micro-batch token budget after machine recipe resolution")
    return TokenBudget(
        target_tokens_per_update=int(target_tokens_per_update),
        target_tokens_per_microbatch=int(target_tokens_per_microbatch),
    )


def apply_explicit_max_steps_cap(
    *,
    stage_execution_plans: list[StagePlan],
    max_steps: int,
) -> list[StagePlan]:
    limit = int(max_steps)
    if limit <= 0:
        return list(stage_execution_plans)
    capped: list[StagePlan] = []
    for plan in stage_execution_plans:
        start_step = int(plan.start_step)
        if start_step >= limit:
            break
        capped_end_step = min(int(plan.end_step), int(limit))
        capped.append(
            replace(
                plan,
                end_step=int(capped_end_step),
            )
        )
        if int(capped_end_step) >= limit:
            break
    return capped


def wsd_decay_start_step(
    *,
    total_steps: int,
    warmup_steps: int,
    warmup_ratio: float,
    wsd_stable_ratio: float,
) -> int:
    """Step index where the WSD decay phase begins (mirrors the scheduler math)."""
    total = max(int(total_steps), 1)
    warmup = int(warmup_steps or 0)
    if warmup <= 0:
        warmup = int(max(int(total) * float(warmup_ratio), 0))
    stable = int(max(int(total) * min(max(float(wsd_stable_ratio), 0.0), 1.0), 0))
    if stable > max(int(total) - int(warmup), 0):
        stable = max(int(total) - int(warmup), 0)
    return int(min(int(warmup) + int(stable), int(total)))


def apply_decay_data_stage(
    *,
    stage_execution_plans: list[StagePlan],
    args: PretrainRunConfig,
) -> list[StagePlan]:
    """
    Split the plan covering the WSD decay boundary so decay steps train on
    ``args.decay_data_path``. Plans keep their step geometry; only the tail
    stage's dataset changes.
    """
    decay_data_path = str(getattr(args, "decay_data_path", "") or "").strip()
    if not decay_data_path or not stage_execution_plans:
        return list(stage_execution_plans)
    schedule = str(getattr(args, "lr_schedule", "") or "").strip().lower()
    if schedule not in {"wsd", "warmup_stable_decay"}:
        raise RuntimeError(
            "decay_data_path requires lr_schedule=wsd "
            f"(got lr_schedule={schedule!r})"
        )
    total_steps = int(stage_execution_plans[-1].end_step)
    decay_start = wsd_decay_start_step(
        total_steps=int(total_steps),
        warmup_steps=int(getattr(args, "warmup_steps", 0) or 0),
        warmup_ratio=float(getattr(args, "warmup_ratio", 0.0) or 0.0),
        wsd_stable_ratio=float(getattr(args, "wsd_stable_ratio", 0.0) or 0.0),
    )
    if decay_start >= total_steps:
        raise RuntimeError(
            "decay_data_path is set but the WSD schedule has no decay steps "
            f"(decay_start={decay_start} >= total_steps={total_steps})"
        )

    out: list[StagePlan] = []
    for plan in stage_execution_plans:
        if int(plan.end_step) <= int(decay_start):
            out.append(plan)
            continue
        if int(plan.start_step) >= int(decay_start):
            out.append(replace(plan, data_path=str(decay_data_path)))
            continue
        tokens_per_update = max(int(plan.recipe.tokens_per_update), 1)
        stable_steps = int(decay_start) - int(plan.start_step)
        split_tokens = int(plan.stage.start_tokens) + int(stable_steps) * int(
            tokens_per_update
        )
        split_tokens = min(int(split_tokens), int(plan.stage.end_tokens))
        out.append(
            replace(
                plan,
                stage=replace(plan.stage, end_tokens=int(split_tokens)),
                end_step=int(decay_start),
            )
        )
        out.append(
            replace(
                plan,
                stage=replace(plan.stage, start_tokens=int(split_tokens)),
                start_step=int(decay_start),
                data_path=str(decay_data_path),
            )
        )
    # Re-index stages so downstream bookkeeping (fresh-start tracking, logs)
    # sees unique, ordered stage indices.
    return [
        replace(plan, stage=replace(plan.stage, stage_index=int(idx)))
        for idx, plan in enumerate(out)
    ]


def build_stage_execution_plans(
    *,
    session: PretrainSession,
    args: PretrainRunConfig,
    budget: TokenBudget,
) -> None:
    session.target_tokens_per_update = int(budget.target_tokens_per_update)
    session.runtime_plan.stage_execution_plans = build_stage_execution_plan(
        stages=list(session.runtime_plan.curriculum_stages),
        target_tokens_per_microbatch=int(budget.target_tokens_per_microbatch),
        target_tokens_per_update=int(session.target_tokens_per_update),
        max_batch_size=int(args.auto_batch_size_max),
    )
    session.runtime_plan.stage_execution_plans = apply_explicit_max_steps_cap(
        stage_execution_plans=list(session.runtime_plan.stage_execution_plans),
        max_steps=int(args.max_steps),
    )
    if not session.runtime_plan.stage_execution_plans:
        raise RuntimeError("length curriculum produced no executable training stages")
    session.runtime_plan.stage_execution_plans[0] = replace(
        session.runtime_plan.stage_execution_plans[0],
        recipe=replace(
            session.runtime_plan.stage_execution_plans[0].recipe,
            gradient_checkpointing=int(args.gradient_checkpointing),
            gradient_checkpointing_exclude_first=int(
                args.gradient_checkpointing_exclude_first
            ),
            gradient_checkpointing_exclude_last=int(
                args.gradient_checkpointing_exclude_last
            ),
            loss_chunk_size=int(args.loss_chunk_size),
            machine_recipe_tuned=1,
        ),
    )
    session.runtime_plan.stage_execution_plans = apply_decay_data_stage(
        stage_execution_plans=list(session.runtime_plan.stage_execution_plans),
        args=args,
    )


def apply_entry_stage_recipe(
    *,
    session: PretrainSession,
    args: PretrainRunConfig,
) -> PretrainRunConfig:
    from ml.tasks.pretrain.run_spec_builder import build_pretrain_run_spec

    if not session.runtime_plan.stage_execution_plans:
        raise RuntimeError("session stage execution plans are not initialized")
    session.max_steps = int(session.runtime_plan.stage_execution_plans[-1].end_step)
    args = replace(
        args,
        max_steps=int(session.max_steps),
    )

    first_recipe = session.runtime_plan.stage_execution_plans[0].recipe
    session.seq_len = int(first_recipe.seq_len)
    session.runtime_state.seq_len = int(first_recipe.seq_len)
    session.runtime_state.batch_size = int(first_recipe.batch_size)
    session.runtime_state.accumulation_steps = int(first_recipe.accumulation_steps)
    run_cfg = session.project_run_config(args)
    session.spec = build_pretrain_run_spec(run_cfg)
    return run_cfg


__all__ = [
    "CurriculumStage",
    "StagePlan",
    "RuntimeRecipe",
    "TokenBudget",
    "apply_decay_data_stage",
    "apply_entry_stage_recipe",
    "apply_explicit_max_steps_cap",
    "wsd_decay_start_step",
    "allocate_weighted_token_budgets",
    "build_length_curriculum",
    "build_stage_execution_plan",
    "build_stage_execution_plans",
    "derive_stage_runtime_recipe",
    "resolve_stage_token_budget",
    "scaled_token_schedule",
]
