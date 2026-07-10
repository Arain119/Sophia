from __future__ import annotations

from dataclasses import replace

import torch

import ml.tasks.pretrain.observability as observability
from ml.training.pretrain.machine_runtime import PretrainMachineRuntime
from ml.tasks.pretrain.planner import StagePlan, RuntimeRecipe
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.training.pretrain.engine.lr_schedule_state import LrScheduleStateConfig
from ml.training.pretrain.engine.step_runner_impl import StepRunner
from ml.training.pretrain.model_runtime_controls import (
    require_runtime_recipe_control,
)
from ml.training.pretrain.model_setup import sync_runtime_batch_capacity
from ml.training.pretrain.resume_restore import machine_runtime_payload
from ml.training.pretrain.semantic_defaults import (
    learning_rate_for_tokens_per_update,
    rescale_learning_rate_between_token_budgets,
)
from ml.training.pretrain.batch_size_autofit import apply_gradient_checkpointing
from ml.training.pretrain.train_state import PretrainTrainState


def stage_run_args_dict(
    *,
    args: PretrainRunConfig,
    tokens_per_update: int,
    runner: StepRunner | None = None,
) -> dict[str, object]:
    run_args = args.to_payload()
    run_args["_sophia_requested_target_tokens_per_update"] = run_args.get(
        "target_tokens_per_update"
    )
    run_args["target_tokens_per_update"] = int(tokens_per_update)
    machine_runtime = PretrainMachineRuntime.from_source(args)
    backend = (
        None
        if runner is None
        else getattr(getattr(runner, "execution_plan", None), "backend", None)
    )
    if backend is not None:
        machine_runtime = replace(machine_runtime, step_execution_backend=backend)
        run_args["step_execution_backend"] = str(backend)
        run_args["_sophia_runner_execution_backend"] = str(backend)
    runtime_payload = machine_runtime_payload(
        machine_runtime=machine_runtime,
    )
    if runtime_payload:
        run_args["_sophia_machine_runtime"] = dict(runtime_payload)
    run_args["_sophia_effective_config_schema"] = "pretrain_effective_config_v1"
    return run_args


def build_lr_schedule_state_config(
    *,
    max_steps: int,
    warmup_steps: int,
    wsd_stable_ratio: float,
    lr_schedule: str,
    resume_train_state: PretrainTrainState | dict[str, object] | None = None,
) -> LrScheduleStateConfig:
    schedule_name = str(lr_schedule or "").strip().lower()
    decay_start_step = int(max(warmup_steps, 0))
    if schedule_name in {"wsd", "warmup_stable_decay"}:
        stable_steps = int(max(int(max_steps) * float(wsd_stable_ratio), 0))
        stable_steps = min(stable_steps, max(int(max_steps) - int(warmup_steps), 0))
        decay_start_step = min(int(warmup_steps) + int(stable_steps), int(max_steps))
    state = PretrainTrainState.resolve(resume_train_state)
    if state.has_serialized_state():
        saved_lr_decay_start_step = (
            int(decay_start_step)
            if state.lr_decay_start_step is None
            else int(state.lr_decay_start_step)
        )
        return LrScheduleStateConfig(
            lr_decay_start_step=int(saved_lr_decay_start_step),
        )
    return LrScheduleStateConfig(lr_decay_start_step=int(decay_start_step))


def _rescale_stage_learning_rate(
    *,
    state: PretrainRuntimeState,
    stage_recipe: RuntimeRecipe,
    stage_optimizer: torch.optim.Optimizer | None,
    stage_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> None:
    current_tokens_per_update = (
        int(state.seq_len)
        * max(int(state.batch_size), 1)
        * max(int(state.accumulation_steps), 1)
    )
    next_tokens_per_update = max(int(stage_recipe.tokens_per_update), 1)
    if (
        current_tokens_per_update <= 0
        or current_tokens_per_update == next_tokens_per_update
    ):
        return
    current_runtime_lr = learning_rate_for_tokens_per_update(
        semantic_learning_rate=float(state.learning_rate),
        tokens_per_update=int(current_tokens_per_update),
        target_tokens_per_update=int(state.target_tokens_per_update),
    )
    next_runtime_lr = learning_rate_for_tokens_per_update(
        semantic_learning_rate=float(state.learning_rate),
        tokens_per_update=int(next_tokens_per_update),
        target_tokens_per_update=int(state.target_tokens_per_update),
    )
    scaled_learning_rate = rescale_learning_rate_between_token_budgets(
        learning_rate=float(current_runtime_lr),
        from_tokens_per_update=int(current_tokens_per_update),
        to_tokens_per_update=int(next_tokens_per_update),
    )
    scale = float(scaled_learning_rate) / float(max(float(current_runtime_lr), 1e-30))
    if stage_optimizer is not None and hasattr(stage_optimizer, "param_groups"):
        for group in stage_optimizer.param_groups:
            if "lr" in group:
                group["lr"] = float(group.get("lr", 0.0) or 0.0) * float(scale)
            if "initial_lr" in group:
                group["initial_lr"] = float(group.get("initial_lr", 0.0) or 0.0) * float(
                    scale
                )
    if stage_scheduler is not None:
        base_lrs = getattr(stage_scheduler, "base_lrs", None)
        if isinstance(base_lrs, list):
            stage_scheduler.base_lrs = [
                float(lr) * float(scale) for lr in base_lrs
            ]
        last_lrs = getattr(stage_scheduler, "_last_lr", None)
        if isinstance(last_lrs, list):
            stage_scheduler._last_lr = [
                float(lr) * float(scale) for lr in last_lrs
            ]
    print(
        f"{observability.log_tag()} stage lr derived | "
        f"tokens/update={int(current_tokens_per_update)}->{int(next_tokens_per_update)} | "
        f"scale={float(scale):.3f} | lr_runtime={float(current_runtime_lr):.3e}->{float(next_runtime_lr):.3e}",
        flush=True,
    )


def apply_stage_recipe(
    *,
    args: PretrainRunConfig,
    model: torch.nn.Module,
    recipe: RuntimeRecipe,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    runtime_state: PretrainRuntimeState | None = None,
) -> PretrainRunConfig:
    state = (
        runtime_state
        if runtime_state is not None
        else PretrainRuntimeState.from_config(args)
    )
    _rescale_stage_learning_rate(
        state=state,
        stage_recipe=recipe,
        stage_optimizer=optimizer,
        stage_scheduler=scheduler,
    )

    state.seq_len = int(recipe.seq_len)
    state.batch_size = int(recipe.batch_size)
    state.accumulation_steps = int(recipe.accumulation_steps)
    state.gradient_checkpointing = int(recipe.gradient_checkpointing)
    state.gradient_checkpointing_exclude_first = int(
        recipe.gradient_checkpointing_exclude_first
    )
    state.gradient_checkpointing_exclude_last = int(
        recipe.gradient_checkpointing_exclude_last
    )
    state.loss_chunk_size = int(recipe.loss_chunk_size)
    run_cfg = state.project_run_config(args)
    control = require_runtime_recipe_control(model)
    control.apply_runtime_recipe_knobs(
        loss_chunk_size=int(recipe.loss_chunk_size),
        gradient_checkpointing_exclude_first=int(
            recipe.gradient_checkpointing_exclude_first
        ),
        gradient_checkpointing_exclude_last=int(
            recipe.gradient_checkpointing_exclude_last
        ),
    )
    if hasattr(model, "gradient_checkpointing_enable") and hasattr(
        model, "gradient_checkpointing_disable"
    ):
        apply_gradient_checkpointing(
            model, enabled=bool(int(recipe.gradient_checkpointing) == 1)
        )
    control.ensure_runtime_max_seq_len(int(recipe.seq_len))
    sync_runtime_batch_capacity(model=model, args=run_cfg)
    return run_cfg


def seed_active_stage_plan_recipe(
    *,
    args: PretrainRunConfig,
    stage_plans: list[StagePlan],
    current_step: int,
) -> list[StagePlan]:
    if not stage_plans:
        return stage_plans
    active_stage_idx = 0
    for idx, plan in enumerate(stage_plans):
        if int(plan.end_step) > int(current_step):
            active_stage_idx = int(idx)
            break
    if int(stage_plans[active_stage_idx].recipe.machine_recipe_tuned) == 1:
        return stage_plans
    stage_plans[active_stage_idx] = replace(
        stage_plans[active_stage_idx],
        recipe=replace(
            stage_plans[active_stage_idx].recipe,
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
    return stage_plans


def realize_stage_plan_machine_recipe(
    *,
    args: PretrainRunConfig,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: str,
    vocab_size: int,
    base_dtype: torch.dtype,
    plan: StagePlan,
) -> StagePlan:
    del model, device, output_dir, vocab_size, base_dtype
    if int(plan.recipe.machine_recipe_tuned) == 1:
        return plan
    return replace(
        plan,
        recipe=replace(
            plan.recipe,
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


__all__ = [
    "apply_stage_recipe",
    "build_lr_schedule_state_config",
    "realize_stage_plan_machine_recipe",
    "seed_active_stage_plan_recipe",
    "stage_run_args_dict",
]
