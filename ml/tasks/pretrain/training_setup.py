from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ml.errors import SophiaUsageError
from ml.tasks.pretrain.deps import PretrainDeps
from ml.tasks.pretrain.planner import (
    CurriculumStage,
    StagePlan,
    RuntimeRecipe,
)
from ml.tasks.pretrain.session import ensure_pretrain_session
from ml.training.ema import ModelEMA
from ml.training.pretrain.semantic_defaults import (
    learning_rate_for_tokens_per_update,
)
from ml.training.pretrain.train_state import PretrainTrainState

if TYPE_CHECKING:
    from ml.tasks.pretrain.pipeline import PretrainPipeline
    from ml.tasks.pretrain.session import PretrainSession
    from ml.training.pretrain.resume_loader import PretrainResumeState


@dataclass(frozen=True)
class PretrainLoopSetup:
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    ema: ModelEMA | None
    current_step: int
    train_state: PretrainTrainState
    stage_plans: list[StagePlan]


def ensure_stage_execution_plans(
    pipeline: PretrainPipeline,
) -> list[StagePlan]:
    session = ensure_pretrain_session(pipeline)
    if session.runtime_plan.stage_execution_plans:
        return list(session.runtime_plan.stage_execution_plans)
    base_recipe = RuntimeRecipe(
        seq_len=int(session.seq_len),
        batch_size=int(session.runtime_state.batch_size),
        accumulation_steps=int(session.runtime_state.accumulation_steps),
        tokens_per_update=(
            int(session.seq_len)
            * max(int(session.runtime_state.batch_size), 1)
            * int(session.runtime_state.accumulation_steps)
        ),
    )
    base_stage = CurriculumStage(
        stage_index=0,
        seq_len=int(session.seq_len),
        start_tokens=0,
        end_tokens=max(
            int(session.seq_len)
            * max(int(session.runtime_state.batch_size), 1)
            * int(session.runtime_state.accumulation_steps)
            * max(int(session.max_steps), 1),
            1,
        ),
    )
    return [
        StagePlan(
            stage=base_stage,
            recipe=base_recipe,
            start_step=0,
            end_step=int(session.runtime_state.max_steps),
        )
    ]


def build_pretrain_loop_setup(
    pipeline: PretrainPipeline,
    *,
    deps: PretrainDeps,
) -> PretrainLoopSetup:
    session = ensure_pretrain_session(pipeline)
    resume = session.resume
    if resume is None or session.model is None:
        raise RuntimeError("pipeline is missing resume/model state")

    optimizer = resume.optimizer
    if optimizer is None:
        tokens_per_update = (
            int(session.seq_len)
            * max(int(session.runtime_state.batch_size), 1)
            * max(int(session.runtime_state.accumulation_steps), 1)
        )
        optimizer = deps.create_optimizer(
            session.model,
            lr=float(
                learning_rate_for_tokens_per_update(
                    semantic_learning_rate=float(session.runtime_state.learning_rate),
                    tokens_per_update=int(tokens_per_update),
                    target_tokens_per_update=int(
                        session.runtime_state.target_tokens_per_update
                    ),
                )
            ),
            weight_decay=float(pipeline.args.weight_decay),
            betas=(float(pipeline.args.beta1), float(pipeline.args.beta2)),
            eps=float(pipeline.args.adam_eps),
            layerwise_lr_decay=float(pipeline.args.layerwise_lr_decay),
            embedding_lr_scale=float(pipeline.args.embedding_lr_scale),
            muon_ns_steps=int(pipeline.args.muon_ns_steps),
            muon_target_rms=(
                None
                if pipeline.args.muon_target_rms is None
                else float(pipeline.args.muon_target_rms)
            ),
        )
        print("[OPT] optimizer=torch_muon_hybrid", flush=True)

    scheduler = deps.build_lr_scheduler(
        optimizer=optimizer,
        max_steps=int(session.runtime_state.max_steps),
        warmup_steps=int(pipeline.args.warmup_steps),
        warmup_ratio=float(pipeline.args.warmup_ratio),
        min_lr_ratio=float(pipeline.args.min_lr_ratio),
        schedule=str(pipeline.args.lr_schedule),
        wsd_stable_ratio=float(pipeline.args.wsd_stable_ratio),
        wsd_decay_style=str(pipeline.args.wsd_decay_style),
        resume_state=resume.resume_scheduler_state,
    )
    if session.resume_path is not None and resume.resume_scheduler_state is None:
        print(
            "[WARN] resume checkpoint is missing scheduler state; LR schedule may be incorrect."
        )

    current_step = int(resume.start_step)
    return PretrainLoopSetup(
        optimizer=optimizer,
        scheduler=scheduler,
        ema=_build_ema(session=session, resume=resume, deps=deps),
        current_step=int(current_step),
        train_state=PretrainTrainState.resolve(resume.resume_train_state),
        stage_plans=deps.seed_active_stage_plan_recipe(
            args=pipeline.args,
            stage_plans=ensure_stage_execution_plans(pipeline),
            current_step=int(current_step),
        ),
    )


def _build_ema(
    *,
    session: PretrainSession,
    resume: PretrainResumeState,
    deps: PretrainDeps,
) -> ModelEMA | None:
    resume_ema_state = None if resume.ckpt is None else getattr(resume.ckpt, "ema", None)
    ema_decay = float(session.runtime_state.ema_decay)
    if not 0.0 < ema_decay < 1.0:
        return None
    ema = deps.model_ema_cls(
        session.model,
        decay=float(ema_decay),
        update_interval=max(int(session.runtime_state.ema_update_interval), 1),
    )
    if resume_ema_state is not None:
        try:
            ema.load_state_dict(resume_ema_state)
        except Exception as exc:
            raise SophiaUsageError(f"[ERR] failed to load EMA state: {exc}") from exc
        print("[EMA] resumed", flush=True)
    elif session.resume_path is not None:
        print(
            "[EMA] enabled but checkpoint has no EMA state; starting fresh.",
            flush=True,
        )
    print(
        f"[EMA] enabled | decay={float(ema_decay):.6f} "
        f"update_interval={max(int(session.runtime_state.ema_update_interval), 1)}",
        flush=True,
    )
    return ema


__all__ = [
    "PretrainLoopSetup",
    "build_pretrain_loop_setup",
    "ensure_stage_execution_plans",
]
