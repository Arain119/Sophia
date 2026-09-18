from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ml.errors import SophiaUsageError
from ml.tasks.pretrain.deps import PretrainDeps
from ml.tasks.pretrain.session import ensure_pretrain_session
from ml.training.pretrain.optimizer import TORCH_MUON_HYBRID
from ml.training.pretrain.train_state import PretrainTrainState

if TYPE_CHECKING:
    from ml.tasks.pretrain.pipeline import PretrainPipeline


@dataclass(frozen=True)
class PretrainLoopSetup:
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    current_step: int
    train_state: PretrainTrainState


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
        optimizer_lr = float(session.runtime_state.learning_rate)
        optimizer_kind = str(pipeline.args.optimizer_kind or TORCH_MUON_HYBRID)
        common_optimizer_kwargs = {
            "lr": float(optimizer_lr),
            "weight_decay": float(pipeline.args.weight_decay),
            "betas": (float(pipeline.args.beta1), float(pipeline.args.beta2)),
            "eps": float(pipeline.args.adam_eps),
        }
        if optimizer_kind != TORCH_MUON_HYBRID:
            raise SophiaUsageError(
                f"unsupported pretrain optimizer_kind={optimizer_kind!r}"
            )
        optimizer = deps.create_optimizer(
            session.model,
            **common_optimizer_kwargs,
            muon_ns_steps=int(pipeline.args.muon_ns_steps),
        )
        print(f"[OPT] optimizer={optimizer_kind}", flush=True)

    set_runner_optimizer = getattr(session.runner, "set_optimizer", None)
    if callable(set_runner_optimizer):
        set_runner_optimizer(optimizer)

    run_max_steps = int(session.runtime_state.max_steps)
    schedule_horizon_steps = int(
        pipeline.args._sophia_lr_schedule_horizon_steps or run_max_steps
    )
    if schedule_horizon_steps < run_max_steps:
        raise SophiaUsageError(
            "LR schedule horizon cannot end before the training run: "
            f"horizon={schedule_horizon_steps} run_max_steps={run_max_steps}"
        )
    scheduler = deps.build_lr_scheduler(
        optimizer=optimizer,
        max_steps=schedule_horizon_steps,
        warmup_steps=int(pipeline.args.warmup_steps),
        warmup_ratio=float(pipeline.args.warmup_ratio),
        min_lr_ratio=float(pipeline.args.min_lr_ratio),
        schedule=str(pipeline.args.lr_schedule),
        resume_state=resume.resume_scheduler_state,
        resume_step=(
            int(resume.start_step) if resume.resume_scheduler_state is not None else None
        ),
    )
    if session.resume_path is not None and resume.resume_scheduler_state is None:
        raise SophiaUsageError(
            "[ERR] resume checkpoint is missing required scheduler state."
        )

    current_step = int(resume.start_step)
    return PretrainLoopSetup(
        optimizer=optimizer,
        scheduler=scheduler,
        current_step=int(current_step),
        train_state=PretrainTrainState.resolve(resume.resume_train_state),
    )


__all__ = [
    "PretrainLoopSetup",
    "build_pretrain_loop_setup",
]
