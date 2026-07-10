from __future__ import annotations

import math
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ml.core.common.io import write_json_atomic
from ml.training.pretrain.resources import PretrainDataIter
from ml.training.pretrain.resources import PretrainTokenizerLike
from ml.training.pretrain.grad_clip_semantics import PINNED_AGC_EPS
from ml.training.ema import ModelEMA
from ml.training.pretrain.engine.checkpoints import (
    maybe_eval_and_save_best as _maybe_eval_and_save_best,
    maybe_run_extra_evals as _maybe_run_extra_evals,
    restore_best_eval_loss as _restore_best_eval_loss,
)
from ml.training.pretrain.engine.checkpoints import (
    maybe_log_train_step as _maybe_log_train_step,
)
from ml.training.pretrain.engine.checkpoints import (
    maybe_save_interval_checkpoint as _maybe_save_interval_checkpoint,
    maybe_save_interval_weights_checkpoint as _maybe_save_interval_weights_checkpoint,
    maybe_save_model_export as _maybe_save_model_export,
)
from ml.training.pretrain.engine.checkpoints import (
    save_checkpoint_step as _save_checkpoint_step,
)
from ml.training.pretrain.engine.io import _AsyncCheckpointWriter
from ml.training.pretrain.engine.io import _MetricsLogger
from ml.training.pretrain.engine.io import _TensorBoardLogger
from ml.training.pretrain.engine.lr_schedule_state import (
    LrScheduleStateConfig,
)
from ml.training.pretrain.engine.state_support import (
    _capture_data_iter_state_or_raise,
)
from ml.training.pretrain.engine.step_runner_impl import StepRunner
from ml.training.pretrain.engine.train_step import (
    LoopAccumulators,
    LoopControllers,
    build_train_state,
    run_train_update,
)
from ml.training.pretrain.external_eval import set_current_step_on_model
from ml.training.pretrain.train_state import PretrainTrainState


@dataclass(frozen=True)
class TrainLoopResult:
    last_step: int
    train_state: dict[str, object]


@dataclass(frozen=True)
class LoopConfig:
    output_dir: str
    max_steps: int
    accumulation_steps: int
    max_grad_norm: float
    log_interval: int
    save_interval: int
    save_total_limit: int
    tokens_per_update: int
    save_weights_interval: int = 0
    save_weights_total_limit: int = 0
    grad_clip_mode: str = "norm"
    agc_clip: float = 0.0
    agc_eps: float = PINNED_AGC_EPS
    agc_exclude_bias_and_norm: int = 1
    eval_interval: int = 0
    save_best: int = 1
    token_weighted_loss: int = 0
    enable_checkpoints: int = 1
    ckpt_staging_dir: str = ""
    ema_eval: int = 0
    ema_use_for_export: int = 0
    ema_in_ckpt: int = 1
    async_checkpoint: int = 0
    async_metrics: int = 1
    stability_guard_enabled: int = 0
    stability_guard_loss_window: int = 20
    stability_guard_loss_max: float = 0.0
    stability_guard_grad_norm_max: float = 0.0
    stability_guard_grad_window: int = 100
    stability_guard_grad_spike_limit: int = 0
    stability_guard_min_step: int = 0


@dataclass(frozen=True)
class ResolvedLoopConfig:
    output_dir: str
    max_steps: int
    save_total_limit: int
    async_metrics: bool
    enable_checkpoints: bool
    async_checkpoint: bool
    ckpt_staging_dir: str | None
    eval_interval: int
    save_best: bool
    token_weighted_loss: bool
    ema_eval: bool
    ema_use_for_export: bool
    ema_in_ckpt: bool
    save_interval: int
    stability_guard_enabled: bool
    stability_guard_loss_window: int
    stability_guard_loss_max: float
    stability_guard_grad_norm_max: float
    stability_guard_grad_window: int
    stability_guard_grad_spike_limit: int
    stability_guard_min_step: int


@dataclass
class _LoopState:
    controllers: LoopControllers
    accumulators: LoopAccumulators
    best_eval_loss: float | None
    final_train_state: PretrainTrainState
    ckpt_staging_dir: str | None
    last_saved_step: int | None = None
    last_weights_saved_step: int | None = None
    last_log_t: float = 0.0
    updates_since_log: int = 0
    ema_update_s: float | None = None


@dataclass
class _LoopWriters:
    metrics: _MetricsLogger
    tb: _TensorBoardLogger
    ckpt_writer: _AsyncCheckpointWriter | None

    def close(self) -> None:
        finalize_exc: BaseException | None = None
        if self.ckpt_writer is not None:
            try:
                self.ckpt_writer.wait()
            except Exception as exc:
                if finalize_exc is None:
                    finalize_exc = exc
        try:
            self.metrics.close()
        except Exception as exc:
            if finalize_exc is None:
                finalize_exc = exc
        try:
            self.tb.close()
        except Exception as exc:
            if finalize_exc is None:
                finalize_exc = exc
        if finalize_exc is not None:
            raise finalize_exc


class _StabilityGuard:
    def __init__(self, *, cfg: ResolvedLoopConfig) -> None:
        self._cfg = cfg
        self._losses: deque[float] = deque(
            maxlen=max(int(cfg.stability_guard_loss_window), 1)
        )
        self._grad_spikes: deque[bool] = deque(
            maxlen=max(int(cfg.stability_guard_grad_window), 1)
        )

    def check(
        self,
        *,
        step: int,
        loss_value: float,
        pre_clip_grad_norm: torch.Tensor | None,
        post_clip_grad_norm: torch.Tensor | None,
        output_dir: str,
    ) -> None:
        if not bool(self._cfg.stability_guard_enabled):
            return

        loss_f = float(loss_value)
        self._losses.append(loss_f)
        pre_clip_grad_f = None
        if pre_clip_grad_norm is not None:
            pre_clip_grad_f = float(pre_clip_grad_norm.detach().item())
        post_clip_grad_f = None
        if post_clip_grad_norm is not None:
            post_clip_grad_f = float(post_clip_grad_norm.detach().item())
        observed_grad_f = (
            pre_clip_grad_f
            if pre_clip_grad_f is not None
            else post_clip_grad_f
        )
        grad_limit = float(self._cfg.stability_guard_grad_norm_max)
        self._grad_spikes.append(
            bool(
                observed_grad_f is not None
                and grad_limit > 0.0
                and observed_grad_f > grad_limit
            )
        )
        if int(step) < int(self._cfg.stability_guard_min_step):
            return

        reason = ""
        loss_window = int(self._cfg.stability_guard_loss_window)
        loss_limit = float(self._cfg.stability_guard_loss_max)
        if not math.isfinite(loss_f):
            reason = "nonfinite_loss"
        elif loss_limit > 0.0 and loss_f > loss_limit:
            reason = "loss_exceeded"
        elif (
            loss_limit > 0.0
            and len(self._losses) >= max(loss_window, 1)
            and (sum(self._losses) / float(len(self._losses))) > loss_limit
        ):
            reason = "loss_window_exceeded"

        spike_limit = int(self._cfg.stability_guard_grad_spike_limit)
        if (
            not reason
            and observed_grad_f is not None
            and not math.isfinite(observed_grad_f)
        ):
            reason = "nonfinite_grad_norm"
        if (
            not reason
            and observed_grad_f is not None
            and grad_limit > 0.0
            and observed_grad_f > grad_limit
        ):
            reason = "grad_norm_exceeded"
        if not reason and grad_limit > 0.0 and spike_limit > 0:
            if sum(1 for item in self._grad_spikes if item) >= spike_limit:
                reason = "grad_spike_limit_exceeded"

        if not reason:
            return

        incident = {
            "schema": "pretrain_stability_incident_v1",
            "reason": str(reason),
            "step": int(step),
            "loss": float(loss_f),
            "loss_window": list(self._losses),
            "loss_window_avg": float(sum(self._losses) / float(len(self._losses))),
            "loss_window_max": float(loss_limit),
            "grad_norm": observed_grad_f,
            "pre_clip_grad_norm": pre_clip_grad_f,
            "post_clip_grad_norm": post_clip_grad_f,
            "grad_norm_max": float(grad_limit),
            "grad_spike_count": int(sum(1 for item in self._grad_spikes if item)),
            "grad_spike_limit": int(spike_limit),
            "grad_window": int(self._cfg.stability_guard_grad_window),
        }
        os.makedirs(str(output_dir), exist_ok=True)
        write_json_atomic(
            os.path.join(str(output_dir), "stability_incident.json"),
            incident,
        )
        raise RuntimeError(
            f"pretrain stability guard tripped at step={int(step)} reason={reason}"
        )


def _build_loop_controllers(
    *,
    resume_train_state: PretrainTrainState | dict[str, object] | None,
    lr_schedule_state_cfg: LrScheduleStateConfig | None,
) -> tuple[LoopControllers, int, float | None]:
    resume_state = PretrainTrainState.resolve(resume_train_state)
    resume_seen_tokens = int(resume_state.seen_supervised_tokens)
    return (
        LoopControllers(
            lr_schedule_state_cfg=lr_schedule_state_cfg or LrScheduleStateConfig(),
        ),
        int(resume_seen_tokens),
        resume_state.best_eval_loss,
    )


def _initialize_loop_state(
    *,
    device: torch.device,
    resolved_cfg: ResolvedLoopConfig,
    output_dir: str,
    resume_train_state: PretrainTrainState | dict[str, object] | None,
    lr_schedule_state_cfg: LrScheduleStateConfig | None,
) -> _LoopState:
    controllers, resume_seen_tokens, best_eval_loss = _build_loop_controllers(
        resume_train_state=resume_train_state,
        lr_schedule_state_cfg=lr_schedule_state_cfg,
    )
    accumulators = LoopAccumulators.create(
        device=device,
        resume_seen_tokens=int(resume_seen_tokens),
    )
    restored_best_eval_loss = _restore_best_eval_loss(
        output_dir=str(output_dir),
        best_eval_loss=best_eval_loss,
    )
    train_state = build_train_state(
        controllers=controllers,
        seen_tokens=int(accumulators.seen_supervised_tokens_value),
        best_eval_loss=restored_best_eval_loss,
    )
    return _LoopState(
        controllers=controllers,
        accumulators=accumulators,
        best_eval_loss=restored_best_eval_loss,
        final_train_state=train_state,
        ckpt_staging_dir=resolved_cfg.ckpt_staging_dir,
        last_log_t=float(time.perf_counter()),
    )


def _build_loop_writers(
    *,
    resolved_cfg: ResolvedLoopConfig,
    args_dict: dict[str, object] | None,
    start_step: int,
    device: torch.device,
) -> _LoopWriters:
    metrics = _MetricsLogger(
        output_dir=resolved_cfg.output_dir,
        args_dict=args_dict,
        start_step=int(start_step),
        max_steps=int(resolved_cfg.max_steps),
        device=device,
        async_write=bool(resolved_cfg.async_metrics),
    )
    tb = _TensorBoardLogger(
        output_dir=resolved_cfg.output_dir,
        args_dict=args_dict,
        start_step=int(start_step),
    )
    ckpt_writer: _AsyncCheckpointWriter | None = None
    if resolved_cfg.enable_checkpoints and resolved_cfg.async_checkpoint:
        ckpt_writer = _AsyncCheckpointWriter()
    return _LoopWriters(metrics=metrics, tb=tb, ckpt_writer=ckpt_writer)


def _build_periodic_train_state(
    *,
    resolved_cfg: ResolvedLoopConfig,
    global_step: int,
    data_iter: PretrainDataIter,
    loop_state: _LoopState,
) -> PretrainTrainState | None:
    if not (
        bool(resolved_cfg.enable_checkpoints)
        and int(resolved_cfg.save_interval) > 0
        and int(global_step) % int(resolved_cfg.save_interval) == 0
    ):
        return None
    return build_train_state(
        controllers=loop_state.controllers,
        seen_tokens=int(loop_state.accumulators.seen_supervised_tokens_value),
        best_eval_loss=loop_state.best_eval_loss,
        data_iter=data_iter,
    )


def _build_eval_train_state(
    *,
    resolved_cfg: ResolvedLoopConfig,
    global_step: int,
    data_iter: PretrainDataIter,
    eval_fn: Callable[[], float] | None,
    checkpoint_train_state: PretrainTrainState | None,
    loop_state: _LoopState,
) -> PretrainTrainState | None:
    if checkpoint_train_state is not None:
        return checkpoint_train_state
    if (
        eval_fn is None
        or int(resolved_cfg.eval_interval) <= 0
        or (
            int(global_step) % int(resolved_cfg.eval_interval) != 0
            and int(global_step) != int(resolved_cfg.max_steps)
        )
    ):
        return None
    return build_train_state(
        controllers=loop_state.controllers,
        seen_tokens=int(loop_state.accumulators.seen_supervised_tokens_value),
        best_eval_loss=loop_state.best_eval_loss,
        data_iter=data_iter,
    )


def _finalize_train_state(
    *,
    resolved_cfg: ResolvedLoopConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    data_iter: PretrainDataIter,
    args_dict: dict[str, object] | None,
    ema: ModelEMA | None,
    loop_state: _LoopState,
    ckpt_writer: _AsyncCheckpointWriter | None,
) -> PretrainTrainState:
    final_train_state = loop_state.final_train_state
    if resolved_cfg.enable_checkpoints and loop_state.last_saved_step != int(
        resolved_cfg.max_steps
    ):
        final_train_state = build_train_state(
            controllers=loop_state.controllers,
            seen_tokens=int(loop_state.accumulators.seen_supervised_tokens_value),
            best_eval_loss=loop_state.best_eval_loss,
            data_iter=data_iter,
        )
        _save_checkpoint_step(
            output_dir=resolved_cfg.output_dir,
            step=int(resolved_cfg.max_steps),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args_dict,
            save_total_limit=int(resolved_cfg.save_total_limit),
            staging_dir=loop_state.ckpt_staging_dir,
            ckpt_writer=ckpt_writer,
            ema=ema,
            ema_in_ckpt=bool(resolved_cfg.ema_in_ckpt),
            train_state=final_train_state,
        )
        return final_train_state
    if final_train_state.data_iter_state is None:
        final_train_state = build_train_state(
            controllers=loop_state.controllers,
            seen_tokens=int(loop_state.accumulators.seen_supervised_tokens_value),
            best_eval_loss=loop_state.best_eval_loss,
            data_iter=data_iter,
        )
    return final_train_state


def _advance_train_loop_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    data_iter: PretrainDataIter,
    runner: StepRunner,
    device: torch.device,
    cfg: LoopConfig,
    resolved_cfg: ResolvedLoopConfig,
    step: int,
    args_dict: dict[str, object] | None,
    eval_fn: Callable[[], float] | None,
    extra_evals: list[tuple[str, int, Callable[[], float]]] | None,
    ema: ModelEMA | None,
    loop_state: _LoopState,
    writers: _LoopWriters,
    stability_guard: _StabilityGuard,
) -> int:
    update_result = run_train_update(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_iter=data_iter,
        runner=runner,
        device=device,
        cfg=cfg,
        step=int(step),
        best_eval_loss=loop_state.best_eval_loss,
        token_weighted_cfg=bool(resolved_cfg.token_weighted_loss),
        ema=ema,
        controllers=loop_state.controllers,
        accumulators=loop_state.accumulators,
        stability_check_fn=lambda loss_value, pre_clip_grad_norm, post_clip_grad_norm: stability_guard.check(
            step=int(step) + 1,
            loss_value=float(loss_value),
            pre_clip_grad_norm=pre_clip_grad_norm,
            post_clip_grad_norm=post_clip_grad_norm,
            output_dir=str(resolved_cfg.output_dir),
        ),
    )
    loop_state.final_train_state = update_result.train_state
    global_step = int(update_result.global_step)
    loop_state.updates_since_log += 1
    (
        loop_state.last_log_t,
        loop_state.updates_since_log,
        loop_state.ema_update_s,
    ) = _maybe_log_train_step(
        global_step=int(global_step),
        cfg=cfg,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_norm=update_result.grad_norm,
        token_weighted_active=bool(update_result.token_weighted_active),
        running_loss=loop_state.accumulators.running_loss,
        running_loss_sum=loop_state.accumulators.running_loss_sum,
        running_supervised_tokens=loop_state.accumulators.running_supervised_tokens,
        seen_supervised_tokens=loop_state.accumulators.seen_supervised_tokens,
        last_log_t=float(loop_state.last_log_t),
        updates_since_log=int(loop_state.updates_since_log),
        ema_update_s=loop_state.ema_update_s,
        metrics=writers.metrics,
        tb=writers.tb,
    )

    checkpoint_train_state = _build_periodic_train_state(
        resolved_cfg=resolved_cfg,
        global_step=int(global_step),
        data_iter=data_iter,
        loop_state=loop_state,
    )
    loop_state.last_saved_step = _maybe_save_interval_checkpoint(
        cfg=cfg,
        global_step=int(global_step),
        enable_checkpoints=bool(resolved_cfg.enable_checkpoints),
        ckpt_writer=writers.ckpt_writer,
        ckpt_staging_dir=loop_state.ckpt_staging_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args_dict=args_dict,
        ema=ema,
        ema_in_ckpt=bool(resolved_cfg.ema_in_ckpt),
        last_saved_step=loop_state.last_saved_step,
        train_state=checkpoint_train_state,
    )

    eval_train_state = _build_eval_train_state(
        resolved_cfg=resolved_cfg,
        global_step=int(global_step),
        data_iter=data_iter,
        eval_fn=eval_fn,
        checkpoint_train_state=checkpoint_train_state,
        loop_state=loop_state,
    )
    (
        loop_state.best_eval_loss,
        loop_state.last_saved_step,
    ) = _maybe_eval_and_save_best(
        cfg=cfg,
        global_step=int(global_step),
        eval_fn=eval_fn,
        eval_interval=int(resolved_cfg.eval_interval),
        metrics=writers.metrics,
        tb=writers.tb,
        ema=ema,
        ema_eval=bool(resolved_cfg.ema_eval),
        best_eval_loss=loop_state.best_eval_loss,
        save_best=bool(resolved_cfg.save_best),
        enable_checkpoints=bool(resolved_cfg.enable_checkpoints),
        last_saved_step=loop_state.last_saved_step,
        ckpt_writer=writers.ckpt_writer,
        ckpt_staging_dir=loop_state.ckpt_staging_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args_dict=args_dict,
        ema_in_ckpt=bool(resolved_cfg.ema_in_ckpt),
        train_state=eval_train_state,
    )

    set_current_step_on_model(model=model, step=int(global_step))
    _maybe_run_extra_evals(
        global_step=int(global_step),
        max_steps=int(resolved_cfg.max_steps),
        extra_evals=extra_evals,
        metrics=writers.metrics,
        tb=writers.tb,
        ema=ema,
        ema_eval=bool(resolved_cfg.ema_eval),
    )
    loop_state.last_weights_saved_step = _maybe_save_interval_weights_checkpoint(
        cfg=cfg,
        global_step=int(global_step),
        enable_checkpoints=bool(resolved_cfg.enable_checkpoints),
        ckpt_writer=writers.ckpt_writer,
        ckpt_staging_dir=loop_state.ckpt_staging_dir,
        model=model,
        args_dict=args_dict,
        ema=ema,
        ema_in_ckpt=bool(resolved_cfg.ema_in_ckpt),
        last_full_saved_step=loop_state.last_saved_step,
        last_weights_saved_step=loop_state.last_weights_saved_step,
    )
    return int(update_result.next_step)


def train_loop(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    data_iter: PretrainDataIter,
    runner: StepRunner,
    device: torch.device,
    cfg: LoopConfig,
    start_step: int,
    args_dict: dict[str, object] | None,
    tokenizer: PretrainTokenizerLike | None,
    safe_serialization: bool,
    export_model_artifacts_fn: Callable[..., None],
    eval_fn: Callable[[], float] | None = None,
    extra_evals: list[tuple[str, int, Callable[[], float]]] | None = None,
    ema: ModelEMA | None = None,
    resume_train_state: PretrainTrainState | dict[str, object] | None = None,
    lr_schedule_state_cfg: LrScheduleStateConfig | None = None,
) -> TrainLoopResult:
    resolved_cfg = ResolvedLoopConfig(
        output_dir=str(cfg.output_dir),
        max_steps=int(cfg.max_steps),
        save_total_limit=int(cfg.save_total_limit),
        async_metrics=False if cfg.async_metrics is None else bool(int(cfg.async_metrics) == 1),
        enable_checkpoints=False
        if cfg.enable_checkpoints is None
        else bool(int(cfg.enable_checkpoints) == 1),
        async_checkpoint=False
        if cfg.async_checkpoint is None
        else bool(int(cfg.async_checkpoint) == 1),
        ckpt_staging_dir=(
            None
            if cfg.ckpt_staging_dir is None or not str(cfg.ckpt_staging_dir).strip()
            else str(cfg.ckpt_staging_dir).strip()
        ),
        eval_interval=0 if cfg.eval_interval is None else max(int(cfg.eval_interval), 0),
        save_best=False if cfg.save_best is None else bool(int(cfg.save_best) == 1),
        token_weighted_loss=False
        if cfg.token_weighted_loss is None
        else bool(int(cfg.token_weighted_loss) == 1),
        ema_eval=False if cfg.ema_eval is None else bool(int(cfg.ema_eval) == 1),
        ema_use_for_export=False
        if cfg.ema_use_for_export is None
        else bool(int(cfg.ema_use_for_export) == 1),
        ema_in_ckpt=False if cfg.ema_in_ckpt is None else bool(int(cfg.ema_in_ckpt) == 1),
        save_interval=0 if cfg.save_interval is None else max(int(cfg.save_interval), 0),
        stability_guard_enabled=False
        if cfg.stability_guard_enabled is None
        else bool(int(cfg.stability_guard_enabled) == 1),
        stability_guard_loss_window=(
            20
            if cfg.stability_guard_loss_window is None
            else max(int(cfg.stability_guard_loss_window), 1)
        ),
        stability_guard_loss_max=(
            0.0
            if cfg.stability_guard_loss_max is None
            else float(cfg.stability_guard_loss_max)
        ),
        stability_guard_grad_norm_max=(
            0.0
            if cfg.stability_guard_grad_norm_max is None
            else float(cfg.stability_guard_grad_norm_max)
        ),
        stability_guard_grad_window=(
            100
            if cfg.stability_guard_grad_window is None
            else max(int(cfg.stability_guard_grad_window), 1)
        ),
        stability_guard_grad_spike_limit=(
            0
            if cfg.stability_guard_grad_spike_limit is None
            else max(int(cfg.stability_guard_grad_spike_limit), 0)
        ),
        stability_guard_min_step=(
            0
            if cfg.stability_guard_min_step is None
            else max(int(cfg.stability_guard_min_step), 0)
        ),
    )
    os.makedirs(resolved_cfg.output_dir, exist_ok=True)
    writers = _build_loop_writers(
        resolved_cfg=resolved_cfg,
        args_dict=args_dict,
        start_step=int(start_step),
        device=device,
    )
    try:
        _capture_data_iter_state_or_raise(
            data_iter,
            feature_name="checkpointable training data stream",
        )
        stability_guard = _StabilityGuard(cfg=resolved_cfg)
        loop_state = _initialize_loop_state(
            device=device,
            resolved_cfg=resolved_cfg,
            output_dir=resolved_cfg.output_dir,
            resume_train_state=resume_train_state,
            lr_schedule_state_cfg=lr_schedule_state_cfg,
        )

        step = int(start_step)
        while int(step) < int(resolved_cfg.max_steps):
            step = _advance_train_loop_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                data_iter=data_iter,
                runner=runner,
                device=device,
                cfg=cfg,
                resolved_cfg=resolved_cfg,
                step=int(step),
                args_dict=args_dict,
                eval_fn=eval_fn,
                extra_evals=extra_evals,
                ema=ema,
                loop_state=loop_state,
                writers=writers,
                stability_guard=stability_guard,
            )

        final_train_state = _finalize_train_state(
            resolved_cfg=resolved_cfg,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_iter=data_iter,
            args_dict=args_dict,
            ema=ema,
            loop_state=loop_state,
            ckpt_writer=writers.ckpt_writer,
        )

        saved_export = _maybe_save_model_export(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            safe_serialization=bool(safe_serialization),
            export_model_artifacts_fn=export_model_artifacts_fn,
            ema=ema,
            ema_use_for_export=bool(resolved_cfg.ema_use_for_export),
        )
        if saved_export:
            print(f"[OK] saved model export to: {cfg.output_dir}")

        writers.metrics.log_end(
            step=int(resolved_cfg.max_steps),
            max_steps=int(resolved_cfg.max_steps),
        )
        return TrainLoopResult(
            last_step=int(resolved_cfg.max_steps),
            train_state=final_train_state.to_payload(),
        )
    finally:
        writers.close()


__all__ = [
    "LoopConfig",
    "ResolvedLoopConfig",
    "TrainLoopResult",
    "train_loop",
]
