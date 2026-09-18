from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ml.core.common.io import write_json_atomic
from ml.training.pretrain.resources import PretrainDataIter
from ml.training.pretrain.resources import PretrainTokenizerLike
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
    maybe_save_model_export as _maybe_save_model_export,
)
from ml.training.pretrain.engine.checkpoints import (
    save_checkpoint_step as _save_checkpoint_step,
)
from ml.training.pretrain.engine.io import _AsyncCheckpointWriter
from ml.training.pretrain.engine.io import _MetricsLogger
from ml.training.pretrain.engine.io import _TensorBoardLogger
from ml.training.pretrain.engine.state_support import (
    _capture_data_iter_state_or_raise,
)
from ml.training.pretrain.engine.step_runner_impl import StepRunner
from ml.training.pretrain.engine.train_step import (
    LoopAccumulators,
    build_train_state,
    check_finite_update_state,
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
    log_interval: int
    save_interval: int
    save_total_limit: int
    tokens_per_update: int
    eval_interval: int = 0
    save_best: int = 1
    token_weighted_loss: int = 0
    enable_checkpoints: int = 1
    ckpt_staging_dir: str = ""
    async_checkpoint: int = 0
    async_metrics: int = 1


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
    save_interval: int


@dataclass
class _LoopState:
    accumulators: LoopAccumulators
    best_eval_loss: float | None
    final_train_state: PretrainTrainState
    ckpt_staging_dir: str | None
    last_saved_step: int | None = None
    last_log_t: float = 0.0
    updates_since_log: int = 0
    smoothed_update_s: float | None = None


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


class _FiniteGuard:
    def check(
        self,
        *,
        step: int,
        loss_value: float,
        grad_norm: torch.Tensor | None,
        output_dir: str,
    ) -> None:
        loss_f = float(loss_value)
        grad_f = None if grad_norm is None else float(grad_norm.detach().item())
        reason = ""
        if not torch.isfinite(torch.tensor(loss_f)):
            reason = "nonfinite_loss"
        if (
            not reason
            and grad_f is not None
            and not torch.isfinite(torch.tensor(grad_f))
        ):
            reason = "nonfinite_grad_norm"

        if not reason:
            return

        incident = {
            "schema": "pretrain_stability_incident_v1",
            "reason": str(reason),
            "step": int(step),
            "loss": float(loss_f),
            "grad_norm": grad_f,
        }
        os.makedirs(str(output_dir), exist_ok=True)
        write_json_atomic(
            os.path.join(str(output_dir), "stability_incident.json"),
            incident,
        )
        raise RuntimeError(
            f"pretrain stability guard tripped at step={int(step)} reason={reason}"
        )

    def check_update_state(
        self,
        *,
        step: int,
        loss_value: float,
        grad_norm: torch.Tensor | None,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        output_dir: str,
    ) -> None:
        try:
            check_finite_update_state(model=model, optimizer=optimizer)
        except RuntimeError as exc:
            loss_f = float(loss_value)
            grad_f = None if grad_norm is None else float(grad_norm.detach().item())
            incident = {
                "schema": "pretrain_stability_incident_v1",
                "reason": "nonfinite_parameter_or_optimizer_state",
                "step": int(step),
                "loss": loss_f,
                "grad_norm": grad_f,
                "detail": str(exc),
            }
            os.makedirs(str(output_dir), exist_ok=True)
            write_json_atomic(
                os.path.join(str(output_dir), "stability_incident.json"),
                incident,
            )
            raise RuntimeError(
                "pretrain stability guard tripped at "
                f"step={int(step)} reason=nonfinite_parameter_or_optimizer_state"
            ) from exc


def _restore_loop_progress(
    resume_train_state: PretrainTrainState | dict[str, object] | None,
) -> tuple[int, float | None]:
    resume_state = PretrainTrainState.resolve(resume_train_state)
    return int(resume_state.seen_supervised_tokens), resume_state.best_eval_loss


def _initialize_loop_state(
    *,
    device: torch.device,
    resolved_cfg: ResolvedLoopConfig,
    output_dir: str,
    resume_train_state: PretrainTrainState | dict[str, object] | None,
) -> _LoopState:
    resume_seen_tokens, best_eval_loss = _restore_loop_progress(resume_train_state)
    accumulators = LoopAccumulators.create(
        device=device,
        resume_seen_tokens=int(resume_seen_tokens),
    )
    restored_best_eval_loss = _restore_best_eval_loss(
        output_dir=str(output_dir),
        best_eval_loss=best_eval_loss,
    )
    train_state = build_train_state(
        seen_tokens=int(accumulators.seen_supervised_tokens_value),
        best_eval_loss=restored_best_eval_loss,
    )
    return _LoopState(
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
    loop_state: _LoopState,
    ckpt_writer: _AsyncCheckpointWriter | None,
) -> PretrainTrainState:
    final_train_state = loop_state.final_train_state
    if resolved_cfg.enable_checkpoints and loop_state.last_saved_step != int(
        resolved_cfg.max_steps
    ):
        final_train_state = build_train_state(
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
            train_state=final_train_state,
        )
        return final_train_state
    if final_train_state.data_iter_state is None:
        final_train_state = build_train_state(
            seen_tokens=int(loop_state.accumulators.seen_supervised_tokens_value),
            best_eval_loss=loop_state.best_eval_loss,
            data_iter=data_iter,
        )
    return final_train_state


def _log_attention_logit_metrics(
    *,
    global_step: int,
    runner: StepRunner,
    metrics: _MetricsLogger,
    tb: _TensorBoardLogger,
) -> None:
    payload = runner.attention_logit_metrics()
    if payload is None:
        return
    layer_indices, maxima = payload
    if maxima.ndim != 2 or int(maxima.size(0)) != len(layer_indices):
        raise RuntimeError("MLA attention logit telemetry shape mismatch")
    rows = maxima.detach().float().cpu().tolist()
    for layer_index, row in zip(layer_indices, rows, strict=True):
        for head_index, value in enumerate(row):
            name = f"attention/mla_layer_{layer_index}/head_{head_index}_max_logit"
            metrics.log_scalar(step=int(global_step), name=name, value=float(value))
            tb.log_scalar(step=int(global_step), name=name, value=float(value))


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
    loop_state: _LoopState,
    writers: _LoopWriters,
    stability_guard: _FiniteGuard,
) -> int:
    if writers.ckpt_writer is not None:
        writers.ckpt_writer.raise_if_failed()
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
        accumulators=loop_state.accumulators,
        stability_check_fn=lambda loss_value, grad_norm: stability_guard.check(
            step=int(step) + 1,
            loss_value=float(loss_value),
            grad_norm=grad_norm,
            output_dir=str(resolved_cfg.output_dir),
        ),
        post_update_stability_check_fn=lambda loss_value, grad_norm: stability_guard.check_update_state(
            step=int(step) + 1,
            loss_value=float(loss_value),
            grad_norm=grad_norm,
            model=model,
            optimizer=optimizer,
            output_dir=str(resolved_cfg.output_dir),
        ),
    )
    loop_state.final_train_state = update_result.train_state
    global_step = int(update_result.global_step)
    loop_state.updates_since_log += 1
    (
        loop_state.last_log_t,
        loop_state.updates_since_log,
        loop_state.smoothed_update_s,
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
        smoothed_update_s=loop_state.smoothed_update_s,
        metrics=writers.metrics,
        tb=writers.tb,
    )
    _log_attention_logit_metrics(
        global_step=int(global_step),
        runner=runner,
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
        train_state=eval_train_state,
    )

    set_current_step_on_model(model=model, step=int(global_step))
    _maybe_run_extra_evals(
        global_step=int(global_step),
        max_steps=int(resolved_cfg.max_steps),
        extra_evals=extra_evals,
        metrics=writers.metrics,
        tb=writers.tb,
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
    resume_train_state: PretrainTrainState | dict[str, object] | None = None,
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
        save_interval=0 if cfg.save_interval is None else max(int(cfg.save_interval), 0),
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
        stability_guard = _FiniteGuard()
        loop_state = _initialize_loop_state(
            device=device,
            resolved_cfg=resolved_cfg,
            output_dir=resolved_cfg.output_dir,
            resume_train_state=resume_train_state,
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
            loop_state=loop_state,
            ckpt_writer=writers.ckpt_writer,
        )

        if writers.ckpt_writer is not None:
            writers.ckpt_writer.wait()

        saved_export = _maybe_save_model_export(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            safe_serialization=bool(safe_serialization),
            export_model_artifacts_fn=export_model_artifacts_fn,
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
