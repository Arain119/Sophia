from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import torch

from ml.core.common.rng import capture_rng_state
from ml.core.engine.checkpointing import (
    build_checkpoint_state,
    save_checkpoint,
    save_checkpoint_state,
)
from ml.integrations.export.artifacts import export_model_artifacts
from ml.training.pretrain.engine.io import (
    _AsyncCheckpointWriter,
    _copy_best_checkpoint,
    _fmt_duration_s,
    _MetricsLogger,
    _read_best_metric_value,
    _sync,
    _TensorBoardLogger,
    _write_best_metric,
)
from ml.training.pretrain.resources import PretrainTokenizerLike
from ml.training.pretrain.train_state import PretrainTrainState

if TYPE_CHECKING:
    from ml.training.ema import ModelEMA


class CheckpointLoopConfig(Protocol):
    output_dir: str
    save_interval: int
    save_total_limit: int
    save_weights_interval: int
    save_weights_total_limit: int


class CheckpointMetricConfig(CheckpointLoopConfig):
    max_steps: int
    log_interval: int
    tokens_per_update: int


def _tree_to_cpu(obj: object) -> object:
    if torch.is_tensor(obj):
        tensor = obj.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.to(device="cpu")
        return tensor
    if isinstance(obj, dict):
        return {key: _tree_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_tree_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_tree_to_cpu(value) for value in obj)
    return obj


def save_checkpoint_step(
    *,
    output_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    args: dict[str, object] | None,
    save_total_limit: int,
    staging_dir: str | None,
    ckpt_writer: _AsyncCheckpointWriter | None,
    ema: ModelEMA | None,
    ema_in_ckpt: bool,
    train_state: PretrainTrainState | dict[str, object] | None = None,
) -> None:
    serialized_train_state = PretrainTrainState.resolve(train_state).to_payload()
    if not serialized_train_state:
        serialized_train_state = None
    if ckpt_writer is not None:
        state = build_checkpoint_state(
            step=int(step),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            rng=capture_rng_state(),
            ema=ema.state_dict() if (ema is not None and ema_in_ckpt) else None,
            train_state=serialized_train_state,
        )
        ckpt_writer.save(
            output_dir=str(output_dir),
            step=int(step),
            state=_tree_to_cpu(state),  # type: ignore[arg-type]
            save_total_limit=int(save_total_limit),
            staging_dir=staging_dir,
            prefix="ckpt_step",
        )
        return

    save_checkpoint(
        output_dir=str(output_dir),
        step=int(step),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        rng=capture_rng_state(),
        ema=ema.state_dict() if (ema is not None and ema_in_ckpt) else None,
        train_state=serialized_train_state,
        save_total_limit=int(save_total_limit),
        staging_dir=staging_dir,
    )


def save_weights_checkpoint_step(
    *,
    output_dir: str,
    step: int,
    model: torch.nn.Module,
    args: dict[str, object] | None,
    save_total_limit: int,
    staging_dir: str | None,
    ckpt_writer: _AsyncCheckpointWriter | None,
    ema: ModelEMA | None,
    ema_in_ckpt: bool,
) -> None:
    state = {
        "kind": "weights_only",
        "step": int(step),
        "model": model.state_dict(),
        "args": args,
        "ema": ema.state_dict() if (ema is not None and ema_in_ckpt) else None,
    }

    if ckpt_writer is not None:
        ckpt_writer.save(
            output_dir=str(output_dir),
            step=int(step),
            state=_tree_to_cpu(state),  # type: ignore[arg-type]
            save_total_limit=int(save_total_limit),
            staging_dir=staging_dir,
            prefix="weights_step",
        )
        return

    save_checkpoint_state(
        output_dir=str(output_dir),
        step=int(step),
        state=state,
        save_total_limit=int(save_total_limit),
        staging_dir=staging_dir,
        prefix="weights_step",
    )


def maybe_save_interval_checkpoint(
    *,
    cfg: CheckpointLoopConfig,
    global_step: int,
    enable_checkpoints: bool,
    ckpt_writer: _AsyncCheckpointWriter | None,
    ckpt_staging_dir: str | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    args_dict: dict[str, object] | None,
    ema: ModelEMA | None,
    ema_in_ckpt: bool,
    last_saved_step: int | None,
    train_state: PretrainTrainState | dict[str, object] | None = None,
) -> int | None:
    save_interval = int(cfg.save_interval or 0)
    if (not enable_checkpoints) or save_interval <= 0:
        return last_saved_step
    if int(global_step) % int(save_interval) != 0:
        return last_saved_step

    save_checkpoint_step(
        output_dir=str(cfg.output_dir),
        step=int(global_step),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args_dict,
        save_total_limit=int(cfg.save_total_limit),
        staging_dir=ckpt_staging_dir,
        ckpt_writer=ckpt_writer,
        ema=ema,
        ema_in_ckpt=bool(ema_in_ckpt),
        train_state=train_state,
    )
    return int(global_step)


def maybe_save_interval_weights_checkpoint(
    *,
    cfg: CheckpointLoopConfig,
    global_step: int,
    enable_checkpoints: bool,
    ckpt_writer: _AsyncCheckpointWriter | None,
    ckpt_staging_dir: str | None,
    model: torch.nn.Module,
    args_dict: dict[str, object] | None,
    ema: ModelEMA | None,
    ema_in_ckpt: bool,
    last_full_saved_step: int | None,
    last_weights_saved_step: int | None,
) -> int | None:
    save_interval = int(cfg.save_weights_interval or 0)
    if (not enable_checkpoints) or save_interval <= 0:
        return last_weights_saved_step
    if int(global_step) % int(save_interval) != 0:
        return last_weights_saved_step
    if last_full_saved_step == int(global_step):
        return last_weights_saved_step

    limit = int(cfg.save_weights_total_limit or 0)
    if limit <= 0:
        limit = int(cfg.save_total_limit or 0)

    save_weights_checkpoint_step(
        output_dir=str(cfg.output_dir),
        step=int(global_step),
        model=model,
        args=args_dict,
        save_total_limit=int(limit),
        staging_dir=ckpt_staging_dir,
        ckpt_writer=ckpt_writer,
        ema=ema,
        ema_in_ckpt=bool(ema_in_ckpt),
    )
    return int(global_step)


def maybe_save_model_export(
    *,
    model: torch.nn.Module,
    tokenizer: PretrainTokenizerLike | None,
    cfg: CheckpointLoopConfig,
    safe_serialization: bool,
    export_model_artifacts_fn: Callable[..., None] = export_model_artifacts,
    ema: ModelEMA | None,
    ema_use_for_export: bool,
) -> bool:
    if tokenizer is None:
        return False
    export_model_artifacts_fn(
        model=model,
        tokenizer=tokenizer,
        output_dir=str(cfg.output_dir),
        safe_serialization=bool(int(safe_serialization)),
        ema=(ema if ema is not None and ema_use_for_export else None),
    )
    return True


def maybe_eval_and_save_best(
    *,
    cfg: CheckpointMetricConfig,
    global_step: int,
    eval_fn: Callable[[], float] | None,
    eval_interval: int,
    metrics: _MetricsLogger,
    tb: _TensorBoardLogger,
    ema: ModelEMA | None,
    ema_eval: bool,
    best_eval_loss: float | None,
    save_best: bool,
    enable_checkpoints: bool,
    last_saved_step: int | None,
    ckpt_writer: _AsyncCheckpointWriter | None,
    ckpt_staging_dir: str | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    args_dict: dict[str, object] | None,
    ema_in_ckpt: bool,
    train_state: PretrainTrainState | dict[str, object] | None = None,
) -> tuple[float | None, int | None]:
    max_steps = int(cfg.max_steps)
    if eval_fn is None or eval_interval <= 0:
        return best_eval_loss, last_saved_step
    if (int(global_step) % int(eval_interval) != 0) and (int(global_step) != int(max_steps)):
        return best_eval_loss, last_saved_step

    if ema is not None and ema_eval:
        with ema.apply_to_model():
            val_loss = float(eval_fn())
    else:
        val_loss = float(eval_fn())

    print(
        f"[EVAL] step={int(global_step)}/{int(max_steps)} val_loss={float(val_loss):.4f}",
        flush=True,
    )
    improved = (best_eval_loss is None) or (float(val_loss) < float(best_eval_loss))
    metrics.log_eval(
        step=int(global_step),
        max_steps=int(max_steps),
        val_loss=float(val_loss),
        improved=bool(improved),
    )
    tb.log_eval(step=int(global_step), val_loss=float(val_loss))
    if not improved:
        return best_eval_loss, last_saved_step

    best_eval_loss = float(val_loss)
    print(f"[BEST] step={int(global_step)} val_loss={float(val_loss):.4f}", flush=True)

    if save_best and enable_checkpoints:
        if last_saved_step != int(global_step):
            save_checkpoint_step(
                output_dir=str(cfg.output_dir),
                step=int(global_step),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args_dict,
                save_total_limit=int(cfg.save_total_limit),
                staging_dir=ckpt_staging_dir,
                ckpt_writer=ckpt_writer,
                ema=ema,
                ema_in_ckpt=bool(ema_in_ckpt),
                train_state=train_state,
            )
            last_saved_step = int(global_step)
        if ckpt_writer is not None:
            ckpt_writer.wait()
        _copy_best_checkpoint(output_dir=str(cfg.output_dir), step=int(global_step))

    _write_best_metric(
        output_dir=str(cfg.output_dir),
        step=int(global_step),
        metric_name="val_loss",
        metric_value=float(val_loss),
    )
    return best_eval_loss, last_saved_step


def maybe_run_extra_evals(
    *,
    global_step: int,
    max_steps: int,
    extra_evals: list[tuple[str, int, Callable[[], float]]] | None,
    metrics: _MetricsLogger,
    tb: _TensorBoardLogger,
    ema: ModelEMA | None,
    ema_eval: bool,
) -> None:
    if not extra_evals:
        return

    for name, interval, fn in list(extra_evals):
        try:
            itv = int(interval)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"extra eval interval for {str(name)!r} must be an integer"
            ) from exc
        if itv <= 0:
            raise ValueError(
                f"extra eval interval for {str(name)!r} must be > 0, got {itv}"
            )
        if (int(global_step) % int(itv) != 0) and (int(global_step) != int(max_steps)):
            continue

        try:
            if ema is not None and bool(ema_eval):
                with ema.apply_to_model():
                    val = float(fn())
            else:
                val = float(fn())
        except Exception as exc:
            raise RuntimeError(
                "extra eval failed at "
                f"step={int(global_step)} name={str(name)!r}: {type(exc).__name__}: {exc}"
            ) from exc
        if not math.isfinite(val):
            raise ValueError(
                f"extra eval {str(name)!r} returned non-finite value {val!r}"
            )

        print(
            f"[METRIC] step={int(global_step)}/{int(max_steps)} {str(name)}={float(val):.4f}",
            flush=True,
        )
        metrics.log_scalar(step=int(global_step), name=str(name), value=float(val))
        tb.log_scalar(step=int(global_step), name=str(name), value=float(val))


def restore_best_eval_loss(
    *,
    output_dir: str,
    best_eval_loss: float | None,
) -> float | None:
    persisted_best_eval_loss = _read_best_metric_value(
        output_dir=str(output_dir),
        metric_name="val_loss",
    )
    if best_eval_loss is None:
        return persisted_best_eval_loss
    if (
        persisted_best_eval_loss is not None
        and float(persisted_best_eval_loss) < float(best_eval_loss)
    ):
        return float(persisted_best_eval_loss)
    return best_eval_loss


def _optimizer_diagnostics(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    diagnostics_fn = getattr(optimizer, "diagnostics", None)
    if not callable(diagnostics_fn):
        return {}
    try:
        payload = diagnostics_fn()
    except Exception as exc:
        raise RuntimeError("optimizer diagnostics failed") from exc
    if not isinstance(payload, dict):
        raise TypeError("optimizer diagnostics must return a dictionary")
    diagnostics: dict[str, float] = {}
    for key, value in payload.items():
        try:
            diagnostics[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"optimizer diagnostic {str(key)!r} must be numeric"
            ) from exc
    return diagnostics


def maybe_log_train_step(
    *,
    global_step: int,
    cfg: CheckpointMetricConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    grad_norm: torch.Tensor | None,
    token_weighted_active: bool,
    running_loss: torch.Tensor,
    running_loss_sum: torch.Tensor,
    running_supervised_tokens: torch.Tensor,
    seen_supervised_tokens: torch.Tensor,
    last_log_t: float,
    updates_since_log: int,
    ema_update_s: float | None,
    metrics: _MetricsLogger,
    tb: _TensorBoardLogger,
) -> tuple[float, int, float | None]:
    log_interval = int(cfg.log_interval or 0)
    if log_interval <= 0 or (int(global_step) % int(log_interval) != 0):
        return float(last_log_t), int(updates_since_log), ema_update_s

    _sync(device)
    now = time.perf_counter()
    dt = max(now - float(last_log_t), 1e-6)
    update_s = float(dt) / float(max(int(updates_since_log), 1))
    if ema_update_s is None:
        ema_update_s = float(update_s)
    else:
        alpha = 0.1
        ema_update_s = (1.0 - float(alpha)) * float(ema_update_s) + float(alpha) * float(
            update_s
        )
    remaining_updates = max(int(cfg.max_steps) - int(global_step), 0)
    eta_s = float(remaining_updates) * float(ema_update_s or 0.0)

    if token_weighted_active:
        supervised = running_supervised_tokens.clamp_min(1).to(dtype=torch.float32)
        toks_per_s = int(float(supervised.detach().item()) / dt)
        avg_loss = float((running_loss_sum / supervised).detach().item())
    else:
        toks_per_s = int((int(cfg.tokens_per_update) * int(updates_since_log)) / dt)
        avg_loss = float(
            (running_loss / float(max(int(updates_since_log), 1))).detach().item()
        )

    lr = (
        float(scheduler.get_last_lr()[0])
        if scheduler is not None
        else float(optimizer.param_groups[0].get("lr", 0.0) or 0.0)
    )
    mem_gb = 0.0
    if device.type == "cuda":
        mem_gb = float(torch.cuda.max_memory_allocated(device)) / (1024**3)
    grad_norm_value = None
    if grad_norm is not None:
        grad_norm_value = float(grad_norm.detach().item())

    print(
        f"[TRAIN] step={int(global_step)}/{int(cfg.max_steps)} "
        f"loss={float(avg_loss):.4f} lr={float(lr):.3e} tok/s={int(toks_per_s)} "
        + (f"grad_norm={float(grad_norm_value):.3f} " if grad_norm_value is not None else "")
        + f"mem={float(mem_gb):.2f}GB eta={_fmt_duration_s(float(eta_s))}",
        flush=True,
    )

    optimizer_diagnostics = _optimizer_diagnostics(optimizer)
    metrics.log_train(
        step=int(global_step),
        max_steps=int(cfg.max_steps),
        loss=float(avg_loss),
        lr=float(lr),
        tok_s=int(toks_per_s),
        grad_norm=grad_norm_value,
        mem_gb=float(mem_gb),
        dt_s=float(dt),
        updates=int(updates_since_log),
        token_weighted=bool(token_weighted_active),
        tokens_per_update=int(cfg.tokens_per_update),
        seen_supervised_tokens=seen_supervised_tokens,
        optimizer_diagnostics=optimizer_diagnostics,
    )
    tb.log_train(
        step=int(global_step),
        loss=float(avg_loss),
        lr=float(lr),
        tok_s=int(toks_per_s),
        grad_norm=grad_norm_value,
        mem_gb=float(mem_gb),
        dt_s=float(dt),
        updates=int(updates_since_log),
        token_weighted=bool(token_weighted_active),
        tokens_per_update=int(cfg.tokens_per_update),
        seen_supervised_tokens=seen_supervised_tokens,
        optimizer_diagnostics=optimizer_diagnostics,
    )
    for name, value in optimizer_diagnostics.items():
        metrics.log_scalar(step=int(global_step), name=str(name), value=float(value))
        tb.log_scalar(step=int(global_step), name=str(name), value=float(value))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    running_loss.zero_()
    running_loss_sum.zero_()
    running_supervised_tokens.zero_()
    return float(now), 0, ema_update_s


__all__ = [
    "CheckpointLoopConfig",
    "CheckpointMetricConfig",
    "maybe_eval_and_save_best",
    "maybe_log_train_step",
    "maybe_run_extra_evals",
    "maybe_save_interval_checkpoint",
    "maybe_save_interval_weights_checkpoint",
    "maybe_save_model_export",
    "restore_best_eval_loss",
    "save_checkpoint_step",
    "save_weights_checkpoint_step",
]
