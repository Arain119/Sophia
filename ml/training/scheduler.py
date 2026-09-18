from __future__ import annotations

import math

import torch


def _validate_ratio(name: str, x: float) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number in [0, 1], got {x!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _load_scheduler_state(
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer: torch.optim.Optimizer,
    resume_state: dict | None,
    expected_step: int | None,
) -> None:
    if resume_state is None:
        return
    try:
        scheduler.load_state_dict(resume_state)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("scheduler resume state is invalid") from exc
    last_epoch = int(getattr(scheduler, "last_epoch", -1))
    step_count = int(getattr(scheduler, "_step_count", -1))
    if last_epoch < 0 or step_count != last_epoch + 1:
        raise ValueError(
            "scheduler resume counters are inconsistent: "
            f"last_epoch={last_epoch} _step_count={step_count}"
        )
    if expected_step is not None and last_epoch != int(expected_step):
        raise ValueError(
            "scheduler resume step does not match checkpoint step: "
            f"last_epoch={last_epoch} checkpoint_step={int(expected_step)}"
        )
    saved_lrs = [float(value) for value in scheduler.get_last_lr()]
    if len(saved_lrs) != len(optimizer.param_groups):
        raise ValueError(
            "scheduler LR group count does not match optimizer param groups: "
            f"{len(saved_lrs)} != {len(optimizer.param_groups)}"
        )
    if any(not math.isfinite(value) or value < 0.0 for value in saved_lrs):
        raise ValueError(f"scheduler state contains invalid learning rates: {saved_lrs}")
    for group, learning_rate in zip(optimizer.param_groups, saved_lrs, strict=True):
        group["lr"] = learning_rate


def cosine_lr_lambda(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    total = int(total_steps)
    warmup = int(warmup_steps)
    if total <= 0:
        raise ValueError(f"total_steps must be > 0, got {total}")
    if warmup < 0 or warmup >= total:
        raise ValueError(
            f"warmup_steps must be in [0, total_steps), got {warmup}"
        )
    current = int(step)
    if current < 0 or current > total:
        raise ValueError(f"step must be in [0, {total}], got {current}")
    min_ratio = _validate_ratio("min_lr_ratio", min_lr_ratio)

    if warmup > 0 and current < warmup:
        return float(current) / float(warmup)
    progress = float(current - warmup) / float(total - warmup)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


def build_cosine_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int | None,
    warmup_ratio: float,
    min_lr_ratio: float,
    resume_state: dict | None,
    resume_step: int | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = int(max_steps)
    warmup = int(warmup_steps or 0)
    if total_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {total_steps}")
    if warmup < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup}")
    warmup_fraction = _validate_ratio("warmup_ratio", warmup_ratio)
    _validate_ratio("min_lr_ratio", min_lr_ratio)
    if warmup == 0:
        warmup = int(int(total_steps) * warmup_fraction)
    if warmup >= total_steps:
        raise ValueError(
            f"warmup must leave at least one scheduled step, got {warmup}/{total_steps}"
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(
            step=int(step),
            total_steps=int(total_steps),
            warmup_steps=int(warmup),
            min_lr_ratio=float(min_lr_ratio),
        ),
    )
    _load_scheduler_state(
        scheduler=scheduler,
        optimizer=optimizer,
        resume_state=resume_state,
        expected_step=resume_step,
    )
    return scheduler


def build_lr_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int | None,
    warmup_ratio: float,
    min_lr_ratio: float,
    schedule: str = "cosine",
    resume_state: dict | None,
    resume_step: int | None = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    normalized_schedule = str(schedule or "").strip().lower()
    if normalized_schedule != "cosine":
        raise ValueError(f"lr_schedule must be 'cosine', got {schedule!r}")
    return build_cosine_scheduler(
        optimizer=optimizer,
        max_steps=int(max_steps),
        warmup_steps=warmup_steps,
        warmup_ratio=float(warmup_ratio),
        min_lr_ratio=float(min_lr_ratio),
        resume_state=resume_state,
        resume_step=resume_step,
    )
