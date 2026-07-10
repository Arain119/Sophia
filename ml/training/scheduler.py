from __future__ import annotations

import math

import torch


def _clamp01(x: float) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return min(max(value, 0.0), 1.0)


def cosine_lr_lambda(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    total = max(int(total_steps), 1)
    warmup = max(int(warmup_steps), 0)
    current = max(int(step), 0)
    min_ratio = _clamp01(float(min_lr_ratio))

    if warmup > 0 and current < warmup:
        return float(current) / float(max(warmup, 1))
    if total <= warmup:
        return min_ratio
    progress = float(current - warmup) / float(max(total - warmup, 1))
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


def wsd_lr_lambda(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    stable_steps: int,
    min_lr_ratio: float,
    decay_style: str = "cosine",
) -> float:
    total = max(int(total_steps), 1)
    warmup = max(int(warmup_steps), 0)
    stable = max(int(stable_steps), 0)
    current = max(int(step), 0)
    min_ratio = _clamp01(float(min_lr_ratio))

    if warmup > 0 and current < warmup:
        return float(current) / float(max(warmup, 1))

    stable_end = int(min(int(warmup) + int(stable), int(total)))
    if current < stable_end:
        return 1.0
    if stable_end >= total:
        return float(min_ratio)

    progress = float(current - stable_end) / float(max(total - stable_end, 1))
    progress = _clamp01(progress)

    normalized_style = str(decay_style or "").strip().lower()
    if normalized_style == "linear":
        return float(1.0 - progress * (1.0 - float(min_ratio)))
    return float(min_ratio) + 0.5 * (1.0 - float(min_ratio)) * (
        1.0 + math.cos(math.pi * progress)
    )


def build_cosine_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int | None,
    warmup_ratio: float,
    min_lr_ratio: float,
    resume_state: dict | None,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = int(max_steps)
    warmup = int(warmup_steps or 0)
    if warmup <= 0:
        warmup = int(max(int(total_steps) * float(warmup_ratio), 0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(
            step=int(step),
            total_steps=int(total_steps),
            warmup_steps=int(warmup),
            min_lr_ratio=float(min_lr_ratio),
        ),
    )
    if resume_state is not None:
        scheduler.load_state_dict(resume_state)
    return scheduler


def build_wsd_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int | None,
    warmup_ratio: float,
    stable_ratio: float,
    min_lr_ratio: float,
    decay_style: str = "cosine",
    resume_state: dict | None,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = int(max_steps)
    warmup = int(warmup_steps or 0)
    if warmup <= 0:
        warmup = int(max(int(total_steps) * float(warmup_ratio), 0))

    stable = int(max(int(total_steps) * _clamp01(float(stable_ratio)), 0))
    if stable > max(int(total_steps) - int(warmup), 0):
        stable = max(int(total_steps) - int(warmup), 0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: wsd_lr_lambda(
            step=int(step),
            total_steps=int(total_steps),
            warmup_steps=int(warmup),
            stable_steps=int(stable),
            min_lr_ratio=float(min_lr_ratio),
            decay_style=str(decay_style),
        ),
    )
    if resume_state is not None:
        scheduler.load_state_dict(resume_state)
    return scheduler


def build_lr_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int | None,
    warmup_ratio: float,
    min_lr_ratio: float,
    schedule: str = "cosine",
    wsd_stable_ratio: float = 0.0,
    wsd_decay_style: str = "cosine",
    resume_state: dict | None,
) -> torch.optim.lr_scheduler.LRScheduler:
    normalized_schedule = str(schedule or "").strip().lower()
    if normalized_schedule in {"wsd", "warmup_stable_decay"}:
        return build_wsd_scheduler(
            optimizer=optimizer,
            max_steps=int(max_steps),
            warmup_steps=warmup_steps,
            warmup_ratio=float(warmup_ratio),
            stable_ratio=float(wsd_stable_ratio),
            min_lr_ratio=float(min_lr_ratio),
            decay_style=str(wsd_decay_style),
            resume_state=resume_state,
        )
    if normalized_schedule in {"cosine", ""}:
        return build_cosine_scheduler(
            optimizer=optimizer,
            max_steps=int(max_steps),
            warmup_steps=warmup_steps,
            warmup_ratio=float(warmup_ratio),
            min_lr_ratio=float(min_lr_ratio),
            resume_state=resume_state,
        )
    raise ValueError(f"Unsupported lr schedule: {schedule!r}")
