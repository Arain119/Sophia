"""Training-loop helpers shared across engine-owned tasks."""

from __future__ import annotations

import time
from collections.abc import Callable

from ml.core.engine.types import StatePayload


def compute_throughput_metrics(
    *,
    tokens: int,
    train_time_s: float,
    wall_time_s: float,
) -> dict[str, float | int]:
    sanitized_tokens = max(int(tokens), 0)
    sanitized_train_time_s = max(float(train_time_s), 1e-6)
    sanitized_wall_time_s = max(float(wall_time_s), 1e-6)
    train_tokens_per_s = int(float(sanitized_tokens) / sanitized_train_time_s)
    wall_tokens_per_s = int(float(sanitized_tokens) / sanitized_wall_time_s)
    return {
        "tokens_per_s": int(train_tokens_per_s),
        "train_tokens_per_s": int(train_tokens_per_s),
        "wall_tokens_per_s": int(wall_tokens_per_s),
        "train_time_s": float(sanitized_train_time_s),
        "wall_time_s": float(sanitized_wall_time_s),
    }


def should_fire_interval(
    *,
    global_step: int,
    interval: int,
    max_steps: int,
) -> bool:
    return (
        int(global_step) % max(int(interval), 1) == 0
        or int(global_step) == int(max_steps)
    )


def run_training_loop(
    *,
    output_dir: str,
    start_step: int,
    max_steps: int,
    step_fn: Callable[[int, float, float], object],
    finalize_fn: Callable[[], None],
    append_metrics_row_fn: Callable[[str, dict[str, object]], None],
    save_checkpoint_fn: Callable[[int, StatePayload], None] | None = None,
    time_fn: Callable[[], float] = time.time,
) -> None:
    if int(start_step) >= int(max_steps):
        finalize_fn()
        return

    interval_wall_start = time_fn()
    interval_train_time_s = 0.0
    for step in range(int(start_step), int(max_steps)):
        global_step = int(step) + 1
        result = step_fn(
            global_step,
            float(interval_wall_start),
            float(interval_train_time_s),
        )
        interval_train_time_s += max(float(getattr(result, "train_time_s", 0.0)), 1e-6)
        metrics_row = getattr(result, "metrics_row", None)
        if metrics_row is not None:
            append_metrics_row_fn(str(output_dir), dict(metrics_row))
        for extra_row in tuple(getattr(result, "extra_metrics_rows", ())):
            append_metrics_row_fn(str(output_dir), dict(extra_row))
        print_line = getattr(result, "print_line", None)
        if print_line is not None:
            print(str(print_line), flush=True)
        checkpoint_train_state = getattr(result, "checkpoint_train_state", None)
        if checkpoint_train_state is not None and save_checkpoint_fn is not None:
            save_checkpoint_fn(int(global_step), dict(checkpoint_train_state))
        checkpoint_report_callback = getattr(result, "checkpoint_report_callback", None)
        if callable(checkpoint_report_callback):
            checkpoint_report_callback()
        if bool(getattr(result, "reset_interval", False)):
            interval_wall_start = time_fn()
            interval_train_time_s = 0.0
    finalize_fn()


__all__ = [
    "compute_throughput_metrics",
    "run_training_loop",
    "should_fire_interval",
]
