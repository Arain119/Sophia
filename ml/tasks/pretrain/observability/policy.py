from __future__ import annotations

import math

import torch

from ml.tasks.pretrain.observability.eta import (
    load_step_time_s_from_machine_artifacts,
)
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.tasks.pretrain.run_kinds import is_budgeted_short_pretrain_run
from ml.training.pretrain.runtime_state import PretrainRuntimeState


def auto_eval_steps_from_tokens(
    *,
    target_tokens: int,
    seq_len: int,
    batch_size: int,
    min_steps: int,
    max_steps: int,
    default_steps: int,
) -> int:
    seq_len = int(seq_len)
    batch_size = int(batch_size)
    if seq_len <= 0 or batch_size <= 0:
        return int(default_steps)
    toks_per_step = int(seq_len) * int(batch_size)
    if toks_per_step <= 0:
        return int(default_steps)
    steps = int(math.ceil(float(int(target_tokens)) / float(toks_per_step)))
    steps = max(int(min_steps), min(int(steps), int(max_steps)))
    return int(steps)


def auto_interval_steps(
    *,
    step_time_s: float,
    target_seconds: int,
    min_steps: int,
    max_steps: int,
) -> int:
    step_time_s = float(step_time_s)
    if not (step_time_s > 0) or not math.isfinite(step_time_s):
        return int(max(min_steps, 1))
    steps = int(math.ceil(float(target_seconds) / float(step_time_s)))
    steps = max(int(min_steps), min(int(steps), int(max_steps)))
    return int(steps)


def auto_observability_policy(
    *,
    args: PretrainRunConfig,
    output_dir: str,
    seq_len: int,
    max_steps: int,
    runtime_state: PretrainRuntimeState | None = None,
) -> PretrainRunConfig:
    cfg = args
    state = (
        runtime_state
        if runtime_state is not None
        else PretrainRuntimeState.from_config(args)
    )
    step_time_s = load_step_time_s_from_machine_artifacts(output_dir=str(output_dir))
    if step_time_s is None:
        return cfg

    batch_size = int(state.batch_size)
    run_kind = str(args._sophia_run_kind or "pretrain")
    validation_budgeted = is_budgeted_short_pretrain_run(run_kind)
    eval_target_tokens = max(int(state.target_tokens_per_update), 1) * 8
    eval_min_steps = 1 if validation_budgeted else 32
    eval_max_steps = max(1, min(int(max_steps), 16)) if validation_budgeted else 256
    eval_enabled = int(state.eval_interval) > 0 and int(state.eval_steps) > 0
    default_eval_steps = int(state.eval_steps)

    if eval_enabled:
        state.eval_steps = auto_eval_steps_from_tokens(
            target_tokens=int(eval_target_tokens),
            seq_len=int(seq_len),
            batch_size=int(batch_size),
            min_steps=int(eval_min_steps),
            max_steps=int(eval_max_steps),
            default_steps=int(default_eval_steps),
        )
    if validation_budgeted:
        state.log_interval = 1
    else:
        state.log_interval = auto_interval_steps(
            step_time_s=float(step_time_s),
            target_seconds=10,
            min_steps=10,
            max_steps=max(int(max_steps), 1),
        )

    if eval_enabled:
        eval_overhead_max = 0.05
        min_eval_interval = int(
            math.ceil(float(state.eval_steps) / float(eval_overhead_max))
        )
        state.eval_interval = max(
            int(min_eval_interval),
            auto_interval_steps(
                step_time_s=float(step_time_s),
                target_seconds=600,
                min_steps=1,
                max_steps=max(int(max_steps), 1),
            ),
        )
    if int(state.save_interval) <= 0:
        state.save_interval = 0
    elif validation_budgeted:
        state.save_interval = 0
    else:
        state.save_interval = auto_interval_steps(
            step_time_s=float(step_time_s),
            target_seconds=1800,
            min_steps=100,
            max_steps=max(int(max_steps), 1),
        )
    cfg = state.project_run_config(args)

    print(
        "[AUTOTUNE] policy | "
        f"step_time_s={float(step_time_s):.4f} "
        f"log_interval={int(state.log_interval)} "
        f"eval_interval={int(state.eval_interval)} eval_steps={int(state.eval_steps)} "
        f"save_interval={int(state.save_interval)}",
        flush=True,
    )
    return cfg


def estimate_cuda_mem_bandwidth_gb_s(*, device: torch.device) -> float | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        idx = device.index
        if idx is None:
            idx = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(int(idx))
    except Exception:
        return None

    clock_khz = None
    for key in ("memory_clock_rate", "memoryClockRate"):
        value = getattr(props, key, None)
        if value is None:
            continue
        try:
            clock_khz = float(value)
        except (TypeError, ValueError):
            clock_khz = None
        break

    bus_bits = None
    for key in ("memory_bus_width", "memoryBusWidth"):
        value = getattr(props, key, None)
        if value is None:
            continue
        try:
            bus_bits = float(value)
        except (TypeError, ValueError):
            bus_bits = None
        break

    if clock_khz is None or bus_bits is None:
        return None
    if not (clock_khz > 0) or not (bus_bits > 0):
        return None

    bw = 2.0 * (float(clock_khz) * 1e3) * (float(bus_bits) / 8.0) / 1e9
    if not (bw > 0) or not math.isfinite(bw):
        return None
    return float(bw)


def model_param_bytes(model: torch.nn.Module) -> int:
    total = 0
    for param in model.parameters():
        if not torch.is_tensor(param):
            continue
        try:
            total += int(param.numel()) * int(param.element_size())
        except Exception:
            continue
    return int(total)


def auto_ema_policy(
    *,
    args: PretrainRunConfig,
    max_steps: int,
    output_dir: str,
    model: torch.nn.Module,
    device: torch.device,
    runtime_state: PretrainRuntimeState | None = None,
) -> PretrainRunConfig:
    cfg = args
    state = (
        runtime_state
        if runtime_state is not None
        else PretrainRuntimeState.from_config(args)
    )
    raw = float(state.ema_decay)
    if not (0.0 < raw < 1.0):
        return cfg
    max_steps = max(int(max_steps), 1)

    try:
        interval_req = int(state.ema_update_interval)
    except (TypeError, ValueError):
        interval_req = -1
    if interval_req > 0:
        return cfg

    step_time_s = load_step_time_s_from_machine_artifacts(output_dir=str(output_dir))
    interval = 1
    bw_gb_s: float | None = None
    ema_update_s_est: float | None = None

    if step_time_s is not None and float(step_time_s) > 0:
        param_bytes = model_param_bytes(model)
        if param_bytes > 0:
            bw_gb_s = estimate_cuda_mem_bandwidth_gb_s(device=device)
            if bw_gb_s is None:
                bw_gb_s = 1000.0
            ema_update_s_est = (3.0 * float(param_bytes)) / (float(bw_gb_s) * 1e9)

            target_frac = 0.01
            denom = float(target_frac) * float(step_time_s)
            interval = int(
                math.ceil(float(ema_update_s_est) / float(max(denom, 1e-6)))
            )
            interval = max(int(interval), 1)

            eval_interval = int(cfg.eval_interval)
            max_interval = 512
            if eval_interval > 0:
                max_interval = min(int(max_interval), max(1, int(eval_interval) // 4))
            interval = min(int(interval), int(max_interval))

    half_life_steps = int(max(200, min(int(max_steps * 0.1), 20_000)))
    per_step_decay = math.exp(math.log(0.5) / float(half_life_steps))
    per_update_decay = float(per_step_decay) ** float(interval)

    state.ema_update_interval = int(interval)
    state.ema_decay = float(per_update_decay)
    cfg = state.project_run_config(args)
    if (
        step_time_s is not None
        and ema_update_s_est is not None
        and bw_gb_s is not None
    ):
        print(
            "[EMA] policy | "
            f"step_time_s={float(step_time_s):.4f} "
            f"bw_est_gb_s={float(bw_gb_s):.0f} "
            f"ema_update_s_est={float(ema_update_s_est):.4f} "
            f"update_interval={int(interval)} "
            f"half_life_steps={int(half_life_steps)} "
            f"ema_decay={float(state.ema_decay):.6f}",
            flush=True,
        )
    else:
        print(
            "[EMA] policy | "
            f"update_interval={int(interval)} "
            f"half_life_steps={int(half_life_steps)} "
            f"ema_decay={float(state.ema_decay):.6f}",
            flush=True,
        )
    return cfg


__all__ = [
    "auto_ema_policy",
    "auto_eval_steps_from_tokens",
    "auto_interval_steps",
    "auto_observability_policy",
    "estimate_cuda_mem_bandwidth_gb_s",
    "model_param_bytes",
]
