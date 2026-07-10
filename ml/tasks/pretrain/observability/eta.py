from __future__ import annotations

import json
import math
import os

from ml.training.pretrain.run_config import PretrainRunConfig
from ml.tasks.pretrain.observability.support import (
    format_duration_s,
    try_load_json,
)
from ml.training.pretrain.artifacts import (
    PRETRAIN_MACHINE_RECIPE_ARTIFACT,
    PRETRAIN_MACHINE_RUNTIME_ARTIFACT,
    PRETRAIN_UPDATE_PROFILE_ARTIFACT,
)


def load_step_time_s_from_machine_artifacts(*, output_dir: str) -> float | None:
    output_dir = os.path.abspath(str(output_dir))
    for fname in (
        PRETRAIN_MACHINE_RUNTIME_ARTIFACT,
        PRETRAIN_MACHINE_RECIPE_ARTIFACT,
    ):
        try:
            path = os.path.join(str(output_dir), str(fname))
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as handle:
                obj = json.load(handle)
            best = obj.get("best") if isinstance(obj, dict) else None
            if not isinstance(best, dict):
                continue
            step_time_s = float(best.get("step_time_s", 0.0) or 0.0)
            if step_time_s > 0 and math.isfinite(step_time_s):
                return float(step_time_s)
        except Exception:
            continue
    return None


def load_update_time_s_for_estimate(*, output_dir: str) -> tuple[float | None, str]:
    output_dir = os.path.abspath(str(output_dir))
    try:
        path = os.path.join(str(output_dir), PRETRAIN_UPDATE_PROFILE_ARTIFACT)
        if os.path.exists(path):
            obj = try_load_json(path=str(path))
            if isinstance(obj, dict):
                avg = obj.get("avg_update_breakdown_s")
                if isinstance(avg, dict):
                    try:
                        value = float(avg.get("update_s", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        value = 0.0
                    if value > 0 and math.isfinite(value):
                        return float(value), PRETRAIN_UPDATE_PROFILE_ARTIFACT
    except Exception:
        pass

    step_time_s = load_step_time_s_from_machine_artifacts(output_dir=str(output_dir))
    if step_time_s is not None and float(step_time_s) > 0 and math.isfinite(
        float(step_time_s)
    ):
        return float(step_time_s), PRETRAIN_MACHINE_RUNTIME_ARTIFACT
    return None, ""


def maybe_print_train_eta(
    *,
    args: PretrainRunConfig,
    output_dir: str,
    max_steps: int,
    start_step: int,
    seq_len: int,
) -> None:
    bs = int(args.batch_size)
    accum = int(args.accumulation_steps)
    if bs <= 0 or accum <= 0 or int(seq_len) <= 0:
        return

    update_s, src = load_update_time_s_for_estimate(output_dir=str(output_dir))
    if update_s is None:
        return

    max_steps = max(int(max_steps), 1)
    start_step = max(int(start_step), 0)
    remaining_steps = max(int(max_steps - start_step), 0)
    if remaining_steps <= 0:
        return

    tokens_per_update = int(bs) * int(seq_len) * int(accum)
    toks_s = float(tokens_per_update) / float(update_s) if update_s > 0 else 0.0
    est_full_s = float(update_s) * float(max_steps)
    est_rem_s = float(update_s) * float(remaining_steps)

    note = "excl eval/ckpt"
    if int(args.eval_interval) > 0 and int(args.eval_steps) > 0:
        note = "eval/ckpt overhead is bounded"

    print(
        f"[INFO] eta | src={src} update_s~{float(update_s):.4f} toks/s~{toks_s:.0f} | "
        f"full~{format_duration_s(est_full_s)} "
        f"remaining~{format_duration_s(est_rem_s)} ({note})",
        flush=True,
    )


__all__ = [
    "load_step_time_s_from_machine_artifacts",
    "load_update_time_s_for_estimate",
    "maybe_print_train_eta",
]
