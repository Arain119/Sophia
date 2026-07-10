from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
from enum import Enum

import torch

from ml.core.engine.types import StatePayload


def checkpoint_safe_value(value: object) -> object:
    if isinstance(value, Enum):
        return checkpoint_safe_value(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        return {str(k): checkpoint_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [checkpoint_safe_value(v) for v in value]
    if is_dataclass(value):
        return checkpoint_safe_value(asdict(value))
    return str(value)


@dataclass(frozen=True)
class EngineCheckpoint:
    step: int
    model: StatePayload
    optimizer: StatePayload
    scheduler: StatePayload | None
    args: StatePayload | None
    rng: StatePayload | None
    ema: StatePayload | None
    train_state: StatePayload | None


def load_checkpoint(
    path: str,
    *,
    torch_load_fn: Callable[..., object] = torch.load,
) -> EngineCheckpoint:
    try:
        obj = torch_load_fn(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        message = str(exc)
        if "Weights only load failed" in message or "Unsupported global" in message:
            raise RuntimeError(
                "checkpoint contains unsupported object payloads; only "
                "weights-only Sophia checkpoints are supported"
            ) from exc
        raise
    if not isinstance(obj, dict):
        raise RuntimeError(f"Checkpoint is not a dict: {path}")
    return EngineCheckpoint(
        step=int(obj.get("step", 0) or 0),
        model=dict(obj.get("model") or {}),
        optimizer=dict(obj.get("optimizer") or {}),
        scheduler=(dict(obj["scheduler"]) if obj.get("scheduler") is not None else None),
        args=(
            checkpoint_safe_value(dict(obj["args"]))
            if isinstance(obj.get("args"), dict)
            else None
        ),
        rng=(dict(obj["rng"]) if isinstance(obj.get("rng"), dict) else None),
        ema=(dict(obj["ema"]) if isinstance(obj.get("ema"), dict) else None),
        train_state=(
            dict(obj["train_state"]) if isinstance(obj.get("train_state"), dict) else None
        ),
    )


def build_checkpoint_state(
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    args: StatePayload | None,
    rng: StatePayload | None,
    ema: StatePayload | None,
    train_state: StatePayload | None = None,
) -> StatePayload:
    return {
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "args": checkpoint_safe_value(args) if isinstance(args, dict) else args,
        "rng": rng,
        "ema": ema,
        "train_state": train_state,
    }


__all__ = [
    "EngineCheckpoint",
    "build_checkpoint_state",
    "checkpoint_safe_value",
    "load_checkpoint",
]
