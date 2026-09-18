from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
from enum import Enum

import torch

from ml.core.engine.types import StatePayload


ENGINE_CHECKPOINT_SCHEMA = "sophia_engine_checkpoint_v1"
FULL_CHECKPOINT_KIND = "full"
WEIGHTS_ONLY_CHECKPOINT_KIND = "weights_only"
CHECKPOINT_KINDS = frozenset({FULL_CHECKPOINT_KIND, WEIGHTS_ONLY_CHECKPOINT_KIND})


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
    raise TypeError(
        "checkpoint payload contains an unsupported value type: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


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
    schema: str = ENGINE_CHECKPOINT_SCHEMA
    kind: str = FULL_CHECKPOINT_KIND


def load_checkpoint(
    path: str,
    *,
    torch_load_fn: Callable[..., object] = torch.load,
    expected_kind: str | None = None,
) -> EngineCheckpoint:
    sidecar = f"{path}.sha256"
    if not os.path.isfile(sidecar):
        raise RuntimeError(f"checkpoint SHA-256 sidecar is required: {sidecar}")
    with open(sidecar, encoding="ascii") as handle:
        expected = handle.read().split(maxsplit=1)[0].lower()
    if len(expected) != 64:
        raise RuntimeError(f"checkpoint SHA-256 sidecar is invalid: {sidecar}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(
            "checkpoint SHA-256 mismatch: "
            f"path={path} expected={expected} actual={actual}"
        )
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
    validate_checkpoint_state(obj, expected_kind=expected_kind)
    return EngineCheckpoint(
        schema=str(obj["schema"]),
        kind=str(obj["kind"]),
        step=int(obj["step"]),
        model=dict(obj["model"]),
        optimizer=dict(obj.get("optimizer", {})),
        scheduler=(
            dict(obj["scheduler"]) if obj.get("scheduler") is not None else None
        ),
        args=(
            checkpoint_safe_value(dict(obj["args"]))
            if isinstance(obj.get("args"), dict)
            else None
        ),
        rng=(dict(obj["rng"]) if isinstance(obj.get("rng"), dict) else None),
        ema=(dict(obj["ema"]) if isinstance(obj.get("ema"), dict) else None),
        train_state=(
            dict(obj["train_state"])
            if isinstance(obj.get("train_state"), dict)
            else None
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
        "schema": ENGINE_CHECKPOINT_SCHEMA,
        "kind": FULL_CHECKPOINT_KIND,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "args": checkpoint_safe_value(args) if isinstance(args, dict) else args,
        "rng": rng,
        "ema": ema,
        "train_state": train_state,
    }


def validate_checkpoint_state(
    state: object,
    *,
    expected_kind: str | None = None,
) -> None:
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint payload must be an object")
    schema = state.get("schema")
    if schema != ENGINE_CHECKPOINT_SCHEMA:
        raise RuntimeError(
            "unsupported checkpoint schema: "
            f"expected={ENGINE_CHECKPOINT_SCHEMA!r} actual={schema!r}"
        )
    kind = state.get("kind")
    if kind not in CHECKPOINT_KINDS:
        raise RuntimeError(f"unsupported checkpoint kind: {kind!r}")
    if expected_kind is not None and kind != str(expected_kind):
        raise RuntimeError(
            f"checkpoint kind mismatch: expected={expected_kind!r} actual={kind!r}"
        )
    step = state.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise RuntimeError(f"checkpoint step must be an integer >= 0, got {step!r}")
    if not isinstance(state.get("model"), dict):
        raise RuntimeError("checkpoint model state must be an object")

    if kind == FULL_CHECKPOINT_KIND:
        required = {
            "schema",
            "kind",
            "step",
            "model",
            "optimizer",
            "scheduler",
            "args",
            "rng",
            "ema",
            "train_state",
        }
        missing = sorted(required.difference(state))
        unknown = sorted(set(state).difference(required))
        if missing or unknown:
            raise RuntimeError(
                "invalid full checkpoint fields: "
                f"missing={missing} unknown={unknown}"
            )
        if not isinstance(state["optimizer"], dict):
            raise RuntimeError("full checkpoint optimizer state must be an object")
        for field in ("scheduler", "args", "rng", "ema", "train_state"):
            value = state[field]
            if value is not None and not isinstance(value, dict):
                raise RuntimeError(
                    f"full checkpoint {field} state must be an object or null"
                )
        return

    required = {"schema", "kind", "step", "model", "args", "ema"}
    missing = sorted(required.difference(state))
    unknown = sorted(set(state).difference(required))
    if missing or unknown:
        raise RuntimeError(
            "invalid weights-only checkpoint fields: "
            f"missing={missing} unknown={unknown}"
        )
    for field in ("args", "ema"):
        value = state[field]
        if value is not None and not isinstance(value, dict):
            raise RuntimeError(
                f"weights-only checkpoint {field} state must be an object or null"
            )


__all__ = [
    "EngineCheckpoint",
    "ENGINE_CHECKPOINT_SCHEMA",
    "FULL_CHECKPOINT_KIND",
    "WEIGHTS_ONLY_CHECKPOINT_KIND",
    "build_checkpoint_state",
    "checkpoint_safe_value",
    "load_checkpoint",
    "validate_checkpoint_state",
]
