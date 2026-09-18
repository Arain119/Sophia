from __future__ import annotations

import json

from ml.core.common.mapping import object_mapping
from ml.training.pretrain.artifacts import PRETRAIN_LOG_TAG


def log_tag() -> str:
    return f"[{str(PRETRAIN_LOG_TAG)}]"


def int_arg_preserve_zero(
    args: object,
    name: str,
    *,
    default: int,
) -> int:
    raw = object_mapping(args).get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def try_load_json(*, path: str) -> object | None:
    try:
        with open(str(path), encoding="utf-8") as handle:
            return json.load(handle)
    except OSError:
        return None
    except json.JSONDecodeError:
        return None


def format_duration_s(seconds: float) -> str:
    try:
        total_seconds = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "?"
    if total_seconds <= 0:
        return "0s"
    days, rem = divmod(int(total_seconds), 86400)
    hours, rem = divmod(int(rem), 3600)
    minutes, secs = divmod(int(rem), 60)
    if days > 0:
        return f"{days}d{hours}h"
    if hours > 0:
        return f"{hours}h{minutes}m"
    if minutes > 0:
        return f"{minutes}m{secs}s"
    return f"{secs}s"


__all__ = [
    "format_duration_s",
    "int_arg_preserve_zero",
    "log_tag",
    "PRETRAIN_LOG_TAG",
    "try_load_json",
]
