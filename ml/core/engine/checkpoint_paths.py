from __future__ import annotations

import os

from ml.errors import SophiaUsageError


def list_checkpoints(
    ckpt_dir: str,
    *,
    prefix: str = "ckpt_step",
    suffix: str = ".pt",
) -> list[tuple[int, str]]:
    if not os.path.isdir(ckpt_dir):
        return []
    out: list[tuple[int, str]] = []
    for fn in os.listdir(ckpt_dir):
        if not (fn.startswith(str(prefix)) and fn.endswith(str(suffix))):
            continue
        num = fn[len(str(prefix)) : -len(str(suffix))]
        try:
            step = int(num)
        except ValueError:
            continue
        out.append((step, os.path.join(ckpt_dir, fn)))
    out.sort(key=lambda item: item[0])
    return out


def latest_checkpoint_path(output_dir: str) -> str | None:
    ckpts = list_checkpoints(os.path.join(str(output_dir), "checkpoints"))
    return ckpts[-1][1] if ckpts else None


def resolve_resume_checkpoint(resume: str | None, *, output_dir: str) -> str | None:
    val = str(resume or "").strip()
    if not val:
        return None

    lower = val.lower()
    if lower in ("1", "true", "yes", "y", "auto", "latest"):
        return latest_checkpoint_path(output_dir)

    path = os.path.abspath(val)
    if os.path.isdir(path):
        ckpts = list_checkpoints(path)
        if ckpts:
            return ckpts[-1][1]
        nested = latest_checkpoint_path(path)
        if nested is not None:
            return nested
        raise SophiaUsageError(
            "[ERR] --resume_from_checkpoint directory contains no checkpoints.\n"
            f"path={path}"
        )
    if not os.path.exists(path):
        raise SophiaUsageError(
            "[ERR] --resume_from_checkpoint path does not exist.\n"
            f"path={path}"
        )
    return path


__all__ = [
    "latest_checkpoint_path",
    "list_checkpoints",
    "resolve_resume_checkpoint",
]
