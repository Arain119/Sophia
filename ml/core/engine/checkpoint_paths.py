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
        path = os.path.join(ckpt_dir, fn)
        if not os.path.isfile(path):
            continue
        num = fn[len(str(prefix)) : -len(str(suffix))]
        try:
            step = int(num)
        except ValueError:
            continue
        out.append((step, path))
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
        latest = latest_checkpoint_path(output_dir)
        if latest is None:
            raise SophiaUsageError(
                "[ERR] resume requested but no checkpoint exists.\n"
                f"output_dir={os.path.abspath(str(output_dir))}"
            )
        sidecar = f"{latest}.sha256"
        if not os.path.isfile(sidecar):
            raise SophiaUsageError(
                "[ERR] latest checkpoint is missing its SHA-256 sidecar.\n"
                f"checkpoint={latest}\n"
                f"sidecar={sidecar}"
            )
        return latest

    path = os.path.abspath(val)
    if os.path.isdir(path):
        ckpts = list_checkpoints(path)
        if ckpts:
            path = ckpts[-1][1]
        else:
            nested = latest_checkpoint_path(path)
            if nested is not None:
                path = nested
            else:
                raise SophiaUsageError(
                    "[ERR] --resume_from_checkpoint directory contains no checkpoints.\n"
                    f"path={path}"
                )
    if not os.path.exists(path):
        raise SophiaUsageError(
            "[ERR] --resume_from_checkpoint path does not exist.\n"
            f"path={path}"
        )
    if not os.path.isfile(path):
        raise SophiaUsageError(
            "[ERR] --resume_from_checkpoint must identify a regular checkpoint file.\n"
            f"path={path}"
        )
    sidecar = f"{path}.sha256"
    if not os.path.isfile(sidecar):
        raise SophiaUsageError(
            "[ERR] resume checkpoint is missing its SHA-256 sidecar.\n"
            f"checkpoint={path}\n"
            f"sidecar={sidecar}"
        )
    return path


__all__ = [
    "latest_checkpoint_path",
    "list_checkpoints",
    "resolve_resume_checkpoint",
]
