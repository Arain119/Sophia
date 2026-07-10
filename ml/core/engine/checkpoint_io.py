from __future__ import annotations

import errno
import os
import shutil
import uuid
from collections.abc import Callable

import torch

from ml.core.engine.checkpoint_paths import list_checkpoints
from ml.core.engine.types import StatePayload

from .checkpoint_payload import build_checkpoint_state
from .checkpoint_payload import checkpoint_safe_value


def save_checkpoint_state(
    *,
    output_dir: str,
    step: int,
    state: StatePayload,
    save_total_limit: int,
    staging_dir: str | None = None,
    prefix: str = "ckpt_step",
    torch_save_fn: Callable[[object, str], None] = torch.save,
) -> None:
    ckpt_dir = os.path.join(str(output_dir), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    pfx = str(prefix or "ckpt_step")
    path = os.path.join(ckpt_dir, f"{pfx}{int(step)}.pt")
    safe_state = dict(state)
    if isinstance(safe_state.get("args"), dict):
        safe_state["args"] = checkpoint_safe_value(dict(safe_state["args"]))

    suffix = f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
    staging = str(staging_dir or "").strip()
    if staging:
        staging_path = os.path.abspath(staging)
        os.makedirs(staging_path, exist_ok=True)
        staged = os.path.join(staging_path, f"{pfx}{int(step)}.pt{suffix}")
        try:
            torch_save_fn(safe_state, staged)
        except BaseException:
            _cleanup_path(staged)
            raise

        try:
            os.replace(staged, path)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                _cleanup_path(staged)
                raise
            tmp = f"{path}{suffix}"
            try:
                shutil.copyfile(staged, tmp)
                os.replace(tmp, path)
            finally:
                _cleanup_path(tmp)
                _cleanup_path(staged)
    else:
        tmp = f"{path}{suffix}"
        try:
            torch_save_fn(safe_state, tmp)
        except BaseException:
            _cleanup_path(tmp)
            raise
        os.replace(tmp, path)

    _trim_checkpoints(
        ckpt_dir=ckpt_dir,
        save_total_limit=int(save_total_limit),
        prefix=pfx,
    )


def save_checkpoint(
    *,
    output_dir: str,
    step: int,
    model,
    optimizer,
    scheduler,
    args: StatePayload | None,
    rng: StatePayload | None,
    ema: StatePayload | None,
    train_state: StatePayload | None = None,
    save_total_limit: int,
    staging_dir: str | None = None,
    torch_save_fn: Callable[[object, str], None] = torch.save,
) -> None:
    state = build_checkpoint_state(
        step=int(step),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        rng=rng,
        ema=ema,
        train_state=train_state,
    )
    save_checkpoint_state(
        output_dir=str(output_dir),
        step=int(step),
        state=state,
        save_total_limit=int(save_total_limit),
        staging_dir=staging_dir,
        torch_save_fn=torch_save_fn,
    )


def _trim_checkpoints(
    *,
    ckpt_dir: str,
    save_total_limit: int,
    prefix: str,
) -> None:
    limit = max(int(save_total_limit), 0)
    if limit <= 0:
        return
    ckpts = list_checkpoints(ckpt_dir, prefix=prefix, suffix=".pt")
    if len(ckpts) <= limit:
        return
    expired_count = len(ckpts) - limit
    for _expired_step, expired_path in ckpts[:expired_count]:
        _cleanup_path(expired_path)


def _cleanup_path(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


__all__ = ["save_checkpoint", "save_checkpoint_state"]
