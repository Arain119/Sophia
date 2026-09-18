from __future__ import annotations

import errno
import hashlib
import os
import shutil
import uuid
from collections.abc import Callable, Collection

import torch

from ml.core.engine.checkpoint_paths import list_checkpoints
from ml.core.engine.types import StatePayload

from .checkpoint_payload import build_checkpoint_state
from .checkpoint_payload import checkpoint_safe_value
from .checkpoint_payload import validate_checkpoint_state


def save_checkpoint_state(
    *,
    output_dir: str,
    step: int,
    state: StatePayload,
    save_total_limit: int,
    keep_steps: Collection[int] = (),
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
    validate_checkpoint_state(safe_state)
    if int(safe_state["step"]) != int(step):
        raise ValueError(
            "checkpoint payload step does not match destination step: "
            f"payload={safe_state['step']} destination={step}"
        )

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

    _write_checkpoint_sha256(path)

    _trim_checkpoints(
        ckpt_dir=ckpt_dir,
        save_total_limit=int(save_total_limit),
        prefix=pfx,
        keep_steps=keep_steps,
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
    keep_steps: Collection[int] = (),
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
        keep_steps=keep_steps,
        staging_dir=staging_dir,
        torch_save_fn=torch_save_fn,
    )


def _trim_checkpoints(
    *,
    ckpt_dir: str,
    save_total_limit: int,
    prefix: str,
    keep_steps: Collection[int] = (),
) -> None:
    """Keep the newest `save_total_limit` checkpoints, plus any named milestone.

    A plain rolling window is right while a run is in flight and wrong once it
    ends: the survivors are whatever happened to be last, which for a
    multi-epoch run is always inside the final epoch. `keep_steps` names the
    steps that must outlive the window so an earlier one can still be chosen.
    """
    limit = max(int(save_total_limit), 0)
    if limit <= 0:
        return
    ckpts = list_checkpoints(ckpt_dir, prefix=prefix, suffix=".pt")
    if len(ckpts) <= limit:
        return
    protected = {int(step) for step in keep_steps}
    protected.update(int(step) for step, _path in ckpts[-limit:])
    for step, expired_path in ckpts[:-limit]:
        if int(step) in protected:
            continue
        _cleanup_path(expired_path)


def _write_checkpoint_sha256(path: str) -> None:
    sidecar = f"{path}.sha256"
    _cleanup_path(sidecar)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    temporary = f"{sidecar}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with open(temporary, "w", encoding="ascii") as handle:
            handle.write(f"{digest.hexdigest()}  {os.path.basename(path)}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, sidecar)
    finally:
        _cleanup_path(temporary)


def _cleanup_path(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
    sidecar = f"{path}.sha256"
    if os.path.exists(sidecar):
        os.remove(sidecar)


__all__ = ["save_checkpoint", "save_checkpoint_state"]
