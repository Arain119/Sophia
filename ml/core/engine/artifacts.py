"""Shared output directory lifecycle helpers."""

from __future__ import annotations

import os

from ml.errors import SophiaUsageError
from ml.core.engine.checkpointing import (
    latest_checkpoint_path,
    resolve_resume_checkpoint,
)
from ml.core.engine.run_dirs import (
    OUTPUT_DIR_MARKER,
    build_output_dir_marker_payload,
    candidate_output_roots,
    looks_like_run_output_dir,
    safe_delete_output_dir,
    write_output_dir_marker,
)


def prepare_output_dir_and_resume(
    *,
    repo_root: str,
    output_dir: str,
    resume_from_checkpoint: str,
    overwrite_output_dir: bool,
    required_output_dir_message: str,
) -> tuple[str, str | None]:
    """Resolve a run output directory and optional resume checkpoint.

    This captures the shared lifecycle used by current post-train stages:

    - require an explicit output directory
    - optionally resolve an explicit or auto-discovered checkpoint
    - apply safe overwrite rules
    - ensure the output directory is a recognized Sophia run dir
    """

    resolved = os.path.abspath(str(output_dir or "").strip())
    if not resolved:
        raise SophiaUsageError(str(required_output_dir_message))

    resume_path = resolve_resume_checkpoint(
        str(resume_from_checkpoint or "").strip(),
        output_dir=str(resolved),
    )
    if bool(overwrite_output_dir) and resume_path is not None:
        raise SophiaUsageError(
            "[ERR] --overwrite_output_dir=1 cannot be combined with --resume_from_checkpoint."
        )
    if os.path.exists(resolved) and not os.path.isdir(resolved):
        raise SophiaUsageError(f"[ERR] output_dir must be a directory path: {resolved}")
    if bool(overwrite_output_dir) and os.path.isdir(resolved):
        safe_delete_output_dir(
            repo_root=str(repo_root),
            output_dir=resolved,
            candidate_roots_fn=lambda: candidate_output_roots(repo_root=str(repo_root)),
        )
    elif os.path.isdir(resolved) and (
        not looks_like_run_output_dir(repo_root=str(repo_root), path=resolved)
    ):
        if os.listdir(resolved):
            raise SophiaUsageError(
                "[ERR] refusing to use a non-empty directory that is not a recognized Sophia run output.\n"
                f"output_dir={resolved}\n"
                "Use an empty directory or a dedicated Sophia run directory."
            )
    os.makedirs(resolved, exist_ok=True)
    write_output_dir_marker(repo_root=str(repo_root), output_dir=resolved)
    if resume_path is None and not bool(overwrite_output_dir):
        auto = latest_checkpoint_path(str(resolved))
        if auto is not None:
            resume_path = str(auto)
            print(f"[INFO] auto-resume enabled: {resume_path}", flush=True)
    return resolved, resume_path


__all__ = [
    "OUTPUT_DIR_MARKER",
    "build_output_dir_marker_payload",
    "candidate_output_roots",
    "looks_like_run_output_dir",
    "safe_delete_output_dir",
    "write_output_dir_marker",
    "prepare_output_dir_and_resume",
]
