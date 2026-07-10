from __future__ import annotations

import os
from dataclasses import dataclass, replace

from ml.errors import SophiaUsageError
from ml.core.engine.artifacts import (
    OUTPUT_DIR_MARKER,
    prepare_output_dir_and_resume as prepare_run_output_dir_and_resume,
)
from ml.core.spec import ModelSpec
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.core.engine.checkpointing import latest_checkpoint_path


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_MODEL_DIM = int(ModelSpec.default().dim)


@dataclass(frozen=True)
class OutputDirResolution:
    args: PretrainRunConfig
    output_dir: str
    resume_path: str | None

def _default_output_dir() -> str:
    hidden_size = int(_DEFAULT_MODEL_DIM)
    base = os.path.abspath(os.path.join(REPO_ROOT, "out"))
    return os.path.join(base, f"sophia_{hidden_size}_export")


def prepare_output_dir_and_resume(cfg: PretrainRunConfig) -> OutputDirResolution:
    output_dir = str(cfg.output_dir or "").strip()
    output_dir_was_explicit = bool(output_dir)
    if not output_dir:
        output_dir = _default_output_dir()

    output_dir = os.path.abspath(str(output_dir))
    if not output_dir:
        raise SophiaUsageError("[ERR] --output_dir is empty.")
    cfg = replace(cfg, output_dir=str(output_dir))
    overwrite_output_dir = bool(int(cfg.overwrite_output_dir or 0) == 1)
    resume_raw = str(cfg.resume_from_checkpoint or "").strip()
    if overwrite_output_dir and resume_raw:
        raise SophiaUsageError(
            "[ERR] --overwrite_output_dir=1 cannot be combined with --resume_from_checkpoint."
        )
    if (not output_dir_was_explicit) and (not overwrite_output_dir) and (not resume_raw):
        auto = latest_checkpoint_path(str(output_dir))
        if auto is not None:
            raise SophiaUsageError(
                "[ERR] auto-derived output_dir already contains a checkpoint.\n"
                f"Output dir: {output_dir}\n"
                f"Checkpoint: {auto}\n"
                "Pass --output_dir explicitly together with --resume_from_checkpoint, "
                "or set --overwrite_output_dir=1 to start a fresh run."
            )
    resolved_output_dir, resume_path = prepare_run_output_dir_and_resume(
        repo_root=REPO_ROOT,
        output_dir=str(output_dir),
        resume_from_checkpoint=resume_raw,
        overwrite_output_dir=bool(overwrite_output_dir),
        required_output_dir_message="[ERR] --output_dir is empty.",
    )
    if output_dir_was_explicit and (not overwrite_output_dir) and (not resume_raw) and (
        resume_path is not None
    ):
        cfg = replace(cfg, resume_from_checkpoint="auto")
    return OutputDirResolution(
        args=cfg,
        output_dir=str(resolved_output_dir),
        resume_path=resume_path,
    )


__all__ = [
    "OUTPUT_DIR_MARKER",
    "OutputDirResolution",
    "REPO_ROOT",
    "prepare_output_dir_and_resume",
]
