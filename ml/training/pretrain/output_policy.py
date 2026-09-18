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
_HOST_MOUNT_FILESYSTEMS = frozenset({"9p", "drvfs", "fuseblk"})


@dataclass(frozen=True)
class OutputDirResolution:
    args: PretrainRunConfig
    output_dir: str
    resume_path: str | None


def _default_output_dir() -> str:
    hidden_size = int(_DEFAULT_MODEL_DIM)
    base = os.path.abspath(os.path.join(REPO_ROOT, "out"))
    return os.path.join(base, f"sophia_{hidden_size}_export")


def path_uses_host_mounted_storage(path: str) -> bool:
    if os.name != "posix":
        return False
    probe = os.path.realpath(os.path.abspath(path))
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as handle:
            rows = handle.readlines()
    except OSError:
        return False
    selected_mount = ""
    selected_fstype = ""
    for row in rows:
        before, separator, after = row.partition(" - ")
        fields = before.split()
        if not separator or len(fields) < 5:
            continue
        mount_point = (
            fields[4]
            .replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\134", "\\")
        )
        if probe != mount_point and not probe.startswith(mount_point.rstrip("/") + "/"):
            continue
        if len(mount_point) >= len(selected_mount):
            selected_mount = mount_point
            selected_fstype = after.split()[0].casefold()
    return selected_fstype in _HOST_MOUNT_FILESYSTEMS


def prepare_output_dir_and_resume(cfg: PretrainRunConfig) -> OutputDirResolution:
    output_dir = str(cfg.output_dir or "").strip()
    output_dir_was_explicit = bool(output_dir)
    if not output_dir:
        output_dir = _default_output_dir()

    output_dir = os.path.abspath(str(output_dir))
    if not output_dir:
        raise SophiaUsageError("[ERR] --output_dir is empty.")
    formal_release = str(cfg._sophia_run_kind or "") == "pretrain" and bool(
        cfg._sophia_machine_signature
    )
    if formal_release and path_uses_host_mounted_storage(output_dir):
        raise SophiaUsageError(
            "[ERR] formal pretrain output/checkpoints require target-native storage; "
            f"host-mounted filesystems are forbidden: {output_dir}"
        )
    cfg = replace(cfg, output_dir=str(output_dir))
    overwrite_output_dir = bool(int(cfg.overwrite_output_dir or 0) == 1)
    resume_raw = str(cfg.resume_from_checkpoint or "").strip()
    if overwrite_output_dir and resume_raw:
        raise SophiaUsageError(
            "[ERR] --overwrite_output_dir=1 cannot be combined with --resume_from_checkpoint."
        )
    if (
        (not output_dir_was_explicit)
        and (not overwrite_output_dir)
        and (not resume_raw)
    ):
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
    return OutputDirResolution(
        args=cfg,
        output_dir=str(resolved_output_dir),
        resume_path=resume_path,
    )


__all__ = [
    "OUTPUT_DIR_MARKER",
    "OutputDirResolution",
    "REPO_ROOT",
    "path_uses_host_mounted_storage",
    "prepare_output_dir_and_resume",
]
