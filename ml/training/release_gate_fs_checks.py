"""Release-gate filesystem checks for pretraining."""

from __future__ import annotations

import os
import shutil

from ml.errors import SophiaUsageError

# The formal checkpoint is about 10.5 GiB on the 1.01B-parameter contract.
# Six retained checkpoints require one additional file while retention trims,
# so 80 GiB is the fixed release floor with serialization and filesystem headroom.
MIN_DISK_FREE_GB = 80.0


def cleanup_stale_tmp_files(*, output_dir: str) -> int:
    root = os.path.abspath(str(output_dir or "").strip())
    if not root or not os.path.isdir(root):
        return 0
    deleted = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if ".tmp." not in str(name):
                continue
            path = os.path.join(dirpath, name)
            try:
                os.remove(path)
                deleted += 1
            except OSError:
                continue
    return int(deleted)


def ensure_disk_free(
    *,
    path: str,
    min_free_gb: float = MIN_DISK_FREE_GB,
    context: str = "release training",
) -> float:
    probe = os.path.abspath(str(path or "").strip())
    if os.path.isfile(probe):
        probe = os.path.dirname(probe) or "."
    while not os.path.exists(probe):
        parent = os.path.dirname(probe) or os.curdir
        if parent == probe:
            raise SophiaUsageError(
                f"[ERR] unable to resolve an existing parent for disk check: {path}"
            )
        probe = parent
    total, used, free = shutil.disk_usage(probe)
    del total, used
    free_gb = float(free) / float(1024**3)
    if free_gb < float(min_free_gb):
        raise SophiaUsageError(
            f"[ERR] insufficient disk space for {context}.\n"
            f"path={probe}\n"
            f"free_gb={free_gb:.2f}\n"
            f"required_gb={float(min_free_gb):.2f}"
        )
    return float(free_gb)


__all__ = [
    "MIN_DISK_FREE_GB",
    "cleanup_stale_tmp_files",
    "ensure_disk_free",
]
