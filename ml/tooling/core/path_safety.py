"""Filesystem safety helpers for destructive data-tool operations."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import shutil

from ml.errors import SophiaUsageError


def _is_filesystem_root(path: Path) -> bool:
    return path == Path(path.anchor)


def require_safe_path(
    *,
    path: Path | str,
    label: str,
    blocked: Iterable[Path | str] = (),
) -> Path:
    resolved = Path(path).expanduser().resolve()
    if _is_filesystem_root(resolved):
        raise SophiaUsageError(f"[ERR] refusing to operate on filesystem root: {resolved}")
    for blocked_path in blocked:
        blocked_resolved = Path(blocked_path).expanduser().resolve()
        if resolved == blocked_resolved:
            raise SophiaUsageError(f"[ERR] refusing to use {label}: {resolved}")
    return resolved


def require_descendant(*, path: Path | str, root: Path | str, label: str) -> Path:
    resolved = require_safe_path(path=path, label=label)
    root_resolved = Path(root).expanduser().resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise SophiaUsageError(
            f"[ERR] {label} must stay under {root_resolved}: {resolved}"
        ) from exc
    if resolved == root_resolved:
        raise SophiaUsageError(f"[ERR] refusing to operate on {label} root directly: {resolved}")
    return resolved


def require_separate_trees(
    *,
    path: Path | str,
    other: Path | str,
    label: str,
    other_label: str,
) -> Path:
    resolved = require_safe_path(path=path, label=label)
    other_resolved = require_safe_path(path=other, label=other_label)
    if resolved == other_resolved:
        raise SophiaUsageError(
            f"[ERR] {label} must differ from {other_label}: {resolved}"
        )
    try:
        resolved.relative_to(other_resolved)
    except ValueError:
        pass
    else:
        raise SophiaUsageError(
            f"[ERR] {label} must not be inside {other_label}: {resolved}"
        )
    try:
        other_resolved.relative_to(resolved)
    except ValueError:
        return resolved
    raise SophiaUsageError(f"[ERR] {label} must not contain {other_label}: {resolved}")


def remove_tree_checked(
    *,
    path: Path | str,
    label: str,
    root: Path | str | None = None,
    missing_ok: bool = True,
) -> None:
    resolved = (
        require_descendant(path=path, root=root, label=label)
        if root is not None
        else require_safe_path(path=path, label=label)
    )
    if not resolved.exists():
        if missing_ok:
            return
        raise SophiaUsageError(f"[ERR] {label} not found: {resolved}")
    shutil.rmtree(resolved)


def swap_directory_trees(
    *,
    src: Path | str,
    dst: Path | str,
    backup: Path | str,
    dst_root: Path | str,
    backup_root: Path | str,
) -> None:
    src_path = require_safe_path(path=src, label="swap source")
    dst_path = require_descendant(path=dst, root=dst_root, label="swap destination")
    backup_path = require_descendant(
        path=backup,
        root=backup_root,
        label="backup destination",
    )
    if not src_path.exists():
        raise SophiaUsageError(f"[ERR] missing swap source dir: {src_path}")
    if not dst_path.exists():
        raise SophiaUsageError(f"[ERR] missing swap destination dir: {dst_path}")
    if backup_path.exists():
        raise SophiaUsageError(f"[ERR] backup already exists: {backup_path}")

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(dst_path), str(backup_path))
    try:
        shutil.move(str(src_path), str(dst_path))
    except Exception:
        if dst_path.exists():
            remove_tree_checked(
                path=dst_path,
                label="partial swap destination",
                root=dst_root,
            )
        try:
            shutil.move(str(backup_path), str(dst_path))
        except Exception:
            pass
        raise
