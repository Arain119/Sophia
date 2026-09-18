from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable

from ml.errors import SophiaUsageError

OUTPUT_DIR_MARKER = ".sophia_output_dir.json"


def candidate_output_roots(*, repo_root: str) -> tuple[str, ...]:
    return (os.path.abspath(os.path.join(str(repo_root), "out")),)


def build_output_dir_marker_payload(*, repo_root: str, output_dir: str) -> dict[str, object]:
    resolved_output_dir = os.path.abspath(str(output_dir))
    return {
        "kind": "sophia_output_dir",
        "repo_root": os.path.abspath(str(repo_root)),
        "output_dir": resolved_output_dir,
        "version": 1,
    }


def looks_like_run_output_dir(*, repo_root: str, path: str) -> bool:
    marker_path = os.path.join(str(path), OUTPUT_DIR_MARKER)
    if not os.path.isfile(marker_path):
        return False
    try:
        with open(marker_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    expected = build_output_dir_marker_payload(
        repo_root=str(repo_root),
        output_dir=str(path),
    )
    return payload == expected


def write_output_dir_marker(*, repo_root: str, output_dir: str) -> None:
    resolved_output_dir = os.path.abspath(str(output_dir))
    payload = build_output_dir_marker_payload(
        repo_root=str(repo_root),
        output_dir=resolved_output_dir,
    )
    marker_path = os.path.join(resolved_output_dir, OUTPUT_DIR_MARKER)
    tmp_path = f"{marker_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, marker_path)


def safe_delete_output_dir(
    *,
    repo_root: str,
    output_dir: str,
    candidate_roots_fn: Callable[[], tuple[str, ...]] | None = None,
) -> None:
    resolved = os.path.abspath(str(output_dir))
    if not os.path.isdir(resolved):
        return
    if resolved == os.path.abspath(str(repo_root)):
        raise SophiaUsageError(f"[ERR] refusing to delete repository root: {resolved}")
    if resolved == os.path.dirname(resolved):
        raise SophiaUsageError(f"[ERR] refusing to delete filesystem root: {resolved}")
    candidate_roots = (
        candidate_roots_fn()
        if candidate_roots_fn is not None
        else candidate_output_roots(repo_root=str(repo_root))
    )
    if resolved in tuple(os.path.abspath(str(path)) for path in candidate_roots):
        raise SophiaUsageError(f"[ERR] refusing to delete shared output root directly: {resolved}")
    if not looks_like_run_output_dir(repo_root=str(repo_root), path=resolved):
        raise SophiaUsageError(
            "[ERR] refusing to overwrite a directory that is not a recognized Sophia run output.\n"
            f"output_dir={resolved}\n"
            "Use a dedicated run directory, or rerun after this directory has been created by Sophia."
        )
    shutil.rmtree(resolved)
