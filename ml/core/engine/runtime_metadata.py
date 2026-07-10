from __future__ import annotations

import os
import platform
import sys
import time
from collections.abc import Callable, Iterable, Mapping

from ml.core.engine.types import RuntimeMetadata


def _base_runtime_metadata(
    *,
    run_kind: str,
    repo_root: str,
    device: object,
    start_step: int,
    max_steps: int,
    is_wsl_linux: bool,
) -> RuntimeMetadata:
    return {
        "time": float(time.time()),
        "run_kind": str(run_kind),
        "repo_root": str(repo_root),
        "start_step": int(start_step),
        "max_steps": int(max_steps),
        "device": str(device),
        "python": str(getattr(sys, "version", "")).split(" ")[0],
        "python_executable": str(sys.executable or ""),
        "platform": str(getattr(sys, "platform", "")),
        "platform_release": str(platform.release() or ""),
        "machine": str(platform.machine() or ""),
        "is_wsl": bool(is_wsl_linux),
        "git_root": None,
        "git_branch": None,
        "git_commit": None,
        "git_commit_short": None,
        "git_describe": None,
        "git_worktree_dirty": None,
        "git_repo_subtree": None,
        "git_repo_subtree_dirty": None,
    }


def _apply_stack_versions(
    meta: RuntimeMetadata,
    *,
    stack_versions: Mapping[str, str] | None,
) -> None:
    stack_payload = {
        str(key): str(value)
        for key, value in dict(stack_versions or {}).items()
    }
    if stack_payload:
        meta["stack_versions"] = stack_payload


def _collect_package_versions(
    *,
    extra_packages: Iterable[str],
    read_package_version_fn: Callable[[str], str | None],
) -> dict[str, str | None]:
    packages = ["ml", *[str(pkg) for pkg in extra_packages]]
    seen_packages: set[str] = set()
    package_versions: dict[str, str | None] = {}
    for package in packages:
        normalized = str(package or "").strip()
        if not normalized or normalized in seen_packages:
            continue
        seen_packages.add(normalized)
        package_versions[normalized] = read_package_version_fn(normalized)
    return package_versions


def _apply_package_versions(
    meta: RuntimeMetadata,
    *,
    extra_packages: Iterable[str],
    read_package_version_fn: Callable[[str], str | None],
) -> None:
    package_versions = _collect_package_versions(
        extra_packages=extra_packages,
        read_package_version_fn=read_package_version_fn,
    )
    if package_versions:
        meta["package_versions"] = package_versions
    meta["ml_version"] = (
        package_versions.get("ml")
        if "ml" in package_versions
        else read_package_version_fn("ml")
    )


def _git_value_or_none(
    run_command_stdout_fn: Callable[[list[str], str], str | None],
    *,
    argv: list[str],
    cwd: str,
) -> str | None:
    raw = run_command_stdout_fn(list(argv), cwd=str(cwd))
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _apply_git_metadata(
    meta: RuntimeMetadata,
    *,
    repo_root: str,
    run_command_stdout_fn: Callable[[list[str], str], str | None],
) -> None:
    git_root_raw = run_command_stdout_fn(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(repo_root),
    )
    if git_root_raw is None:
        return
    git_root = os.path.abspath(str(git_root_raw).strip())
    if not git_root:
        return

    meta["git_root"] = git_root
    meta["git_branch"] = _git_value_or_none(
        run_command_stdout_fn,
        argv=["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=git_root,
    )
    meta["git_commit"] = _git_value_or_none(
        run_command_stdout_fn,
        argv=["git", "rev-parse", "HEAD"],
        cwd=git_root,
    )
    meta["git_commit_short"] = _git_value_or_none(
        run_command_stdout_fn,
        argv=["git", "rev-parse", "--short", "HEAD"],
        cwd=git_root,
    )
    meta["git_describe"] = _git_value_or_none(
        run_command_stdout_fn,
        argv=["git", "describe", "--always", "--dirty"],
        cwd=git_root,
    )
    git_status = run_command_stdout_fn(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=git_root,
    )
    if git_status is not None:
        meta["git_worktree_dirty"] = bool(str(git_status).strip())
    try:
        repo_subtree = os.path.relpath(str(repo_root), git_root)
    except ValueError:
        repo_subtree = "."
    normalized_subtree = "" if repo_subtree == "." else str(repo_subtree)
    meta["git_repo_subtree"] = normalized_subtree or None
    subtree_status = run_command_stdout_fn(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            normalized_subtree or ".",
        ],
        cwd=git_root,
    )
    if subtree_status is not None:
        meta["git_repo_subtree_dirty"] = bool(str(subtree_status).strip())


def collect_runtime_metadata(
    *,
    run_kind: str,
    repo_root: str,
    device: object,
    start_step: int = 0,
    max_steps: int = 0,
    stack_versions: Mapping[str, str] | None = None,
    extra_packages: Iterable[str] = (),
    extra_meta: Mapping[str, object] | None = None,
    read_package_version_fn: Callable[[str], str | None],
    run_command_stdout_fn: Callable[[list[str], str], str | None],
    is_wsl_linux_fn: Callable[[], bool],
) -> RuntimeMetadata:
    import torch

    resolved_repo_root = os.path.abspath(str(repo_root))
    normalized_run_kind = str(run_kind or "").strip() or "run"
    meta = _base_runtime_metadata(
        run_kind=normalized_run_kind,
        repo_root=resolved_repo_root,
        device=device,
        start_step=int(start_step),
        max_steps=int(max_steps),
        is_wsl_linux=bool(is_wsl_linux_fn()),
    )
    meta["torch"] = str(getattr(torch, "__version__", ""))
    meta["cuda"] = str(getattr(torch.version, "cuda", None))
    _apply_stack_versions(meta, stack_versions=stack_versions)
    _apply_package_versions(
        meta,
        extra_packages=extra_packages,
        read_package_version_fn=read_package_version_fn,
    )
    _apply_git_metadata(
        meta,
        repo_root=resolved_repo_root,
        run_command_stdout_fn=run_command_stdout_fn,
    )
    if extra_meta:
        meta.update({str(key): value for key, value in dict(extra_meta).items()})
    return meta


__all__ = ["collect_runtime_metadata"]
