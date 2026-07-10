"""Shared run bootstrap metadata and artifact writers."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping

from ml.core.engine.run_artifacts import (
    RunArtifacts,
    build_run_artifacts as _build_run_artifacts_impl,
    write_run_artifacts,
)
from ml.core.engine.runtime_metadata import (
    collect_runtime_metadata as _collect_runtime_metadata_impl,
)
from ml.core.engine.types import RuntimeMetadata, StatePayload


def _read_package_version(package: str) -> str | None:
    try:
        return str(importlib.metadata.version(str(package)))
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_command_stdout(argv: list[str], *, cwd: str) -> str | None:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5.0,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None
    return str(completed.stdout)


def _is_wsl_linux() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    release = str(platform.release() or "").lower()
    version = str(platform.version() or "").lower()
    return (
        "microsoft" in release
        or "microsoft" in version
        or "wsl" in release
        or "wsl" in version
    )


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
) -> RuntimeMetadata:
    def _run_command(argv: list[str], *, cwd: str) -> str | None:
        return _run_command_stdout(argv, cwd=cwd)

    return _collect_runtime_metadata_impl(
        run_kind=run_kind,
        repo_root=repo_root,
        device=device,
        start_step=int(start_step),
        max_steps=int(max_steps),
        stack_versions=stack_versions,
        extra_packages=extra_packages,
        extra_meta=extra_meta,
        read_package_version_fn=_read_package_version,
        run_command_stdout_fn=_run_command,
        is_wsl_linux_fn=_is_wsl_linux,
    )


def build_run_artifacts(
    *,
    output_dir: str,
    args_payload: StatePayload,
    run_kind: str,
    start_step: int,
    max_steps: int,
    repo_root: str,
    device: object,
    stack_versions: Mapping[str, str] | None = None,
    extra_packages: Iterable[str] = (),
    extra_meta: Mapping[str, object] | None = None,
) -> RunArtifacts:
    resolved_output_dir = os.path.abspath(str(output_dir))
    meta = collect_runtime_metadata(
        run_kind=str(run_kind or "run"),
        repo_root=str(repo_root),
        device=device,
        start_step=int(start_step),
        max_steps=int(max_steps),
        stack_versions=stack_versions,
        extra_packages=extra_packages,
        extra_meta={
            "output_dir": resolved_output_dir,
            **dict(extra_meta or {}),
        },
    )
    return _build_run_artifacts_impl(
        output_dir=str(output_dir),
        args_payload=args_payload,
        meta=meta,
    )


__all__ = [
    "RunArtifacts",
    "build_run_artifacts",
    "collect_runtime_metadata",
    "write_run_artifacts",
]
