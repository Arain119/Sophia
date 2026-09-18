"""Deterministic fingerprint for the executable Sophia implementation."""

from __future__ import annotations

import hashlib
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_ROOT = _REPO_ROOT / "ml"


def _source_files() -> tuple[Path, ...]:
    if not _SOURCE_ROOT.is_dir():
        raise RuntimeError(f"Sophia implementation root is missing: {_SOURCE_ROOT}")
    return tuple(
        sorted(
            (path for path in _SOURCE_ROOT.rglob("*.py") if path.is_file()),
            key=lambda path: path.relative_to(_REPO_ROOT).as_posix(),
        )
    )


def current_pretrain_implementation_sha256() -> str:
    """Hash every production Python source under ``ml/`` with path framing."""

    digest = hashlib.sha256()
    for path in _source_files():
        relative = path.relative_to(_REPO_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


__all__ = ["current_pretrain_implementation_sha256"]
