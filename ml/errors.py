from __future__ import annotations


class SophiaError(RuntimeError):
    """Base exception for Sophia library failures."""


class SophiaUsageError(SophiaError, ValueError):
    """Raised when library callers provide invalid runtime/config/path inputs."""


__all__ = ["SophiaError", "SophiaUsageError"]
