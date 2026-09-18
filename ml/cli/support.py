from __future__ import annotations

from collections.abc import Callable
import sys

from ml.errors import SophiaUsageError


def run_cli(entrypoint: Callable[[], object | None]) -> int:
    try:
        result = entrypoint()
        return int(result) if isinstance(result, int) else 0
    except SophiaUsageError as exc:
        print(str(exc), file=sys.stderr)
        return 2


__all__ = ["run_cli"]
