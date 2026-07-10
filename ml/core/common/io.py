"""Small shared IO helpers for the Sophia stack."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from typing import Any


def write_json_atomic(
    path: str | os.PathLike[str],
    obj: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
    default: Callable[[Any], Any] | None = None,
    make_parents: bool = False,
) -> None:
    """
    Atomically write ``obj`` as JSON to ``path`` (write temp + ``os.replace``).

    Centralizes the atomic-write pattern that was previously duplicated across the
    training engine and tooling scripts. Callers pass their own JSON options
    (``sort_keys`` / ``default`` / ``make_parents``) so behavior is unchanged.
    """
    target = os.fspath(path)
    if make_parents:
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
    tmp = f"{target}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(
                obj,
                handle,
                ensure_ascii=ensure_ascii,
                indent=indent,
                sort_keys=sort_keys,
                default=default,
            )
        os.replace(tmp, target)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise


__all__ = ["write_json_atomic"]
