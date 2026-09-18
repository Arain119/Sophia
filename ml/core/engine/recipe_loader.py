from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ml.errors import SophiaUsageError


@dataclass(frozen=True)
class LoadedJsonObject:
    path: str
    payload: dict[str, object]


def load_json_object_from_path(
    *,
    path_label: str,
    recipe_path: str,
) -> LoadedJsonObject | None:
    normalized = str(recipe_path or "").strip()
    if not normalized:
        return None
    resolved = os.path.abspath(normalized)
    if not os.path.isfile(resolved):
        raise SophiaUsageError(
            f"[ERR] --{path_label}_json path does not exist.\n"
            f"path={resolved}"
        )
    try:
        payload = json.loads(Path(resolved).read_text(encoding="utf-8"))
    except Exception as exc:
        raise SophiaUsageError(
            f"[ERR] unable to load {path_label} json.\n"
            f"path={resolved}\n"
            f"error={type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SophiaUsageError(
            f"[ERR] {path_label} json must decode to an object.\n"
            f"path={resolved}"
        )
    return LoadedJsonObject(
        path=resolved,
        payload={str(key): value for key, value in payload.items()},
    )
