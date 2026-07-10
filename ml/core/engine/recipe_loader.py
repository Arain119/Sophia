from __future__ import annotations

import json
import os
from collections.abc import Iterable
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


def apply_machine_recipe_from_path(
    *,
    args: object,
    stage: str,
    recipe_path: str,
    kind: str,
    recipe_key: str,
    int_fields: Iterable[str],
    log_label: str,
) -> dict[str, int] | None:
    loaded = load_json_object_from_path(
        path_label="machine_recipe",
        recipe_path=recipe_path,
    )
    if loaded is None:
        return None
    resolved = str(loaded.path)
    payload = loaded.payload
    raw_stage = str(payload.get("stage", "") or "").strip().lower()
    expected_stage = str(stage or "").strip().lower()
    if str(payload.get("kind", "") or "") != str(kind):
        raise SophiaUsageError(
            "[ERR] unsupported machine recipe kind.\n"
            f"path={resolved}\n"
            f"kind={payload.get('kind')!r}"
        )
    if raw_stage and raw_stage != expected_stage:
        raise SophiaUsageError(
            "[ERR] machine recipe stage mismatch.\n"
            f"expected={expected_stage}\n"
            f"found={raw_stage}\n"
            f"path={resolved}"
        )
    recipe = payload.get(recipe_key)
    if not isinstance(recipe, dict):
        raise SophiaUsageError(
            f"[ERR] {recipe_key} must be an object.\n"
            f"path={resolved}"
        )
    applied: dict[str, int] = {}
    for field in tuple(str(item) for item in int_fields):
        raw = recipe.get(field)
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise SophiaUsageError(
                "[ERR] machine recipe field must be an int.\n"
                f"path={resolved}\n"
                f"field={field}\n"
                f"value={raw!r}"
            ) from exc
        setattr(args, field, int(value))
        applied[field] = int(value)
    print(
        f"[INFO] loaded {str(log_label)} machine recipe "
        f"stage={expected_stage} "
        f"path={resolved} "
        f"settings={json.dumps(applied, ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )
    return applied


def apply_semantic_recipe_from_path(
    *,
    args: object,
    stage: str,
    recipe_path: str,
    kind: str,
    recipe_key: str,
    scalar_fields: Iterable[str],
    list_fields: Iterable[str] = (),
    log_label: str,
) -> dict[str, object] | None:
    loaded = load_json_object_from_path(
        path_label="semantic_recipe",
        recipe_path=recipe_path,
    )
    if loaded is None:
        return None
    resolved = str(loaded.path)
    payload = loaded.payload
    if str(payload.get("kind", "") or "") != str(kind):
        raise SophiaUsageError(
            "[ERR] unsupported semantic recipe kind.\n"
            f"path={resolved}\n"
            f"kind={payload.get('kind')!r}"
        )
    raw_stage = str(payload.get("stage", "") or "").strip().lower()
    expected_stage = str(stage or "").strip().lower()
    if raw_stage and raw_stage != expected_stage:
        raise SophiaUsageError(
            "[ERR] semantic recipe stage mismatch.\n"
            f"expected={expected_stage}\n"
            f"found={raw_stage}\n"
            f"path={resolved}"
        )
    recipe = payload.get(recipe_key)
    if not isinstance(recipe, dict):
        raise SophiaUsageError(
            f"[ERR] {recipe_key} must be an object.\n"
            f"path={resolved}"
        )
    applied: dict[str, object] = {}
    for field in tuple(str(item) for item in scalar_fields):
        if field not in recipe:
            continue
        value = recipe.get(field)
        setattr(args, field, value)
        applied[field] = value
    for field in tuple(str(item) for item in list_fields):
        if field not in recipe:
            continue
        raw_items = recipe.get(field)
        if not isinstance(raw_items, list):
            raise SophiaUsageError(
                f"[ERR] semantic recipe {field} must be a list.\n"
                f"path={resolved}"
            )
        normalized_items = [
            str(item) for item in raw_items if str(item or "").strip()
        ]
        setattr(args, field, list(normalized_items))
        applied[field] = list(normalized_items)
    print(
        f"[INFO] loaded {str(log_label)} semantic recipe "
        f"stage={expected_stage} "
        f"path={resolved} "
        f"settings={json.dumps(applied, ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )
    return applied


__all__ = [
    "LoadedJsonObject",
    "apply_machine_recipe_from_path",
    "apply_semantic_recipe_from_path",
    "load_json_object_from_path",
]
