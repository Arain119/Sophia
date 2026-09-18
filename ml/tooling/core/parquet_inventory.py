from __future__ import annotations

"""Helpers for enumerating parquet pools with mixed plain/reshard representations."""

from collections.abc import Iterable
from dataclasses import dataclass
from glob import glob
import os
from pathlib import Path

_PARQUET_SUFFIX = ".parquet"
_RESHARD_MARKER = ".reshard."
_SRC_MARKER = ".src"


@dataclass(frozen=True)
class ParquetRepresentationFamily:
    directory: str
    base_name: str
    plain_paths: tuple[str, ...]
    reshard_paths: tuple[str, ...]

    @property
    def has_both(self) -> bool:
        return bool(self.plain_paths) and bool(self.reshard_paths)


def is_reshard_name(name: str) -> bool:
    return _RESHARD_MARKER in str(name)


def parquet_family_base_name(name: str) -> str:
    filename = Path(str(name)).name
    stem = filename[: -len(_PARQUET_SUFFIX)] if filename.endswith(_PARQUET_SUFFIX) else filename
    if _RESHARD_MARKER in stem:
        return stem.split(_RESHARD_MARKER, 1)[0]
    return stem.split(_SRC_MARKER, 1)[0]


def _normalized_abs_paths(paths: Iterable[str | os.PathLike[str]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        normalized = os.path.abspath(str(path))
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    out.sort(key=str.lower)
    return out


def representation_families_for_paths(
    paths: Iterable[str | os.PathLike[str]],
) -> list[ParquetRepresentationFamily]:
    grouped: dict[tuple[str, str], dict[str, list[str]]] = {}
    for normalized in _normalized_abs_paths(paths):
        path = Path(normalized)
        key = (str(path.parent), parquet_family_base_name(path.name))
        group = grouped.setdefault(key, {"plain": [], "reshard": []})
        if is_reshard_name(path.name):
            group["reshard"].append(normalized)
        else:
            group["plain"].append(normalized)

    families: list[ParquetRepresentationFamily] = []
    for (directory, base_name), group in sorted(grouped.items(), key=lambda kv: (kv[0][0].lower(), kv[0][1].lower())):
        families.append(
            ParquetRepresentationFamily(
                directory=directory,
                base_name=base_name,
                plain_paths=tuple(sorted(group["plain"], key=str.lower)),
                reshard_paths=tuple(sorted(group["reshard"], key=str.lower)),
            )
        )
    return families


def choose_canonical_parquet_paths(
    paths: Iterable[str | os.PathLike[str]],
) -> tuple[list[str], list[str], list[ParquetRepresentationFamily]]:
    kept: list[str] = []
    dropped: list[str] = []
    families = representation_families_for_paths(paths)
    for family in families:
        if family.plain_paths:
            kept.extend(family.plain_paths)
            dropped.extend(family.reshard_paths)
        else:
            kept.extend(family.reshard_paths)
    kept.sort(key=str.lower)
    dropped.sort(key=str.lower)
    return kept, dropped, families


def list_glob_prefer_plain(pattern: str) -> list[str]:
    matched = glob(str(pattern), recursive=True)
    kept, _dropped, _families = choose_canonical_parquet_paths(matched)
    return kept


def list_tree_parquets_prefer_plain(root: str | os.PathLike[str]) -> list[Path]:
    root_path = Path(str(root)).expanduser().resolve()
    matched = root_path.rglob("*.parquet")
    kept, _dropped, _families = choose_canonical_parquet_paths(matched)
    return [Path(path) for path in kept]
