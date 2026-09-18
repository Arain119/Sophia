from __future__ import annotations

import os

def resolve_manifest_path(data_path: str) -> str:
    path = str(data_path or "").strip()
    if not path:
        return path
    if os.path.isdir(path):
        candidate = os.path.join(path, "manifest.json")
        if os.path.exists(candidate):
            return candidate
    return path


def paths_same(lhs: str, rhs: str) -> bool:
    left = os.path.normcase(os.path.normpath(os.path.abspath(str(lhs or ""))))
    right = os.path.normcase(os.path.normpath(os.path.abspath(str(rhs or ""))))
    return bool(left) and bool(right) and left == right


__all__ = [
    "paths_same",
    "resolve_manifest_path",
]
