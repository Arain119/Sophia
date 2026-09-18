from __future__ import annotations

import os
import re

from ml.errors import SophiaUsageError

WINDOWS_ABS_PATH_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")


def candidate_local_paths(path: str) -> list[str]:
    raw = str(path).strip()
    if not raw:
        return [os.path.abspath(raw)]
    candidates: list[str] = []
    win_match = WINDOWS_ABS_PATH_RE.match(raw)
    if win_match and os.name != "nt":
        drive = str(win_match.group("drive")).lower()
        rest = str(win_match.group("rest")).replace("\\", "/").lstrip("/")
        candidates.append(f"/mnt/{drive}/{rest}")
    candidates.append(os.path.abspath(raw))
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        normalized = os.path.normpath(str(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def require_local_tokenizer_dir(tokenizer_path: str) -> str:
    candidates = candidate_local_paths(str(tokenizer_path))
    resolved = next((path for path in candidates if os.path.isdir(path)), None)
    if resolved is None:
        raise SophiaUsageError(
            f"[ERR] tokenizer_path must be a local directory: {candidates[0]}"
        )
    tok_json = os.path.join(resolved, "tokenizer.json")
    if not os.path.exists(tok_json):
        raise SophiaUsageError(f"[ERR] Missing tokenizer.json under: {resolved}")
    return resolved


__all__ = ["candidate_local_paths", "require_local_tokenizer_dir"]
