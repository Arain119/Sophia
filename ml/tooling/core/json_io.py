from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_json_dict(
    path: Path,
    *,
    required: bool = True,
    label: str = "json",
) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.is_file():
        if required:
            raise FileNotFoundError(f"missing {label}: {file_path}")
        return None
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object for {label}: {file_path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = file_path.with_name(f"{file_path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, file_path)
