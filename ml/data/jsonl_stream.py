from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
import os

import orjson  # type: ignore

from ml.errors import SophiaUsageError

@dataclass
class JsonlReadStats:
    """
    Shared JSONL read stats for preprocessing scripts.

    - `lines`: non-empty lines observed
    - `decoded`: successfully decoded JSON values (any JSON type)
    - `decode_errors`: JSON decode failures
    - `non_object`: decoded JSON values that are not objects (dict)
    """

    lines: int = 0
    decoded: int = 0
    decode_errors: int = 0
    non_object: int = 0


@dataclass
class JsonlFieldStats(JsonlReadStats):
    """
    Stats for extracting a specific JSON field from JSONL objects.

    - `missing_field`: JSON objects missing the requested field (or `field` is null)
    - `empty_after_strip`: empty strings produced after caller-side normalization/strip
    - `yielded`: number of yielded field values (pre-strip)
    """

    missing_field: int = 0
    empty_after_strip: int = 0
    yielded: int = 0


def is_json_dataset_file(filename: str) -> bool:
    """
    Return True for JSONL-like dataset source files and False for common artifacts.

    Accepts `.jsonl` and `.json` but excludes:
    - manifests (`manifest.json`)
    - cleaning reports (`*.clean_report.json`, `*.clean_summary.json`)
    - patch reports (`*.dingo_patch_report.json`)
    - token-shard stats (`*.shard_stats.jsonl`)
    - jsonl-index metadata (`*.idx.meta.json`)
    - jsonl-index arrays (`*.idx.npy`, `*.idx.npz`)
    """

    lower = str(filename or "").lower()
    if not lower:
        return False
    if os.path.basename(lower) == "manifest.json":
        return False
    if lower.endswith(".clean_report.json") or lower.endswith(".clean_summary.json"):
        return False
    if lower.endswith(".dingo_patch_report.json"):
        return False
    if lower.endswith(".shard_stats.jsonl"):
        return False
    if lower.endswith(".idx.meta.json"):
        return False
    if lower.endswith(".idx.npy") or lower.endswith(".idx.npz"):
        return False
    return lower.endswith((".jsonl", ".json"))


def reject_json_array_file(path: str) -> None:
    """
    Fail fast on JSON-array files (`[...]`) passed as `.json` inputs.

    JSONL is one JSON object per line. A JSON array will otherwise be silently
    processed line-by-line, which is almost always a mistake.
    """

    lower = str(path).lower()
    if lower.endswith(".jsonl") or (not lower.endswith(".json")):
        return
    try:
        with open(path, "rb") as handle:
            head = handle.read(4096)
    except Exception:
        return
    head = head.lstrip(b"\xef\xbb\xbf").lstrip()
    if head.startswith(b"["):
        raise ValueError(
            f"JSON array files are not supported: {path}\n"
            "Expected JSONL (one JSON object per line). Convert the file to .jsonl before processing."
        )


def iter_jsonl_objects(
    path: str,
    *,
    stats: JsonlReadStats | None = None,
    strict_json: bool = False,
    max_json_errors: int = 0,
    include_non_objects: bool = False,
) -> Iterator[Any]:
    """
    Iterate decoded JSON values from a JSONL file or directory tree.

    - `strict_json=True` re-raises JSON decode errors
    - `max_json_errors>0` aborts after N decode failures
    - `include_non_objects=False` yields only JSON objects (dict)
    """

    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            dirs[:] = sorted(
                [
                    directory
                    for directory in dirs
                    if directory not in (".cache", "__pycache__") and not directory.startswith(".")
                ],
                key=str.lower,
            )
            for filename in sorted(files):
                if filename.startswith("."):
                    continue
                if not is_json_dataset_file(filename):
                    continue
                yield from iter_jsonl_objects(
                    os.path.join(root, filename),
                    stats=stats,
                    strict_json=bool(strict_json),
                    max_json_errors=int(max_json_errors),
                    include_non_objects=bool(include_non_objects),
                )
        return

    reject_json_array_file(path)

    local_decode_errors = 0

    def record_error(err: Exception) -> None:
        nonlocal local_decode_errors
        local_decode_errors += 1
        current = local_decode_errors
        if stats is not None:
            stats.decode_errors += 1
            current = int(stats.decode_errors)
        if int(max_json_errors) > 0 and int(current) >= int(max_json_errors):
            raise SophiaUsageError(
                f"Too many JSON decode errors ({current}) while reading: {path}"
            ) from err

    with open(path, "rb") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(b"\xef\xbb\xbf"):
                line = line[3:]
            if not line:
                continue
            if stats is not None:
                stats.lines += 1
            try:
                obj = orjson.loads(line)
            except Exception as exc:
                record_error(exc)
                if bool(strict_json):
                    raise
                continue
            if stats is not None:
                stats.decoded += 1
            if isinstance(obj, dict):
                yield obj
            else:
                if stats is not None:
                    stats.non_object += 1
                if bool(include_non_objects):
                    yield obj


def iter_jsonl_field(
    path: str,
    field: str,
    *,
    stats: JsonlFieldStats | None = None,
    strict_json: bool = False,
    max_json_errors: int = 0,
) -> Iterator[str]:
    """
    Iterate stringified `field` values from JSONL objects.

    Non-object lines are skipped (counted in `stats.non_object`); objects missing `field`
    are skipped (counted in `stats.missing_field`).
    """

    read_stats: JsonlReadStats | None = stats if stats is not None else None
    for obj in iter_jsonl_objects(
        path,
        stats=read_stats,
        strict_json=bool(strict_json),
        max_json_errors=int(max_json_errors),
        include_non_objects=False,
    ):
        if not isinstance(obj, dict):
            if stats is not None:
                stats.missing_field += 1
            continue
        value = obj.get(field)
        if value is None:
            if stats is not None:
                stats.missing_field += 1
            continue
        if stats is not None:
            stats.yielded += 1
        yield str(value)
