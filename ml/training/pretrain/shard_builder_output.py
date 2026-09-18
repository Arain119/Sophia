from __future__ import annotations

import json
import os
import shutil
import uuid

from ml.errors import SophiaUsageError


OUTPUT_DIR_MARKER = ".sophia_token_shards_output.json"


def _is_non_empty_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    with os.scandir(path) as entries:
        return any(True for _ in entries)


def _write_output_dir_marker(out_dir: str) -> None:
    marker_path = os.path.join(str(out_dir), str(OUTPUT_DIR_MARKER))
    payload = {
        "kind": "sophia_token_shards_output",
        "version": 1,
    }
    tmp_path = f"{marker_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, marker_path)

def _normalize_out_dir(path: str) -> str:
    out_dir = os.path.abspath(str(path))
    if not out_dir:
        raise SophiaUsageError("[ERR] --out_dir is empty.")
    if out_dir == os.path.dirname(out_dir):
        raise SophiaUsageError(f"[ERR] refusing to use filesystem root as --out_dir: {out_dir}")
    return out_dir


def _promotion_backup_dir(final_out_dir: str) -> str:
    return f"{_normalize_out_dir(final_out_dir)}.backup_promote"


def _recover_interrupted_promotion(final_out_dir: str) -> None:
    final_out_dir = _normalize_out_dir(final_out_dir)
    backup_dir = _promotion_backup_dir(final_out_dir)
    if not os.path.exists(backup_dir):
        return
    if not os.path.isfile(os.path.join(backup_dir, OUTPUT_DIR_MARKER)):
        raise SophiaUsageError(
            "[ERR] found a shard-builder promotion backup without a Sophia marker.\n"
            f"backup_dir={backup_dir}\n"
            "Inspect and clean this path manually before retrying."
        )
    if os.path.exists(final_out_dir):
        if os.path.isdir(final_out_dir) and os.path.isfile(
            os.path.join(final_out_dir, OUTPUT_DIR_MARKER)
        ):
            shutil.rmtree(backup_dir)
            return
        raise SophiaUsageError(
            "[ERR] found a stale shard-builder promotion backup alongside an unexpected live output directory.\n"
            f"out_dir={final_out_dir}\n"
            f"backup_dir={backup_dir}\n"
            "Inspect and clean these paths manually before retrying."
        )
    os.replace(backup_dir, final_out_dir)


def _prepare_staging_out_dir(*, out_dir: str, overwrite_output_dir: bool) -> str:
    final_out_dir = _normalize_out_dir(out_dir)
    _recover_interrupted_promotion(final_out_dir)
    if os.path.exists(final_out_dir) and not os.path.isdir(final_out_dir):
        raise SophiaUsageError(f"[ERR] --out_dir exists and is not a directory: {final_out_dir}")
    if _is_non_empty_dir(final_out_dir):
        if not bool(overwrite_output_dir):
            raise SophiaUsageError(
                f"[ERR] out_dir is non-empty: {final_out_dir} "
                "(use --overwrite_output_dir=1 or choose a new --out_dir)"
            )
        if not os.path.isfile(os.path.join(final_out_dir, OUTPUT_DIR_MARKER)):
            raise SophiaUsageError(
                "[ERR] refusing to overwrite a directory that does not look like a Sophia token-shards dataset.\n"
                f"out_dir={final_out_dir}"
            )
    parent = os.path.dirname(final_out_dir) or "."
    os.makedirs(parent, exist_ok=True)
    staging_out_dir = os.path.join(
        parent,
        f"{os.path.basename(final_out_dir)}.tmp_build.{os.getpid()}.{uuid.uuid4().hex}",
    )
    os.makedirs(staging_out_dir, exist_ok=False)
    return staging_out_dir


def _promote_staging_out_dir(
    *,
    staging_out_dir: str,
    final_out_dir: str,
    overwrite_output_dir: bool,
) -> None:
    staging_out_dir = os.path.abspath(str(staging_out_dir))
    final_out_dir = _normalize_out_dir(final_out_dir)
    if not os.path.isdir(staging_out_dir):
        raise RuntimeError(f"staging output directory is missing: {staging_out_dir}")
    manifest_path = os.path.join(staging_out_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise RuntimeError(
            f"staging output directory is incomplete (missing manifest.json): {staging_out_dir}"
        )

    backup_dir = ""
    if os.path.exists(final_out_dir):
        if not os.path.isdir(final_out_dir):
            raise SophiaUsageError(f"[ERR] --out_dir exists and is not a directory: {final_out_dir}")
        if _is_non_empty_dir(final_out_dir):
            if not bool(overwrite_output_dir):
                raise SophiaUsageError(
                    f"[ERR] out_dir became non-empty during build: {final_out_dir}"
                )
            if not os.path.isfile(os.path.join(final_out_dir, OUTPUT_DIR_MARKER)):
                raise SophiaUsageError(
                    "[ERR] refusing to overwrite a directory that does not look like a Sophia token-shards dataset.\n"
                    f"out_dir={final_out_dir}"
                )
        backup_dir = _promotion_backup_dir(final_out_dir)
        if os.path.exists(backup_dir):
            raise SophiaUsageError(
                "[ERR] shard-builder promotion backup already exists.\n"
                f"out_dir={final_out_dir}\n"
                f"backup_dir={backup_dir}\n"
                "Recover or remove the backup before retrying."
            )
        os.replace(final_out_dir, backup_dir)

    promoted = False
    try:
        os.replace(staging_out_dir, final_out_dir)
        promoted = True
    finally:
        if promoted:
            if backup_dir:
                shutil.rmtree(backup_dir)
        elif backup_dir and not os.path.exists(final_out_dir):
            os.replace(backup_dir, final_out_dir)


__all__ = [
    "OUTPUT_DIR_MARKER",
    "_is_non_empty_dir",
    "_normalize_out_dir",
    "_prepare_staging_out_dir",
    "_promote_staging_out_dir",
    "_promotion_backup_dir",
    "_recover_interrupted_promotion",
    "_write_output_dir_marker",
]
