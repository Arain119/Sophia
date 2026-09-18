from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

from ml.core.common.io import write_json_atomic
from ml.data.token_shards.tokenizer_fingerprint import compute_tokenizer_bundle_sha1
from ml.tooling.scripts.data.build_pretrain_acquisition_bundle import (
    ACQUISITION_SCHEMA,
    INVENTORY_SCHEMA,
)
from ml.tooling.scripts.data.download_hf_source_inventory import sha256_file
from ml.tooling.scripts.data.profile_pretrain_corpus import STATE_SCHEMA


REPORT_SCHEMA = "sophia_append_only_pretrain_extension_v1"
CLEANING_SCHEMA = "sophia_clean_pretrain_corpus_v1"
PROFILE_SCHEMA = "sophia_pretrain_corpus_token_profile_v1"


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {resolved}")
    return payload


def _json_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def _required_path(value: object, *, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"missing {label}")
    return Path(raw).expanduser().resolve()


def _recorded_absolute_path(value: object, *, label: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"empty {label}")
    expanded = Path(raw).expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{label} must be absolute: {raw}")
    return os.path.normpath(str(expanded))


def _ordered_sources(acquisition: dict[str, Any]) -> list[dict[str, Any]]:
    indexed: list[tuple[int, int, dict[str, Any]]] = []
    for index, source in enumerate(acquisition.get("sources", [])):
        if not isinstance(source, dict):
            raise ValueError("acquisition source rows must be objects")
        priority = source.get("deduplication_priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError("source deduplication_priority must be an integer")
        indexed.append((int(priority), index, source))
    return [
        source
        for _priority, _index, source in sorted(
            indexed, key=lambda item: (-item[0], item[1])
        )
    ]


def _acquisition_inputs(acquisition: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for source in acquisition.get("sources", []):
        if not isinstance(source, dict):
            raise ValueError("acquisition source rows must be objects")
        for row in source.get("files", []):
            if not isinstance(row, dict):
                raise ValueError("acquisition file rows must be objects")
            path = _recorded_absolute_path(
                row.get("path"), label="acquisition input path"
            )
            if path in paths:
                raise ValueError(f"duplicate acquisition input path: {path}")
            paths.add(path)
    return paths


def _clean_output_inventory(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for source in report.get("sources", []):
        if not isinstance(source, dict):
            raise ValueError("cleaning report source rows must be objects")
        rows = source.get("output_file_inventory")
        if not isinstance(rows, list):
            raise ValueError("cleaning report has no output file inventory")
        output_files = {
            _recorded_absolute_path(path, label="clean output path")
            for path in source.get("output_files", [])
        }
        source_inventory: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("cleaning output inventory rows must be objects")
            path = _recorded_absolute_path(
                row.get("path"), label="clean output inventory path"
            )
            if path in source_inventory or path in inventory:
                raise ValueError(f"duplicate clean output path: {path}")
            source_inventory[path] = row
        if set(source_inventory) != output_files:
            raise ValueError("cleaning output inventory does not match output files")
        inventory.update(source_inventory)
    if not inventory:
        raise ValueError("cleaning report has no output files")
    return inventory


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in connection.execute("SELECT key, value FROM metadata")
    }


def _replace_metadata(
    connection: sqlite3.Connection,
    *,
    expected: dict[str, str],
    replacements: dict[str, str],
) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        current = _metadata(connection)
        for key, value in expected.items():
            if current.get(key) != str(value):
                raise ValueError(
                    f"append-only state lineage mismatch for {key}: "
                    f"stored={current.get(key)} expected={value}"
                )
        for key, value in replacements.items():
            cursor = connection.execute(
                "UPDATE metadata SET value = ? WHERE key = ?",
                (str(value), str(key)),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"append-only state lacks metadata key: {key}")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def prepare_clean_extension(
    *,
    previous_cleaning_report: str,
    new_acquisition_report: str,
    state_database: str,
    output_report: str,
    verify_output_hashes: bool = True,
) -> dict[str, Any]:
    previous_report_path = Path(previous_cleaning_report).expanduser().resolve()
    previous = _load_json(previous_report_path)
    if (
        str(previous.get("schema")) != CLEANING_SCHEMA
        or str(previous.get("status")) != "complete"
    ):
        raise ValueError("append-only cleaning requires a complete prior report")
    previous_acquisition_path = _required_path(
        previous.get("acquisition_report"), label="prior acquisition report"
    )
    if _json_sha256(previous_acquisition_path) != str(
        previous.get("acquisition_report_sha256") or ""
    ):
        raise ValueError("prior cleaning acquisition fingerprint mismatch")
    old_acquisition = _load_json(previous_acquisition_path)
    if (
        str(old_acquisition.get("schema")) != ACQUISITION_SCHEMA
        or str(old_acquisition.get("status")) != "complete"
    ):
        raise ValueError("prior cleaning report references an invalid acquisition")
    old_inventory_path = _required_path(
        old_acquisition.get("inventory_path"), label="prior acquisition inventory"
    )
    old_inventory = _load_json(old_inventory_path)
    old_inventory_sha256 = _json_sha256(old_inventory_path)
    if (
        str(old_inventory.get("schema")) != INVENTORY_SCHEMA
        or str(old_inventory.get("status")) != "complete"
        or old_inventory_sha256
        != str(old_acquisition.get("inventory_sha256") or "")
        or old_inventory_sha256 != str(previous.get("inventory_sha256") or "")
    ):
        raise ValueError("prior cleaning inventory lineage mismatch")

    new_acquisition_path = Path(new_acquisition_report).expanduser().resolve()
    new_acquisition = _load_json(new_acquisition_path)
    if (
        str(new_acquisition.get("schema")) != ACQUISITION_SCHEMA
        or str(new_acquisition.get("status")) != "complete"
    ):
        raise ValueError("append-only cleaning requires a complete new acquisition")
    new_inventory_path = _required_path(
        new_acquisition.get("inventory_path"), label="new acquisition inventory"
    )
    new_inventory = _load_json(new_inventory_path)
    new_inventory_sha256 = _json_sha256(new_inventory_path)
    if (
        str(new_inventory.get("schema")) != INVENTORY_SCHEMA
        or str(new_inventory.get("status")) != "complete"
        or new_inventory_sha256
        != str(new_acquisition.get("inventory_sha256") or "")
    ):
        raise ValueError("new acquisition inventory fingerprint mismatch")

    prior_output_root = _required_path(
        previous.get("output_root"), label="prior cleaning output root"
    )
    old_acquisition_root = _required_path(
        old_acquisition.get("output_root"), label="prior acquisition output root"
    )
    new_acquisition_root = _required_path(
        new_acquisition.get("output_root"), label="new acquisition output root"
    )
    if new_acquisition_root != old_acquisition_root:
        raise ValueError("new acquisition output root does not match prior acquisition")

    old_order = _ordered_sources(old_acquisition)
    new_order = _ordered_sources(new_acquisition)
    if len(new_order) <= len(old_order) or new_order[: len(old_order)] != old_order:
        raise ValueError(
            "new acquisition is not a strict append-only extension of the prior "
            "deduplication order"
        )
    old_inputs = _acquisition_inputs(old_acquisition)
    new_inputs = _acquisition_inputs(new_acquisition)
    if not old_inputs < new_inputs:
        raise ValueError("new acquisition input files are not a strict superset")

    output_inventory = _clean_output_inventory(previous)
    verified_outputs = 0
    for path_value, row in output_inventory.items():
        path = Path(path_value)
        if not path.is_file() or int(path.stat().st_size) != int(row.get("bytes") or 0):
            raise ValueError(f"prior clean output size mismatch: {path}")
        if verify_output_hashes and sha256_file(path) != str(row.get("sha256") or ""):
            raise ValueError(f"prior clean output fingerprint mismatch: {path}")
        verified_outputs += 1

    state_path = Path(state_database).expanduser().resolve()
    recorded_state_path = _required_path(
        previous.get("state_database"), label="prior cleaning state database"
    )
    if state_path != recorded_state_path or not state_path.is_file():
        raise ValueError("cleaning state database does not match prior report")
    connection = sqlite3.connect(str(state_path), timeout=120.0)
    try:
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("cleaning state database quick_check failed")
        processed = {
            str(row[0])
            for row in connection.execute("SELECT input_path FROM processed_files")
        }
        if processed != old_inputs:
            raise ValueError("cleaning state processed inputs do not match prior acquisition")
        old_acquisition_sha256 = _json_sha256(previous_acquisition_path)
        new_acquisition_sha256 = _json_sha256(new_acquisition_path)
        _replace_metadata(
            connection,
            expected={
                "acquisition_sha256": old_acquisition_sha256,
                "inventory_sha256": old_inventory_sha256,
                "policy_sha256": str(previous.get("admission_policy_sha256") or ""),
                "output_root": str(prior_output_root),
            },
            replacements={
                "acquisition_sha256": new_acquisition_sha256,
                "inventory_sha256": new_inventory_sha256,
            },
        )
    finally:
        connection.close()

    added_sources = new_order[len(old_order) :]
    report = {
        "schema": REPORT_SCHEMA,
        "phase": "clean",
        "status": "pass",
        "previous_cleaning_report": str(previous_report_path),
        "previous_cleaning_report_sha256": _json_sha256(previous_report_path),
        "previous_acquisition_sha256": _json_sha256(previous_acquisition_path),
        "new_acquisition_report": str(new_acquisition_path),
        "new_acquisition_sha256": _json_sha256(new_acquisition_path),
        "new_inventory_sha256": new_inventory_sha256,
        "state_database": str(state_path),
        "old_source_count": len(old_order),
        "new_source_count": len(new_order),
        "added_sources": [str(source.get("name") or "") for source in added_sources],
        "old_input_file_count": len(old_inputs),
        "new_input_file_count": len(new_inputs),
        "verified_prior_output_files": verified_outputs,
        "verified_prior_output_sha256": bool(verify_output_hashes),
        "deduplication_order_rule": "prior ordered sources are an exact prefix",
    }
    write_json_atomic(
        Path(output_report).expanduser().resolve(),
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return report


def prepare_profile_extension(
    *,
    previous_profile: str,
    previous_cleaning_report: str,
    new_cleaning_report: str,
    state_database: str,
    tokenizer_path: str,
    profiler_path: str,
    taxonomy_path: str,
    output_report: str,
) -> dict[str, Any]:
    previous_profile_path = Path(previous_profile).expanduser().resolve()
    profile = _load_json(previous_profile_path)
    if (
        str(profile.get("schema")) != PROFILE_SCHEMA
        or str(profile.get("status")) != "complete"
    ):
        raise ValueError("append-only profile requires a complete prior profile")
    previous_clean_path = Path(previous_cleaning_report).expanduser().resolve()
    previous_clean = _load_json(previous_clean_path)
    if (
        str(previous_clean.get("schema")) != CLEANING_SCHEMA
        or str(previous_clean.get("status")) != "complete"
    ):
        raise ValueError("append-only profile requires a complete prior cleaning report")
    if _json_sha256(previous_clean_path) != str(
        profile.get("cleaning_report_sha256") or ""
    ):
        raise ValueError("prior profile does not bind the supplied cleaning report")
    new_clean_path = Path(new_cleaning_report).expanduser().resolve()
    new_clean = _load_json(new_clean_path)
    if (
        str(new_clean.get("schema")) != CLEANING_SCHEMA
        or str(new_clean.get("status")) != "complete"
    ):
        raise ValueError("append-only profile requires a complete new cleaning report")
    previous_clean_root = _required_path(
        previous_clean.get("output_root"), label="prior cleaning output root"
    )
    if (
        _required_path(new_clean.get("output_root"), label="new cleaning output root")
        != previous_clean_root
        or _required_path(profile.get("clean_root"), label="prior profile clean root")
        != previous_clean_root
    ):
        raise ValueError("append-only profile cleaning roots do not match")
    old_inventory = _clean_output_inventory(previous_clean)
    new_inventory = _clean_output_inventory(new_clean)
    if not set(old_inventory) < set(new_inventory):
        raise ValueError("new clean outputs are not a strict superset")
    for path, old_row in old_inventory.items():
        if new_inventory[path] != old_row:
            raise ValueError(f"prior clean output inventory changed: {path}")

    files = sorted(new_inventory)
    selected_files_sha256 = hashlib.sha256(
        json.dumps(files, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    old_files = sorted(old_inventory)
    old_selected_files_sha256 = hashlib.sha256(
        json.dumps(old_files, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    state_path = Path(state_database).expanduser().resolve()
    recorded_state_path = _required_path(
        profile.get("resume_state"), label="prior profile state database"
    )
    if state_path != recorded_state_path or not state_path.is_file():
        raise ValueError("profile state database does not match prior profile")
    connection = sqlite3.connect(str(state_path), timeout=120.0)
    try:
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("profile state database quick_check failed")
        profiled_paths = {
            str(row[0]) for row in connection.execute("SELECT path FROM files")
        }
        if profiled_paths != set(old_inventory):
            raise ValueError("profile state files do not match prior clean inventory")
        policy_path = Path(str(profile.get("admission_policy") or "")).resolve()
        if (
            not policy_path.is_file()
            or _json_sha256(policy_path)
            != str(profile.get("admission_policy_sha256") or "")
        ):
            raise ValueError("prior profile admission policy fingerprint mismatch")
        expected_tokenizer = str(profile.get("tokenizer_bundle_sha1") or "")
        actual_tokenizer = compute_tokenizer_bundle_sha1(tokenizer_path)
        if actual_tokenizer != expected_tokenizer:
            raise ValueError("append-only profile tokenizer changed")
        _replace_metadata(
            connection,
            expected={
                "state_schema": STATE_SCHEMA,
                "cleaning_report_sha256": _json_sha256(previous_clean_path),
                "admission_policy_sha256": _json_sha256(policy_path),
                "tokenizer_bundle_sha1": actual_tokenizer,
                "selected_files_sha256": old_selected_files_sha256,
                "profiler_code_sha256": sha256_file(profiler_path),
                "taxonomy_code_sha256": sha256_file(taxonomy_path),
            },
            replacements={
                "cleaning_report_sha256": _json_sha256(new_clean_path),
                "selected_files_sha256": selected_files_sha256,
            },
        )
    finally:
        connection.close()

    report = {
        "schema": REPORT_SCHEMA,
        "phase": "profile",
        "status": "pass",
        "previous_profile": str(previous_profile_path),
        "previous_profile_sha256": _json_sha256(previous_profile_path),
        "previous_cleaning_report_sha256": _json_sha256(previous_clean_path),
        "new_cleaning_report_sha256": _json_sha256(new_clean_path),
        "state_database": str(state_path),
        "old_profiled_file_count": len(old_inventory),
        "new_profiled_file_count": len(new_inventory),
        "added_file_count": len(new_inventory) - len(old_inventory),
        "selected_files_sha256": selected_files_sha256,
        "tokenizer_bundle_sha1": actual_tokenizer,
    }
    write_json_atomic(
        Path(output_report).expanduser().resolve(),
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a verified append-only pretrain data state extension."
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)
    clean = subparsers.add_parser("clean")
    clean.add_argument("--previous-cleaning-report", required=True)
    clean.add_argument("--new-acquisition-report", required=True)
    clean.add_argument("--state-database", required=True)
    clean.add_argument("--output-report", required=True)
    clean.add_argument("--skip-output-sha256", action="store_true")
    profile = subparsers.add_parser("profile")
    profile.add_argument("--previous-profile", required=True)
    profile.add_argument("--previous-cleaning-report", required=True)
    profile.add_argument("--new-cleaning-report", required=True)
    profile.add_argument("--state-database", required=True)
    profile.add_argument("--tokenizer", default="ml/modeling/text")
    profile.add_argument(
        "--profiler",
        default="ml/tooling/scripts/data/profile_pretrain_corpus.py",
    )
    profile.add_argument(
        "--taxonomy",
        default="ml/tooling/core/pretrain_taxonomy_user.py",
    )
    profile.add_argument("--output-report", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "clean":
        report = prepare_clean_extension(
            previous_cleaning_report=str(args.previous_cleaning_report),
            new_acquisition_report=str(args.new_acquisition_report),
            state_database=str(args.state_database),
            output_report=str(args.output_report),
            verify_output_hashes=not bool(args.skip_output_sha256),
        )
    else:
        report = prepare_profile_extension(
            previous_profile=str(args.previous_profile),
            previous_cleaning_report=str(args.previous_cleaning_report),
            new_cleaning_report=str(args.new_cleaning_report),
            state_database=str(args.state_database),
            tokenizer_path=str(args.tokenizer),
            profiler_path=str(args.profiler),
            taxonomy_path=str(args.taxonomy),
            output_report=str(args.output_report),
        )
    print(
        f"[DONE] phase={report['phase']} status={report['status']} "
        f"report={Path(args.output_report).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
