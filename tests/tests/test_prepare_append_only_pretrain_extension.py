from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from ml.tooling.scripts.data import prepare_append_only_pretrain_extension as mod


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(name: str, path: Path, priority: int = 0) -> dict[str, object]:
    return {
        "name": name,
        "deduplication_priority": priority,
        "files": [{"path": str(path), "bytes": 1, "sha256": "a" * 64}],
    }


def _clean_state(path: Path, *, acquisition_sha: str, inventory_sha: str, root: Path, input_path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE processed_files(input_path TEXT PRIMARY KEY, source TEXT, file_key TEXT, stage_rel TEXT, stats_json TEXT, output_files_json TEXT);"
    )
    connection.executemany(
        "INSERT INTO metadata(key,value) VALUES (?,?)",
        [
            ("acquisition_sha256", acquisition_sha),
            ("inventory_sha256", inventory_sha),
            ("policy_sha256", "p" * 64),
            ("output_root", str(root)),
        ],
    )
    connection.execute(
        "INSERT INTO processed_files VALUES (?,?,?,?,?,?)",
        (str(input_path), "old", "file", ".staging/old", "{}", "[]"),
    )
    connection.commit()
    connection.close()


def test_clean_extension_updates_only_lineage_after_strict_prefix(tmp_path) -> None:
    old_input = tmp_path / "old.raw"
    new_input = tmp_path / "new.raw"
    old_input.write_bytes(b"x")
    new_input.write_bytes(b"y")
    old_inventory = tmp_path / "old_inventory.json"
    _write(
        old_inventory,
        {"schema": "sophia_hf_source_inventory_v2", "status": "complete"},
    )
    old_acquisition = tmp_path / "old_acquisition.json"
    old_source = _source("old", old_input, priority=100)
    _write(
        old_acquisition,
        {
            "schema": "sophia_hf_acquisition_report_v1",
            "status": "complete",
            "inventory_path": str(old_inventory),
            "inventory_sha256": _sha(old_inventory),
            "output_root": str(tmp_path),
            "sources": [old_source],
        },
    )
    clean_output = tmp_path / "clean.parquet"
    clean_output.write_bytes(b"clean")
    old_report = tmp_path / "old_cleaning.json"
    _write(
        old_report,
        {
            "schema": "sophia_clean_pretrain_corpus_v1",
            "status": "complete",
            "acquisition_report": str(old_acquisition),
            "acquisition_report_sha256": _sha(old_acquisition),
            "inventory_sha256": _sha(old_inventory),
            "admission_policy_sha256": "p" * 64,
            "output_root": str(tmp_path),
            "state_database": str(tmp_path / "clean.sqlite3"),
            "sources": [
                {
                    "output_files": [str(clean_output)],
                    "output_file_inventory": [
                        {
                            "path": str(clean_output),
                            "bytes": len(b"clean"),
                            "sha256": _sha(clean_output),
                        }
                    ],
                }
            ],
        },
    )
    new_inventory = tmp_path / "new_inventory.json"
    _write(
        new_inventory,
        {"schema": "sophia_hf_source_inventory_v2", "status": "complete"},
    )
    new_acquisition = tmp_path / "new_acquisition.json"
    _write(
        new_acquisition,
        {
            "schema": "sophia_hf_acquisition_report_v1",
            "status": "complete",
            "inventory_path": str(new_inventory),
            "inventory_sha256": _sha(new_inventory),
            "output_root": str(tmp_path),
            "sources": [old_source, _source("new", new_input)],
        },
    )
    state = tmp_path / "clean.sqlite3"
    _clean_state(
        state,
        acquisition_sha=_sha(old_acquisition),
        inventory_sha=_sha(old_inventory),
        root=tmp_path,
        input_path=old_input,
    )

    report = mod.prepare_clean_extension(
        previous_cleaning_report=str(old_report),
        new_acquisition_report=str(new_acquisition),
        state_database=str(state),
        output_report=str(tmp_path / "migration.json"),
    )

    assert report["status"] == "pass"
    assert report["added_sources"] == ["new"]
    connection = sqlite3.connect(state)
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    connection.close()
    assert metadata["acquisition_sha256"] == _sha(new_acquisition)
    assert metadata["inventory_sha256"] == _sha(new_inventory)


def test_clean_extension_rejects_priority_insertion(tmp_path) -> None:
    old_input = tmp_path / "old.raw"
    new_input = tmp_path / "new.raw"
    old_input.write_bytes(b"x")
    new_input.write_bytes(b"y")
    old_source = _source("old", old_input)
    old_inventory = tmp_path / "old_inventory.json"
    _write(
        old_inventory,
        {"schema": "sophia_hf_source_inventory_v2", "status": "complete"},
    )
    old_acquisition = tmp_path / "old.json"
    _write(
        old_acquisition,
        {
            "schema": "sophia_hf_acquisition_report_v1",
            "status": "complete",
            "inventory_path": str(old_inventory),
            "inventory_sha256": _sha(old_inventory),
            "output_root": str(tmp_path),
            "sources": [old_source],
        },
    )
    old_report = tmp_path / "clean.json"
    _write(
        old_report,
        {
            "schema": "sophia_clean_pretrain_corpus_v1",
            "status": "complete",
            "acquisition_report": str(old_acquisition),
            "acquisition_report_sha256": _sha(old_acquisition),
            "inventory_sha256": _sha(old_inventory),
            "output_root": str(tmp_path),
        },
    )
    inventory = tmp_path / "inventory.json"
    _write(
        inventory,
        {"schema": "sophia_hf_source_inventory_v2", "status": "complete"},
    )
    new_acquisition = tmp_path / "new.json"
    _write(
        new_acquisition,
        {
            "schema": "sophia_hf_acquisition_report_v1",
            "status": "complete",
            "inventory_path": str(inventory),
            "inventory_sha256": _sha(inventory),
            "output_root": str(tmp_path),
            "sources": [old_source, _source("new", new_input, priority=100)],
        },
    )

    with pytest.raises(ValueError, match="not a strict append-only extension"):
        mod.prepare_clean_extension(
            previous_cleaning_report=str(old_report),
            new_acquisition_report=str(new_acquisition),
            state_database=str(tmp_path / "unused.sqlite3"),
            output_report=str(tmp_path / "migration.json"),
        )


def test_profile_extension_updates_verified_lineage(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_output = tmp_path / "old.parquet"
    new_output = tmp_path / "new.parquet"
    old_output.write_bytes(b"old")
    new_output.write_bytes(b"new")

    def clean_source(path: Path) -> dict[str, object]:
        return {
            "output_files": [str(path)],
            "output_file_inventory": [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}
            ],
        }

    old_clean = tmp_path / "old_clean.json"
    _write(
        old_clean,
        {
            "schema": "sophia_clean_pretrain_corpus_v1",
            "status": "complete",
            "output_root": str(tmp_path),
            "sources": [clean_source(old_output)],
        },
    )
    new_clean = tmp_path / "new_clean.json"
    _write(
        new_clean,
        {
            "schema": "sophia_clean_pretrain_corpus_v1",
            "status": "complete",
            "output_root": str(tmp_path),
            "sources": [clean_source(old_output), clean_source(new_output)],
        },
    )
    policy = tmp_path / "policy.json"
    profiler = tmp_path / "profiler.py"
    taxonomy = tmp_path / "taxonomy.py"
    tokenizer = tmp_path / "tokenizer"
    _write(policy, {"schema": "test"})
    profiler.write_text("# profiler\n", encoding="utf-8")
    taxonomy.write_text("# taxonomy\n", encoding="utf-8")
    tokenizer.mkdir()
    state = tmp_path / "profile.sqlite3"
    old_files_sha = hashlib.sha256(
        json.dumps(
            [str(old_output.resolve())], separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    connection = sqlite3.connect(state)
    connection.executescript(
        "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE files(path TEXT PRIMARY KEY, contribution_json TEXT NOT NULL);"
    )
    connection.executemany(
        "INSERT INTO metadata(key,value) VALUES (?,?)",
        [
            ("state_schema", mod.STATE_SCHEMA),
            ("cleaning_report_sha256", _sha(old_clean)),
            ("admission_policy_sha256", _sha(policy)),
            ("tokenizer_bundle_sha1", "tokenizer-sha1"),
            ("selected_files_sha256", old_files_sha),
            ("profiler_code_sha256", _sha(profiler)),
            ("taxonomy_code_sha256", _sha(taxonomy)),
        ],
    )
    connection.execute(
        "INSERT INTO files(path,contribution_json) VALUES (?,?)",
        (str(old_output.resolve()), "{}"),
    )
    connection.commit()
    connection.close()
    profile = tmp_path / "profile.json"
    _write(
        profile,
        {
            "schema": "sophia_pretrain_corpus_token_profile_v1",
            "status": "complete",
            "clean_root": str(tmp_path),
            "cleaning_report_sha256": _sha(old_clean),
            "admission_policy": str(policy),
            "admission_policy_sha256": _sha(policy),
            "tokenizer_bundle_sha1": "tokenizer-sha1",
            "resume_state": str(state),
        },
    )
    monkeypatch.setattr(
        mod, "compute_tokenizer_bundle_sha1", lambda _path: "tokenizer-sha1"
    )

    report = mod.prepare_profile_extension(
        previous_profile=str(profile),
        previous_cleaning_report=str(old_clean),
        new_cleaning_report=str(new_clean),
        state_database=str(state),
        tokenizer_path=str(tokenizer),
        profiler_path=str(profiler),
        taxonomy_path=str(taxonomy),
        output_report=str(tmp_path / "migration.json"),
    )

    assert report["status"] == "pass"
    assert report["added_file_count"] == 1
    connection = sqlite3.connect(state)
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    connection.close()
    assert metadata["cleaning_report_sha256"] == _sha(new_clean)
    assert metadata["selected_files_sha256"] == report["selected_files_sha256"]
