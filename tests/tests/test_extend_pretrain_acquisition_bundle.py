from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.tooling.scripts.data import extend_pretrain_acquisition_bundle as mod


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(name: str, file_row: dict[str, object]) -> dict[str, object]:
    return {
        "name": name,
        "repo_id": f"owner/{name}",
        "revision": "a" * 40,
        "license": "reviewed",
        "language": "zh",
        "domain": "test",
        "record_format": "parquet",
        "text_column": "text",
        "files": [file_row],
    }


def test_extends_bound_prior_acquisition_without_rehashing_prior_raw(tmp_path) -> None:
    prior_inventory = tmp_path / "prior_inventory.json"
    prior_inventory_row = {
        "path": "old.parquet",
        "bytes": 123,
        "sha256": "b" * 64,
    }
    prior_source = _source("prior", prior_inventory_row)
    _write_json(
        prior_inventory,
        {"schema": mod.INVENTORY_SCHEMA, "sources": [prior_source]},
    )
    prior_acquisition = tmp_path / "prior_acquisition.json"
    _write_json(
        prior_acquisition,
        {
            "schema": mod.ACQUISITION_SCHEMA,
            "inventory_path": str(prior_inventory),
            "inventory_sha256": _sha256(prior_inventory),
            "output_root": str(tmp_path),
            "sources": [
                {
                    **prior_source,
                    "files": [
                        {
                            "filename": "old.parquet",
                            "path": str(tmp_path / "deleted_after_cleaning.parquet"),
                            "bytes": 123,
                            "sha256": "b" * 64,
                        }
                    ],
                }
            ],
        },
    )
    cleaning = tmp_path / "cleaning_report.json"
    _write_json(
        cleaning,
        {
            "schema": mod.CLEANING_SCHEMA,
            "acquisition_report": str(prior_acquisition),
            "acquisition_report_sha256": _sha256(prior_acquisition),
            "inventory_sha256": _sha256(prior_inventory),
        },
    )
    added_file = tmp_path / "new" / "new.parquet"
    added_file.parent.mkdir()
    added_file.write_bytes(b"new-verified-file")
    added_inventory_row = {
        "path": "new.parquet",
        "bytes": added_file.stat().st_size,
        "sha256": _sha256(added_file),
    }
    added_source = _source("added", added_inventory_row)
    added_inventory = tmp_path / "added_inventory.json"
    _write_json(
        added_inventory,
        {"schema": mod.INVENTORY_SCHEMA, "sources": [added_source]},
    )
    added_acquisition = tmp_path / "added_acquisition.json"
    _write_json(
        added_acquisition,
        {
            "schema": mod.ACQUISITION_SCHEMA,
            "inventory_path": str(added_inventory),
            "inventory_sha256": _sha256(added_inventory),
            "output_root": str(tmp_path),
            "sources": [
                {
                    **added_source,
                    "files": [
                        {
                            "filename": "new.parquet",
                            "path": str(added_file),
                            "bytes": added_file.stat().st_size,
                            "sha256": _sha256(added_file),
                        }
                    ],
                }
            ],
        },
    )
    output_inventory = tmp_path / "output_inventory.json"
    output_acquisition = tmp_path / "output_acquisition.json"

    report = mod.extend_bundle(
        prior_acquisition_report=str(prior_acquisition),
        previous_cleaning_report=str(cleaning),
        extension_acquisition_reports=[str(added_acquisition)],
        output_root=str(tmp_path),
        inventory_output=str(output_inventory),
        acquisition_output=str(output_acquisition),
        extension_report=str(tmp_path / "extension_report.json"),
    )

    assert report["status"] == "pass"
    assert report["prior_source_count"] == 1
    assert report["added_sources"] == ["added"]
    inventory = json.loads(output_inventory.read_text(encoding="utf-8"))
    acquisition = json.loads(output_acquisition.read_text(encoding="utf-8"))
    assert [row["name"] for row in inventory["sources"]] == ["prior", "added"]
    assert [row["name"] for row in acquisition["sources"]] == ["prior", "added"]
    assert acquisition["inventory_sha256"] == _sha256(output_inventory)

    cleaning_payload = json.loads(cleaning.read_text(encoding="utf-8"))
    cleaning_payload["acquisition_report_sha256"] = "0" * 64
    _write_json(cleaning, cleaning_payload)
    with pytest.raises(ValueError, match="does not bind"):
        mod.extend_bundle(
            prior_acquisition_report=str(prior_acquisition),
            previous_cleaning_report=str(cleaning),
            extension_acquisition_reports=[str(added_acquisition)],
            output_root=str(tmp_path),
            inventory_output=str(output_inventory),
            acquisition_output=str(output_acquisition),
            extension_report=str(tmp_path / "extension_report.json"),
        )


def test_appends_humanities_derivative_with_bound_report(
    monkeypatch, tmp_path
) -> None:
    prior_inventory = tmp_path / "prior_inventory.json"
    prior_source = _source(
        "prior", {"path": "old.parquet", "bytes": 1, "sha256": "b" * 64}
    )
    _write_json(
        prior_inventory,
        {"schema": mod.INVENTORY_SCHEMA, "sources": [prior_source]},
    )
    prior_acquisition = tmp_path / "prior_acquisition.json"
    _write_json(
        prior_acquisition,
        {
            "schema": mod.ACQUISITION_SCHEMA,
            "inventory_path": str(prior_inventory),
            "inventory_sha256": _sha256(prior_inventory),
            "sources": [
                {
                    **prior_source,
                    "files": [
                        {
                            "filename": "old.parquet",
                            "path": str(tmp_path / "old.parquet"),
                            "bytes": 1,
                            "sha256": "b" * 64,
                        }
                    ],
                }
            ],
        },
    )
    cleaning = tmp_path / "cleaning.json"
    _write_json(
        cleaning,
        {
            "schema": mod.CLEANING_SCHEMA,
            "acquisition_report": str(prior_acquisition),
            "acquisition_report_sha256": _sha256(prior_acquisition),
            "inventory_sha256": _sha256(prior_inventory),
        },
    )
    added_file = tmp_path / "added.parquet"
    added_file.write_bytes(b"added")
    added_source = _source(
        "added",
        {
            "path": "added.parquet",
            "bytes": added_file.stat().st_size,
            "sha256": _sha256(added_file),
        },
    )
    added_inventory = tmp_path / "added_inventory.json"
    _write_json(
        added_inventory,
        {"schema": mod.INVENTORY_SCHEMA, "sources": [added_source]},
    )
    added_acquisition = tmp_path / "added_acquisition.json"
    _write_json(
        added_acquisition,
        {
            "schema": mod.ACQUISITION_SCHEMA,
            "inventory_path": str(added_inventory),
            "inventory_sha256": _sha256(added_inventory),
            "sources": [
                {
                    **added_source,
                    "files": [
                        {
                            "filename": "added.parquet",
                            "path": str(added_file),
                            "bytes": added_file.stat().st_size,
                            "sha256": _sha256(added_file),
                        }
                    ],
                }
            ],
        },
    )
    derivative_report = tmp_path / "derivative.json"
    _write_json(derivative_report, {"schema": "test-derivative"})
    derivative_source = _source(
        "humanities",
        {"path": "humanities.parquet", "bytes": 2, "sha256": "c" * 64},
    )
    monkeypatch.setattr(
        mod,
        "_derivative_source",
        lambda *_args, **_kwargs: (derivative_source, derivative_source),
    )
    output_inventory = tmp_path / "output_inventory.json"
    output_acquisition = tmp_path / "output_acquisition.json"

    report = mod.extend_bundle(
        prior_acquisition_report=str(prior_acquisition),
        previous_cleaning_report=str(cleaning),
        extension_acquisition_reports=[str(added_acquisition)],
        output_root=str(tmp_path),
        inventory_output=str(output_inventory),
        acquisition_output=str(output_acquisition),
        extension_report=str(tmp_path / "extension_report.json"),
        derivative_reports=[str(derivative_report)],
    )

    assert report["added_sources"] == ["added", "humanities"]
    assert report["derivative_reports"] == [
        {"path": str(derivative_report), "sha256": _sha256(derivative_report)}
    ]
    inventory = json.loads(output_inventory.read_text(encoding="utf-8"))
    assert [row["name"] for row in inventory["sources"]] == [
        "prior",
        "added",
        "humanities",
    ]


def _hf_acquisition(
    tmp_path: Path, name: str, payload: bytes
) -> tuple[Path, dict[str, object]]:
    added_file = tmp_path / name / f"{name}.parquet"
    added_file.parent.mkdir(parents=True)
    added_file.write_bytes(payload)
    row = {
        "path": f"{name}.parquet",
        "bytes": added_file.stat().st_size,
        "sha256": _sha256(added_file),
    }
    source = _source(name, row)
    inventory = tmp_path / f"{name}_inventory.json"
    _write_json(inventory, {"schema": mod.INVENTORY_SCHEMA, "sources": [source]})
    acquisition = tmp_path / f"{name}_acquisition.json"
    _write_json(
        acquisition,
        {
            "schema": mod.ACQUISITION_SCHEMA,
            "inventory_path": str(inventory),
            "inventory_sha256": _sha256(inventory),
            "output_root": str(tmp_path),
            "sources": [
                {
                    **source,
                    "files": [
                        {
                            "filename": f"{name}.parquet",
                            "path": str(added_file),
                            "bytes": added_file.stat().st_size,
                            "sha256": _sha256(added_file),
                        }
                    ],
                }
            ],
        },
    )
    return acquisition, source


def test_appends_several_extension_acquisitions_in_argument_order(tmp_path) -> None:
    prior_inventory = tmp_path / "prior_inventory.json"
    prior_source = _source(
        "prior", {"path": "old.parquet", "bytes": 1, "sha256": "b" * 64}
    )
    _write_json(
        prior_inventory, {"schema": mod.INVENTORY_SCHEMA, "sources": [prior_source]}
    )
    prior_acquisition = tmp_path / "prior_acquisition.json"
    _write_json(
        prior_acquisition,
        {
            "schema": mod.ACQUISITION_SCHEMA,
            "inventory_path": str(prior_inventory),
            "inventory_sha256": _sha256(prior_inventory),
            "output_root": str(tmp_path),
            "sources": [
                {
                    **prior_source,
                    "files": [
                        {
                            "filename": "old.parquet",
                            "path": str(tmp_path / "old.parquet"),
                            "bytes": 1,
                            "sha256": "b" * 64,
                        }
                    ],
                }
            ],
        },
    )
    cleaning = tmp_path / "cleaning.json"
    _write_json(
        cleaning,
        {
            "schema": mod.CLEANING_SCHEMA,
            "acquisition_report": str(prior_acquisition),
            "acquisition_report_sha256": _sha256(prior_acquisition),
            "inventory_sha256": _sha256(prior_inventory),
        },
    )
    first, _ = _hf_acquisition(tmp_path, "batch_one", b"first-batch")
    second, _ = _hf_acquisition(tmp_path, "batch_two", b"second-batch")
    output_inventory = tmp_path / "output_inventory.json"
    output_acquisition = tmp_path / "output_acquisition.json"

    report = mod.extend_bundle(
        prior_acquisition_report=str(prior_acquisition),
        previous_cleaning_report=str(cleaning),
        extension_acquisition_reports=[str(first), str(second)],
        output_root=str(tmp_path),
        inventory_output=str(output_inventory),
        acquisition_output=str(output_acquisition),
        extension_report=str(tmp_path / "extension_report.json"),
    )

    assert report["added_sources"] == ["batch_one", "batch_two"]
    assert report["extension_acquisition_reports"] == [
        {"path": str(first), "sha256": _sha256(first)},
        {"path": str(second), "sha256": _sha256(second)},
    ]
    inventory = json.loads(output_inventory.read_text(encoding="utf-8"))
    assert [row["name"] for row in inventory["sources"]] == [
        "prior",
        "batch_one",
        "batch_two",
    ]


def test_rejects_empty_extension_acquisition_list(tmp_path) -> None:
    prior_inventory = tmp_path / "prior_inventory.json"
    prior_source = _source(
        "prior", {"path": "old.parquet", "bytes": 1, "sha256": "b" * 64}
    )
    _write_json(
        prior_inventory, {"schema": mod.INVENTORY_SCHEMA, "sources": [prior_source]}
    )
    prior_acquisition = tmp_path / "prior_acquisition.json"
    _write_json(
        prior_acquisition,
        {
            "schema": mod.ACQUISITION_SCHEMA,
            "inventory_path": str(prior_inventory),
            "inventory_sha256": _sha256(prior_inventory),
            "output_root": str(tmp_path),
            "sources": [
                {
                    **prior_source,
                    "files": [
                        {
                            "filename": "old.parquet",
                            "path": str(tmp_path / "old.parquet"),
                            "bytes": 1,
                            "sha256": "b" * 64,
                        }
                    ],
                }
            ],
        },
    )
    cleaning = tmp_path / "cleaning.json"
    _write_json(
        cleaning,
        {
            "schema": mod.CLEANING_SCHEMA,
            "acquisition_report": str(prior_acquisition),
            "acquisition_report_sha256": _sha256(prior_acquisition),
            "inventory_sha256": _sha256(prior_inventory),
        },
    )

    with pytest.raises(ValueError, match="at least one extension acquisition"):
        mod.extend_bundle(
            prior_acquisition_report=str(prior_acquisition),
            previous_cleaning_report=str(cleaning),
            extension_acquisition_reports=[],
            output_root=str(tmp_path),
            inventory_output=str(tmp_path / "out_inventory.json"),
            acquisition_output=str(tmp_path / "out_acquisition.json"),
            extension_report=str(tmp_path / "extension_report.json"),
        )
