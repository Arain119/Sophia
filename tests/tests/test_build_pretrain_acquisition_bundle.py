from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml.tooling.scripts.data import build_pretrain_acquisition_bundle as mod
from ml.tooling.scripts.data import clean_pretrain_corpus as clean_mod
from ml.tooling.scripts.data.extract_wikimedia_dump import REPORT_SCHEMA as WIKIMEDIA_SCHEMA


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_merges_hf_and_wikimedia_into_cleaner_contract(tmp_path) -> None:
    hf_file = tmp_path / "hf_raw" / "source" / "part.parquet"
    hf_file.parent.mkdir(parents=True)
    hf_file.write_bytes(b"hf-parquet-fixture")
    hf_row = {
        "path": "part.parquet",
        "bytes": hf_file.stat().st_size,
        "sha256": _sha256(hf_file),
    }
    hf_source = {
        "name": "hf_source",
        "repo_id": "owner/repo",
        "revision": "a" * 40,
        "license": "reviewed",
        "language": "en",
        "domain": "knowledge",
        "record_format": "parquet",
        "text_column": "text",
        "files": [hf_row],
    }
    hf_inventory = tmp_path / "hf_inventory.json"
    _write_json(
        hf_inventory,
        {"schema": mod.INVENTORY_SCHEMA, "sources": [hf_source]},
    )
    hf_acquisition = tmp_path / "hf_acquisition.json"
    _write_json(
        hf_acquisition,
        {
            "schema": mod.ACQUISITION_SCHEMA,
            "inventory_path": str(hf_inventory),
            "inventory_sha256": _sha256(hf_inventory),
            "output_root": str(tmp_path / "hf_raw"),
            "sources": [
                {
                    **hf_source,
                    "files": [
                        {
                            "filename": "part.parquet",
                            "path": str(hf_file),
                            "bytes": hf_file.stat().st_size,
                            "sha256": _sha256(hf_file),
                        }
                    ],
                }
            ],
        },
    )

    wiki_root = tmp_path / "wiki_extracted"
    wiki_name = "wikimedia_zhwiki_20260701_v1"
    wiki_file = wiki_root / wiki_name / "shard-00001" / "data.parquet"
    wiki_file.parent.mkdir(parents=True)
    wiki_file.write_bytes(b"wiki-parquet-fixture")
    wiki_report = tmp_path / "wiki_report.json"
    _write_json(
        wiki_report,
        {
            "schema": WIKIMEDIA_SCHEMA,
            "status": "complete",
            "inventory_sha256": "b" * 64,
            "output_root": str(wiki_root),
            "sources": [
                {
                    "name": wiki_name,
                    "wiki": "zhwiki",
                    "snapshot_date": "20260701",
                    "shards": [
                        {
                            "status": "complete",
                            "shard_index": 1,
                            "output_sha256": _sha256(wiki_file),
                        }
                    ],
                }
            ],
        },
    )
    inventory_output = tmp_path / "bundle_inventory.json"
    acquisition_output = tmp_path / "bundle_acquisition.json"
    report = mod.build_bundle(
        hf_acquisition_reports=[str(hf_acquisition)],
        wikimedia_extraction_reports=[str(wiki_report)],
        output_root=str(tmp_path),
        inventory_output=str(inventory_output),
        acquisition_output=str(acquisition_output),
    )

    assert report["status"] == "complete"
    assert {row["name"] for row in report["sources"]} == {"hf_source", wiki_name}
    assert report["sources"][1]["files"][0]["path"] == str(wiki_file.resolve())
    assert report["sources"][1]["deduplication_priority"] == 100
    policy = tmp_path / "policy.json"
    _write_json(
        policy,
        {
            "schema": "sophia_pretrain_data_admission_policy_v1",
            "allow_legacy_assets_as_training_input": False,
            "forbidden_input_roots": [],
            "forbidden_upstream_repo_ids": [],
        },
    )
    acquisition, inventory, *_rest = clean_mod._validate_acquisition(
        acquisition_path=str(acquisition_output),
        policy_path=str(policy),
    )
    assert len(acquisition["sources"]) == 2
    assert len(inventory["sources"]) == 2


def test_bundle_accepts_controlled_wikimedia_sister_project(tmp_path) -> None:
    root = tmp_path / "wiki_extracted"
    name = "wikimedia_zhwikibooks_20260701_v1"
    data = root / name / "shard-00001" / "data.parquet"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"wikibooks-parquet-fixture")
    report_path = tmp_path / "wiki_report.json"
    _write_json(
        report_path,
        {
            "schema": WIKIMEDIA_SCHEMA,
            "status": "complete",
            "inventory_sha256": "b" * 64,
            "output_root": str(root),
            "sources": [
                {
                    "name": name,
                    "wiki": "zhwikibooks",
                    "snapshot_date": "20260701",
                    "shards": [
                        {
                            "status": "complete",
                            "shard_index": 1,
                            "output_sha256": _sha256(data),
                        }
                    ],
                }
            ],
        },
    )

    result = mod.build_bundle(
        hf_acquisition_reports=[],
        wikimedia_extraction_reports=[str(report_path)],
        output_root=str(tmp_path),
        inventory_output=str(tmp_path / "inventory.json"),
        acquisition_output=str(tmp_path / "acquisition.json"),
    )

    source = result["sources"][0]
    assert source["repo_id"] == "wikimedia/zhwikibooks"
    assert source["language"] == "zh"
    assert source["domain"] == "open_chinese_textbooks_and_humanities"
    assert source["license"] == "cc-by-sa-4.0-and-gfdl-page-level"
    assert source["deduplication_priority"] == 0
