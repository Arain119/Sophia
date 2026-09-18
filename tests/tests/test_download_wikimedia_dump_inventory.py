from __future__ import annotations

import hashlib
import json

import pytest

from ml.tooling.scripts.data import download_wikimedia_dump_inventory as mod
from ml.tooling.scripts.data.snapshot_wikimedia_dump_inventory import INVENTORY_SCHEMA


def test_download_attempt_override_is_bounded(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SOPHIA_WIKIMEDIA_DOWNLOAD_ATTEMPTS", "10001")
    with pytest.raises(ValueError, match="must be in"):
        mod._download_one(
            row={"bytes": 1, "url": "https://example.invalid/file"},
            target=tmp_path / "file",
        )


def test_download_inventory_records_verified_files(tmp_path) -> None:
    content = b"immutable-wikimedia-dump"
    inventory = tmp_path / "inventory.json"
    row = {
        "filename": "zhwiki-20260701-pages-articles-multistream.xml.bz2",
        "role": "articles",
        "bytes": len(content),
        "sha1": hashlib.sha1(content, usedforsecurity=False).hexdigest(),
        "md5": hashlib.md5(content, usedforsecurity=False).hexdigest(),
        "url": "https://dumps.wikimedia.org/zhwiki/20260701/file.bz2",
    }
    inventory.write_text(
        json.dumps(
            {
                "schema": INVENTORY_SCHEMA,
                "status": "complete",
                "mutable_latest_urls_forbidden": True,
                "sources": [
                    {
                        "name": "wikimedia_zhwiki_20260701_v1",
                        "wiki": "zhwiki",
                        "snapshot_date": "20260701",
                        "files": [row],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def _download(*, row, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return mod._verify_file(target, row)

    report = mod.download_inventory(
        inventory_path=str(inventory),
        output_root=str(tmp_path / "raw"),
        report_path=str(tmp_path / "report.json"),
        downloader=_download,
    )

    assert report["status"] == "complete"
    assert report["acquired_file_count"] == 1
    assert report["acquired_bytes"] == len(content)
    assert report["download_workers"] == 1
    assert report["sources"][0]["files"][0]["sha1"] == row["sha1"]


def test_download_inventory_rejects_invalid_worker_count(tmp_path) -> None:
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": INVENTORY_SCHEMA,
                "status": "complete",
                "mutable_latest_urls_forbidden": True,
                "sources": [
                    {
                        "name": "wikimedia_enwiki_20260701_v1",
                        "wiki": "enwiki",
                        "snapshot_date": "20260701",
                        "files": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="workers must be between"):
        mod.download_inventory(
            inventory_path=str(inventory),
            output_root=str(tmp_path / "raw"),
            report_path=str(tmp_path / "report.json"),
            workers=0,
        )


def test_download_workers_span_sources_and_preserve_source_order(tmp_path) -> None:
    payloads = [b"first-source", b"second-source"]
    sources = []
    for index, content in enumerate(payloads):
        filename = f"source-{index}.bz2"
        sources.append(
            {
                "name": f"source_{index}",
                "wiki": "zhwiki",
                "snapshot_date": "20260701",
                "files": [
                    {
                        "filename": filename,
                        "role": "articles",
                        "bytes": len(content),
                        "sha1": hashlib.sha1(
                            content, usedforsecurity=False
                        ).hexdigest(),
                        "md5": hashlib.md5(
                            content, usedforsecurity=False
                        ).hexdigest(),
                        "url": f"https://example.invalid/{filename}",
                    }
                ],
            }
        )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": INVENTORY_SCHEMA,
                "status": "complete",
                "mutable_latest_urls_forbidden": True,
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )

    def _download(*, row, target):
        source_index = int(str(row["filename"]).split("-")[1].split(".")[0])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[source_index])
        return mod._verify_file(target, row)

    report = mod.download_inventory(
        inventory_path=str(inventory),
        output_root=str(tmp_path / "raw"),
        report_path=str(tmp_path / "report.json"),
        downloader=_download,
        workers=2,
    )

    assert [source["name"] for source in report["sources"]] == [
        "source_0",
        "source_1",
    ]
    assert [source["files"][0]["filename"] for source in report["sources"]] == [
        "source-0.bz2",
        "source-1.bz2",
    ]
