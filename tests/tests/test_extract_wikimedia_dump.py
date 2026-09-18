from __future__ import annotations

import bz2
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from ml.tooling.scripts.data import extract_wikimedia_dump as mod
from ml.tooling.scripts.data.snapshot_wikimedia_dump_inventory import INVENTORY_SCHEMA


def _hashes(content: bytes) -> tuple[str, str]:
    return (
        hashlib.sha1(content, usedforsecurity=False).hexdigest(),
        hashlib.md5(content, usedforsecurity=False).hexdigest(),
    )


def _file_row(filename: str, role: str, content: bytes) -> dict[str, object]:
    sha1, md5 = _hashes(content)
    return {
        "filename": filename,
        "role": role,
        "shard_index": 1,
        "page_id_start": 1,
        "page_id_end": 99,
        "bytes": len(content),
        "sha1": sha1,
        "md5": md5,
        "url": f"https://dumps.wikimedia.org/zhwiki/20260701/{filename}",
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    source_name = "wikimedia_zhwiki_20260701_v1"
    article_name = "zhwiki-20260701-pages-articles-multistream1.xml-p1p99.bz2"
    index_name = "zhwiki-20260701-pages-articles-multistream-index1.txt-p1p99.bz2"
    xml = b"""<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
      <page><title>Traditional page</title><ns>0</ns><id>2</id>
        <revision><id>20</id><timestamp>2026-07-01T00:00:00Z</timestamp>
          <text xml:space="preserve">'''\xe7\xb9\x81\xe9\xab\x94'''\xe8\x88\x87[[\xe8\xb3\x87\xe6\x96\x99|\xe8\xb3\x87\xe6\x96\x99\xe9\x9b\x86]]</text>
        </revision>
      </page>
      <page><title>Redirect</title><ns>0</ns><id>3</id><redirect title="Target" />
        <revision><id>30</id><timestamp>2026-07-01T00:00:00Z</timestamp><text>#REDIRECT</text></revision>
      </page>
      <page><title>Talk</title><ns>1</ns><id>4</id>
        <revision><id>40</id><timestamp>2026-07-01T00:00:00Z</timestamp><text>ignored</text></revision>
      </page>
    </mediawiki>"""
    article_content = bz2.compress(xml)
    index_content = bz2.compress(b"0:2:Traditional page\n")
    article = _file_row(article_name, "articles", article_content)
    index = _file_row(index_name, "index", index_content)
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "status": "complete",
        "mutable_latest_urls_forbidden": True,
        "sources": [
            {
                "name": source_name,
                "wiki": "zhwiki",
                "language": "zh",
                "output_language": "zh-Hans",
                "snapshot_date": "20260701",
                "canonical_host": "zh.wikipedia.org",
                "script_normalization": "opencc_t2s_after_wikitext_extraction",
                "files": [article, index],
            }
        ],
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    raw_source = tmp_path / "raw" / source_name
    raw_source.mkdir(parents=True)
    (raw_source / article_name).write_bytes(article_content)
    (raw_source / index_name).write_bytes(index_content)
    return inventory_path, tmp_path / "raw", source_name


def test_extracts_main_namespace_and_preserves_normalization_lineage(tmp_path) -> None:
    inventory, raw, source_name = _fixture(tmp_path)
    output = tmp_path / "extracted"
    report_path = tmp_path / "report.json"

    report = mod.extract_inventory(
        inventory_path=str(inventory),
        raw_root=str(raw),
        output_root=str(output),
        report_path=str(report_path),
    )

    assert report["pages_in"] == 3
    assert report["pages_out"] == 1
    shard = report["sources"][0]["shards"][0]
    assert shard["stats"]["drop_redirect"] == 1
    assert shard["stats"]["drop_namespace"] == 1
    parquet_path = output / source_name / "shard-00001" / "data.parquet"
    rows = pq.read_table(parquet_path).to_pylist()
    assert len(rows) == 1
    assert rows[0]["text"] == "繁体与资料集"
    assert rows[0]["url"].endswith("/Traditional_page")
    assert rows[0]["raw_wikitext_sha256"] != rows[0]["normalized_text_sha256"]
    assert rows[0]["revision_id"] == 20

    cached = mod.extract_inventory(
        inventory_path=str(inventory),
        raw_root=str(raw),
        output_root=str(output),
        report_path=str(report_path),
    )
    assert cached["pages_out"] == 1


def test_sister_project_uses_its_canonical_host(tmp_path) -> None:
    inventory, raw, source_name = _fixture(tmp_path)
    payload = json.loads(inventory.read_text(encoding="utf-8"))
    payload["sources"][0]["wiki"] = "zhwikibooks"
    payload["sources"][0]["canonical_host"] = "zh.wikibooks.org"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    report = mod.extract_inventory(
        inventory_path=str(inventory),
        raw_root=str(raw),
        output_root=str(tmp_path / "sister-extracted"),
        report_path=str(tmp_path / "sister-report.json"),
    )

    shard = report["sources"][0]["shards"][0]
    rows = pq.read_table(
        tmp_path
        / "sister-extracted"
        / source_name
        / f"shard-{shard['shard_index']:05d}"
        / "data.parquet"
    ).to_pylist()
    assert rows[0]["url"].startswith("https://zh.wikibooks.org/wiki/")


def test_available_only_publishes_partial_report_and_skips_missing_pair(tmp_path) -> None:
    inventory_path, raw, source_name = _fixture(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    source = inventory["sources"][0]
    article_content = bz2.compress(b"<mediawiki />")
    index_content = bz2.compress(b"")
    article = _file_row("missing-articles.bz2", "articles", article_content)
    index = _file_row("missing-index.bz2", "index", index_content)
    article["shard_index"] = 2
    index["shard_index"] = 2
    source["files"].extend((article, index))
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    output = tmp_path / "extracted"
    report = mod.extract_inventory(
        inventory_path=str(inventory_path),
        raw_root=str(raw),
        output_root=str(output),
        report_path=str(tmp_path / "partial.json"),
        available_only=True,
    )

    assert report["status"] == "partial"
    assert report["inventory_shard_count"] == 2
    assert report["shard_count"] == 1
    assert report["sources"][0]["unavailable_shard_count"] == 1
    assert report["sources"][0]["complete"] is False
    assert (output / source_name / "shard-00001" / "data.parquet").is_file()
    assert not (output / source_name / "shard-00002").exists()
