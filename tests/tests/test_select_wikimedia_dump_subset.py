from __future__ import annotations

import json

import pytest

from ml.tooling.scripts.data import select_wikimedia_dump_subset as mod
from ml.tooling.scripts.data.snapshot_wikimedia_dump_inventory import INVENTORY_SCHEMA


def _inventory(tmp_path):
    files = []
    for shard_index in range(1, 6):
        for role in ("articles", "index"):
            files.append(
                {
                    "filename": f"{role}-{shard_index}",
                    "role": role,
                    "shard_index": shard_index,
                    "bytes": shard_index,
                    "sha1": "a" * 40,
                    "md5": "b" * 32,
                    "url": f"https://dumps.wikimedia.org/enwiki/20260701/{role}-{shard_index}",
                }
            )
    path = tmp_path / "full.json"
    path.write_text(
        json.dumps(
            {
                "schema": INVENTORY_SCHEMA,
                "status": "complete",
                "mutable_latest_urls_forbidden": True,
                "snapshot_date": "20260701",
                "sources": [
                    {
                        "name": "wikimedia_enwiki_20260701_v1",
                        "files": files,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_selects_complete_evenly_spaced_pairs(tmp_path) -> None:
    inventory = _inventory(tmp_path)
    report = mod.select_wikimedia_article_shards(
        inventory_path=str(inventory),
        source_name="wikimedia_enwiki_20260701_v1",
        article_shards=3,
        output_path=str(tmp_path / "subset.json"),
    )

    source = report["sources"][0]
    assert source["selected_article_shard_indexes"] == [1, 3, 5]
    assert source["article_shard_count"] == 3
    assert len(source["files"]) == 6
    assert {row["role"] for row in source["files"]} == {"articles", "index"}
    assert report["parent_inventory_sha256"]


def test_rejects_more_shards_than_the_parent_inventory(tmp_path) -> None:
    with pytest.raises(ValueError, match="article_shards"):
        mod.select_wikimedia_article_shards(
            inventory_path=str(_inventory(tmp_path)),
            source_name="wikimedia_enwiki_20260701_v1",
            article_shards=6,
            output_path=str(tmp_path / "subset.json"),
        )
