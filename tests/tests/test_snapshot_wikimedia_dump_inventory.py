from __future__ import annotations

import json

import pytest

from ml.tooling.scripts.data import snapshot_wikimedia_dump_inventory as mod


class _Response:
    def __init__(self, payload) -> None:
        self._payload = payload
        self.content = json.dumps(payload, sort_keys=True).encode()

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _Session:
    def __init__(self, payload) -> None:
        self.payload = payload

    def get(self, _url, **_kwargs):
        return _Response(self.payload)


def _status(
    *,
    state: str = "done",
    include_index: bool = True,
    mismatched_range: bool = False,
    wiki: str = "zhwiki",
    unsharded: bool = False,
):
    date = "20260701"
    article_name = (
        f"{wiki}-{date}-pages-articles-multistream.xml.bz2"
        if unsharded
        else f"{wiki}-{date}-pages-articles-multistream1.xml-p1p99.bz2"
    )
    files = {
        article_name: {
            "size": 11,
            "url": f"/{wiki}/{date}/{article_name}",
            "sha1": "a" * 40,
            "md5": "b" * 32,
        }
    }
    if include_index:
        if unsharded:
            name = f"{wiki}-{date}-pages-articles-multistream-index.txt.bz2"
        else:
            end = 100 if mismatched_range else 99
            name = (
                f"{wiki}-{date}-pages-articles-multistream-index1.txt-p1p{end}.bz2"
            )
        files[name] = {
            "size": 7,
            "url": f"/{wiki}/{date}/{name}",
            "sha1": "c" * 40,
            "md5": "d" * 32,
        }
    return {
        "version": "0.8",
        "jobs": {
            mod.MULTISTREAM_JOB: {
                "status": state,
                "updated": "2026-07-06 12:10:05",
                "files": files,
            }
        },
    }


def test_snapshot_pins_dated_article_dump_and_simplified_normalization(tmp_path) -> None:
    report = mod.snapshot_inventory(
        wikis=["zhwiki"],
        snapshot_date="20260701",
        output_path=str(tmp_path / "inventory.json"),
        session=_Session(_status()),
    )

    source = report["sources"][0]
    assert report["selected_file_count"] == 2
    assert report["selected_bytes"] == 18
    assert source["article_shard_count"] == 1
    assert {row["shard_index"] for row in source["files"]} == {1}
    assert {row["page_id_start"] for row in source["files"]} == {1}
    assert {row["page_id_end"] for row in source["files"]} == {99}
    assert source["output_language"] == "zh-Hans"
    assert source["script_normalization"].startswith("opencc_t2s")
    assert all("/20260701/" in row["url"] for row in source["files"])
    assert all("/latest/" not in row["url"] for row in source["files"])


def test_repeated_dump_part_is_split_by_page_id_range(tmp_path) -> None:
    payload = _status()
    files = payload["jobs"][mod.MULTISTREAM_JOB]["files"]
    for stem, digest in (
        ("multistream1.xml", "e"),
        ("multistream-index1.txt", "f"),
    ):
        name = f"zhwiki-20260701-pages-articles-{stem}-p100p199.bz2"
        files[name] = {
            "size": 5,
            "url": f"/zhwiki/20260701/{name}",
            "sha1": digest * 40,
            "md5": digest * 32,
        }
    report = mod.snapshot_inventory(
        wikis=["zhwiki"],
        snapshot_date="20260701",
        output_path=str(tmp_path / "inventory.json"),
        session=_Session(payload),
    )

    source = report["sources"][0]
    assert source["article_shard_count"] == 2
    assert [row["shard_index"] for row in source["files"]] == [1, 1, 2, 2]
    assert {row["dump_part"] for row in source["files"]} == {1}


def test_snapshot_supports_unsharded_chinese_sister_project(tmp_path) -> None:
    report = mod.snapshot_inventory(
        wikis=["zhwikibooks"],
        snapshot_date="20260701",
        output_path=str(tmp_path / "inventory.json"),
        session=_Session(_status(wiki="zhwikibooks", unsharded=True)),
    )

    source = report["sources"][0]
    assert source["article_shard_count"] == 1
    assert source["canonical_host"] == "zh.wikibooks.org"
    assert source["output_language"] == "zh-Hans"
    assert source["domain"] == "open_chinese_textbooks_and_humanities"
    assert {row["page_id_end"] for row in source["files"]} == {2**63 - 1}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_status(state="running"), "not complete"),
        (_status(include_index=False), "unpaired Wikimedia multistream shard"),
        (_status(mismatched_range=True), "unpaired Wikimedia multistream shard"),
    ],
)
def test_snapshot_fails_closed_for_incomplete_dump(payload, message, tmp_path) -> None:
    with pytest.raises(ValueError, match=message):
        mod.snapshot_inventory(
            wikis=["zhwiki"],
            snapshot_date="20260701",
            output_path=str(tmp_path / "inventory.json"),
            session=_Session(payload),
        )
