from __future__ import annotations

import json

import pytest

from ml.tooling.scripts.data import snapshot_pretrain_source_inventory as mod


class _Response:
    def __init__(self, payload, *, headers=None, status_code=200) -> None:
        self._payload = payload
        self.headers = dict(headers or {})
        self.status_code = int(status_code)

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _Session:
    def __init__(self, *, revision: str, gated=False) -> None:
        self.revision = revision
        self.gated = gated

    def get(self, url, **_kwargs):
        if "/tree/" in str(url):
            return _Response(
                [
                    {
                        "type": "file",
                        "path": "data/000.parquet",
                        "size": 7,
                        "oid": "git-oid",
                        "lfs": {"oid": "b" * 64, "size": 7},
                    }
                ]
            )
        return _Response(
            {
                "sha": self.revision,
                "gated": self.gated,
                "private": False,
                "disabled": False,
            }
        )


class _PagedSession(_Session):
    def get(self, url, **_kwargs):
        if "/tree" not in str(url):
            return super().get(url, **_kwargs)
        page = 2 if "cursor=second" in str(url) else 1
        start = 0 if page == 1 else 2
        stop = 2 if page == 1 else 5
        rows = [
            {
                "type": "file",
                "path": f"data/{index:03d}.parquet",
                "size": index + 1,
                "oid": f"git-{index}",
                "lfs": {"oid": f"{index + 1:064x}", "size": index + 1},
            }
            for index in range(start, stop)
        ]
        headers = (
            {"link": '<https://upstream.example/tree?cursor=second>; rel="next"'}
            if page == 1
            else {}
        )
        return _Response(rows, headers=headers)


def _selection(tmp_path, *, revision: str, repo_id: str = "owner/repo") -> str:
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "sophia_pretrain_data_admission_policy_v1",
                "allow_legacy_assets_as_training_input": False,
                "forbidden_upstream_repo_ids": ["legacy/repo"],
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "schema": mod.SELECTION_SCHEMA,
                "discovery_endpoint": "https://mirror.example",
                "admission_policy": str(policy),
                "sources": [
                    {
                        "name": "source",
                        "repo_id": repo_id,
                        "revision": revision,
                        "admission_status": "admitted",
                        "license": "reviewed-license",
                        "selectors": [
                            {
                                "path_prefix": "data",
                                "include_regex": r"\.parquet$",
                                "max_files": 1,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def test_snapshot_records_pinned_file_size_and_lfs_hash(tmp_path) -> None:
    revision = "a" * 40
    report = mod.snapshot_inventory(
        selection_path=_selection(tmp_path, revision=revision),
        output_path=str(tmp_path / "inventory.json"),
        session=_Session(revision=revision),
    )

    source = report["sources"][0]
    assert source["selected_file_count"] == 1
    assert source["selected_bytes"] == 7
    assert source["files"] == [
        {
            "path": "data/000.parquet",
            "bytes": 7,
            "sha256": "b" * 64,
            "git_oid": "git-oid",
        }
    ]
    assert source["catalog_file_count"] == 0


def test_snapshot_enumerates_full_catalog_before_even_sampling(tmp_path) -> None:
    revision = "a" * 40
    selection_path = _selection(tmp_path, revision=revision)
    selection = json.loads(open(selection_path, encoding="utf-8").read())
    selector = selection["sources"][0]["selectors"][0]
    selector.pop("max_files")
    selector.update(
        {
            "sampling": "evenly_spaced_full_catalog",
            "expected_files": 5,
            "sample_files": 3,
        }
    )
    with open(selection_path, "w", encoding="utf-8") as handle:
        json.dump(selection, handle)

    report = mod.snapshot_inventory(
        selection_path=selection_path,
        output_path=str(tmp_path / "inventory.json"),
        session=_PagedSession(revision=revision),
    )

    source = report["sources"][0]
    assert source["catalog_file_count"] == 5
    assert source["catalog_bytes"] == 15
    assert [row["path"] for row in source["catalog_files"]] == [
        f"data/{index:03d}.parquet" for index in range(5)
    ]
    assert [row["path"] for row in source["files"]] == [
        "data/000.parquet",
        "data/002.parquet",
        "data/004.parquet",
    ]
    assert source["selector_evidence"][0]["catalog_file_count"] == 5


def test_snapshot_forwards_recursive_tree_selection(tmp_path) -> None:
    revision = "a" * 40
    selection_path = _selection(tmp_path, revision=revision)
    selection = json.loads(open(selection_path, encoding="utf-8").read())
    selection["sources"][0]["selectors"][0]["recursive"] = True
    with open(selection_path, "w", encoding="utf-8") as handle:
        json.dump(selection, handle)

    class RecursiveSession(_Session):
        def get(self, url, **kwargs):
            if "/tree/" in str(url):
                assert kwargs["params"]["recursive"] == "true"
            return super().get(url, **kwargs)

    report = mod.snapshot_inventory(
        selection_path=selection_path,
        output_path=str(tmp_path / "inventory.json"),
        session=RecursiveSession(revision=revision),
    )

    assert report["sources"][0]["selector_evidence"][0]["recursive"] is True


def test_snapshot_selects_every_file_in_full_catalog_mode(tmp_path) -> None:
    revision = "a" * 40
    selection_path = _selection(tmp_path, revision=revision)
    selection = json.loads(open(selection_path, encoding="utf-8").read())
    selector = selection["sources"][0]["selectors"][0]
    selector.pop("max_files")
    selector.update({"sampling": "full_catalog", "expected_files": 5})
    with open(selection_path, "w", encoding="utf-8") as handle:
        json.dump(selection, handle)

    report = mod.snapshot_inventory(
        selection_path=selection_path,
        output_path=str(tmp_path / "inventory.json"),
        session=_PagedSession(revision=revision),
    )

    source = report["sources"][0]
    expected_paths = [f"data/{index:03d}.parquet" for index in range(5)]
    assert source["catalog_file_count"] == 5
    assert source["selected_file_count"] == 5
    assert [row["path"] for row in source["files"]] == expected_paths
    assert source["selector_evidence"] == [
        {
            "path_prefix": "data",
            "sampling": "full_catalog",
            "recursive": False,
            "catalog_complete": True,
            "catalog_file_count": 5,
            "catalog_bytes": 15,
            "selected_file_count": 5,
            "first_selected_path": "data/000.parquet",
            "last_selected_path": "data/004.parquet",
        }
    ]


def test_snapshot_rejects_revision_drift_and_gated_source(tmp_path) -> None:
    revision = "a" * 40
    selection = _selection(tmp_path, revision=revision)

    with pytest.raises(RuntimeError, match="pinned revision mismatch"):
        mod.snapshot_inventory(
            selection_path=selection,
            output_path=str(tmp_path / "drift.json"),
            session=_Session(revision="c" * 40),
        )
    with pytest.raises(RuntimeError, match="gated source is forbidden"):
        mod.snapshot_inventory(
            selection_path=selection,
            output_path=str(tmp_path / "gated.json"),
            session=_Session(revision=revision, gated="auto"),
        )


def test_snapshot_rejects_legacy_upstream_before_network_access(tmp_path) -> None:
    revision = "a" * 40
    with pytest.raises(ValueError, match="legacy upstream repository is forbidden"):
        mod.snapshot_inventory(
            selection_path=_selection(
                tmp_path,
                revision=revision,
                repo_id="LEGACY/REPO",
            ),
            output_path=str(tmp_path / "inventory.json"),
            session=_Session(revision=revision),
        )
