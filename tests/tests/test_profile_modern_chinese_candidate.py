from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ml.tooling.scripts.data import profile_modern_chinese_candidate as mod


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class _Tokenizer:
    def __call__(self, texts, **_kwargs):
        return {"length": [len(text) // 2 for text in texts]}


def test_profiles_distributed_script_quality_and_duplicates(
    tmp_path, monkeypatch
) -> None:
    source_name = "candidate"
    raw_source = tmp_path / "raw" / source_name
    raw_source.mkdir(parents=True)
    simplified = (
        "这是现代简体中文教育文本，介绍数学、科学与工程知识。"
        "课程先解释基本概念，再通过实验数据讨论不同方法的适用条件。"
        "最后给出可验证的推导过程和实际项目中的注意事项。"
    )
    traditional = (
        "這是現代繁體中文教育文本，介紹數學、科學與工程知識。"
        "課程先解釋基本概念，再通過實驗資料討論不同方法的適用條件。"
        "最後給出可驗證的推導過程和實際專案中的注意事項。"
    )
    files = []
    for index, rows in enumerate(
        (
            [
                {"text": simplified, "score": 0.9, "source": "CCI3"},
                {"text": traditional, "score": 0.8, "source": "CCI3"},
            ],
            [
                {"text": simplified, "score": 0.85, "source": "CCI3"},
                {"text": simplified + "补充。", "score": 0.95, "source": "CCI3"},
            ],
        )
    ):
        path = raw_source / f"{index:06d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    inventory = tmp_path / "inventory.json"
    _write_json(
        inventory,
        {
            "schema": mod.INVENTORY_SCHEMA,
            "sources": [
                {
                    "name": source_name,
                    "repo_id": "owner/repo",
                    "revision": "a" * 40,
                    "license": "reviewed",
                    "language": "zh",
                    "record_format": "parquet",
                    "text_column": "text",
                    "min_chars": 20,
                    "document_time_coverage": "unknown_no_timestamp",
                    "files": files,
                }
            ],
        },
    )
    policy = tmp_path / "policy.json"
    _write_json(
        policy,
        {
            "schema": "sophia_pretrain_data_admission_policy_v1",
            "allow_legacy_assets_as_training_input": False,
        },
    )
    monkeypatch.setattr(mod, "compute_tokenizer_bundle_sha1", lambda _path: "tokenizer")
    monkeypatch.setattr(
        mod, "load_local_tokenizer", lambda *_args, **_kwargs: _Tokenizer()
    )

    report = mod.profile_candidate(
        inventory_path=str(inventory),
        raw_root=str(tmp_path / "raw"),
        source_name=source_name,
        admission_policy=str(policy),
        tokenizer_path=str(tmp_path / "tokenizer"),
        output_path=str(tmp_path / "report.json"),
        sample_files=2,
        rows_per_file=2,
    )

    assert report["status"] == "complete"
    assert report["sampling"]["sampled_files"] == 2
    assert report["stats"]["clean_documents_profiled"] == 4
    assert report["stats"]["sample_exact_duplicates"] == 1
    assert report["script_profile"]["t2s_changed_positions"] > 0
    assert report["script_profile"]["s2t_changed_positions"] > 0
    assert report["document_time_coverage"] == "unknown_no_timestamp"
    assert report["upstream_score_quantiles"]["min"] == 0.8
    assert sum(report["quality_accepted_taxonomy_documents"].values()) == 4


def test_profiles_explicit_inventory_file_indexes(tmp_path, monkeypatch) -> None:
    source_name = "candidate"
    raw_source = tmp_path / "raw" / source_name
    raw_source.mkdir(parents=True)
    files = []
    for index in range(3):
        path = raw_source / f"{index:06d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [{"text": f"历史哲学教育材料第{index}篇，包含足够的正文内容。" * 8}]
            ),
            path,
        )
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    inventory = tmp_path / "inventory.json"
    _write_json(
        inventory,
        {
            "schema": mod.INVENTORY_SCHEMA,
            "sources": [
                {
                    "name": source_name,
                    "repo_id": "owner/repo",
                    "revision": "a" * 40,
                    "language": "zh",
                    "text_column": "text",
                    "min_chars": 20,
                    "files": files,
                }
            ],
        },
    )
    policy = tmp_path / "policy.json"
    _write_json(
        policy,
        {
            "schema": "sophia_pretrain_data_admission_policy_v1",
            "allow_legacy_assets_as_training_input": False,
        },
    )
    monkeypatch.setattr(mod, "compute_tokenizer_bundle_sha1", lambda _path: "tokenizer")
    monkeypatch.setattr(
        mod, "load_local_tokenizer", lambda *_args, **_kwargs: _Tokenizer()
    )

    report = mod.profile_candidate(
        inventory_path=str(inventory),
        raw_root=str(tmp_path / "raw"),
        source_name=source_name,
        admission_policy=str(policy),
        tokenizer_path=str(tmp_path / "tokenizer"),
        output_path=str(tmp_path / "report.json"),
        rows_per_file=1,
        selected_file_indexes=[0, 2],
    )

    assert report["sampling"]["method"] == "explicit_inventory_file_indexes"
    assert [row["inventory_index"] for row in report["sampling"]["files"]] == [0, 2]
