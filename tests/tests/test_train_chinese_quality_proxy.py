from __future__ import annotations

import json

import pytest

from ml.tooling.scripts.data import train_chinese_quality_proxy as mod


def _write_labels(path) -> None:
    rows = []
    for index in range(80):
        rows.append(
            {
                "text": f"第{index}篇教材内容，介绍代数定理、证明步骤与练习题。",
                "label": 4,
            }
        )
        rows.append(
            {
                "text": f"第{index}条促销广告，限时抢购，点击领取优惠券。",
                "label": 0,
            }
        )
    rows.append(dict(rows[0]))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_trains_hash_pinned_proxy_with_disjoint_splits(tmp_path) -> None:
    labels = tmp_path / "labels.jsonl"
    model = tmp_path / "model.pkl"
    report_path = tmp_path / "report.json"
    _write_labels(labels)

    report = mod.train_quality_proxy(
        labels_path=str(labels),
        model_path=str(model),
        report_path=str(report_path),
        repo_id="example/labels",
        revision="a" * 40,
        repository_license="apache-2.0",
        c_values=(4.0,),
        minimum_validation_precision=0.5,
        minimum_test_macro_f1=0.5,
        minimum_test_positive_precision=0.5,
        minimum_test_positive_recall=0.5,
    )

    assert report["status"] == "pass"
    assert report["label_stats"]["exact_duplicate_rows_removed"] == 1
    assert sum(split["rows"] for split in report["splits"].values()) == 160
    assert len({split["sha256"] for split in report["splits"].values()}) == 3
    artifact = mod.load_quality_proxy(
        model,
        expected_sha256=report["model"]["sha256"],
    )
    scores = mod.score_quality_texts(
        artifact,
        ["教材讲解几何证明和公式。", "促销广告点击领取优惠券。"],
    )
    assert scores.shape == (2,)
    assert scores[0] > scores[1]
    assert json.loads(report_path.read_text())["schema"] == mod.REPORT_SCHEMA


def test_rejects_conflicting_normalized_duplicate_labels(tmp_path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "{\"text\": \"相 同\", \"label\": 0}\n"
        "{\"text\": \"相同\", \"label\": 4}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting labels"):
        mod._load_deduplicated_labels(labels)


def test_model_loader_requires_exact_sha256(tmp_path) -> None:
    path = tmp_path / "model.pkl"
    path.write_bytes(b"not a model")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        mod.load_quality_proxy(path, expected_sha256="0" * 64)
