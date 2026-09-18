from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ml.tooling.scripts.data import build_pretrain_acquisition_bundle as bundle_mod
from ml.tooling.scripts.data import filter_chinese_humanities_derivative as mod


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _QualityProxy:
    enabled = True
    minimum_score = 0.68
    report_sha256 = "a" * 64
    model_sha256 = "b" * 64
    labels_sha256 = "c" * 64

    def score(self, texts: list[str]) -> list[float]:
        return [0.2 if "低质" in text else 0.9 for text in texts]


class _LegacyExclusion:
    action = "report"

    def classify_batch(self, rows):
        return [(False, False) for _row in rows]

    def close(self) -> None:
        return None


class _FailIfScoredQualityProxy(_QualityProxy):
    def score(self, texts: list[str]) -> list[float]:
        raise AssertionError(f"completed derivative was scored again: {len(texts)}")


def test_social_psychology_precision_gate_rejects_incidental_keywords() -> None:
    incidental = (
        "海洋动物昼夜迁移行为的卫星实验使用多个样本，并分析环境变化的相关性。"
        "研究团队据此解释海洋生态系统中的碳循环。"
    )
    focused = (
        "社会心理学研究关注群体中的从众行为。心理学家通过重复实验解释社会认同，"
        "并比较不同群体的态度变化。"
    )

    assert not mod._passes_humanities_precision_gate(
        text=incidental, taxonomy="social_psychology"
    )
    assert mod._passes_humanities_precision_gate(
        text=focused, taxonomy="social_psychology"
    )
    assert mod._passes_humanities_precision_gate(
        text=incidental, taxonomy="humanities_history"
    )


def test_social_psychology_precision_gate_rejects_generic_mental_health_mentions() -> None:
    text = (
        "教师职业道德需要加强，学校也要关注教师心理健康和家庭关系。"
        "文章建议完善监督机制，并再次强调心理健康的重要性。"
    )

    assert not mod._passes_humanities_precision_gate(
        text=text, taxonomy="social_psychology"
    )


def test_filters_quality_and_taxonomy_and_builds_verified_bundle(
    monkeypatch, tmp_path
) -> None:
    history = "".join(
        f"中国历史文化第{index}阶段的制度演变值得结合史料系统研究。"
        for index in range(20)
    )
    legal = "".join(
        f"法律法规第{index}条明确规定行政机关应当依法办理相关事项。"
        for index in range(20)
    )
    low_quality = "".join(
        f"低质中国历史材料第{index}段只有断言而没有可靠论述。"
        for index in range(20)
    )
    upstream_file = tmp_path / "raw" / "upstream" / "3_4" / "000000.parquet"
    upstream_file.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"text": history, "score": 0.75, "source": "CCI3"},
                {"text": legal, "score": 0.72, "source": "CCI3"},
                {"text": low_quality, "score": 0.71, "source": "CCI3"},
            ]
        ),
        upstream_file,
    )
    upstream_row = {
        "path": "3_4/000000.parquet",
        "bytes": upstream_file.stat().st_size,
        "sha256": _sha256(upstream_file),
    }
    upstream_source = {
        "name": "upstream",
        "repo_id": "owner/repo",
        "revision": "d" * 40,
        "license": "reviewed-research-use",
        "language": "zh",
        "domain": "candidate",
        "record_format": "parquet",
        "text_column": "text",
        "min_chars": 100,
        "files": [upstream_row],
    }
    inventory = tmp_path / "inventory.json"
    _write_json(
        inventory,
        {"schema": mod.INVENTORY_SCHEMA, "sources": [upstream_source]},
    )
    acquisition = tmp_path / "raw" / "acquisition_report.json"
    _write_json(
        acquisition,
        {
            "schema": mod.ACQUISITION_SCHEMA,
            "inventory_path": str(inventory),
            "inventory_sha256": _sha256(inventory),
            "output_root": str(tmp_path / "raw"),
            "sources": [
                {
                    **upstream_source,
                    "files": [
                        {
                            "filename": upstream_row["path"],
                            "path": str(upstream_file),
                            "bytes": upstream_row["bytes"],
                            "sha256": upstream_row["sha256"],
                        }
                    ],
                }
            ],
        },
    )
    policy = tmp_path / "policy.json"
    _write_json(policy, {"schema": "test-policy"})
    monkeypatch.setattr(mod, "_load_chinese_quality_proxy", lambda **_kwargs: _QualityProxy())
    monkeypatch.setattr(mod, "_load_legacy_content_exclusion", lambda **_kwargs: _LegacyExclusion())
    monkeypatch.setattr(
        mod.taxonomy_mod,
        "infer_user_taxonomy_label",
        lambda text: "legal_statute" if "法律法规" in text else "humanities_history",
    )
    derivative_root = tmp_path / "derivative"
    derivative_report = derivative_root / "derivative_report.json"

    report = mod.filter_humanities_derivative(
        acquisition_report=str(acquisition),
        source_name="upstream",
        admission_policy=str(policy),
        output_root=str(derivative_root),
        report_path=str(derivative_report),
        batch_size=128,
    )

    assert report["status"] == "complete"
    assert report["stats"]["documents_out"] == 1
    assert report["stats"]["drop_non_humanities_taxonomy"] == 1
    assert report["stats"]["drop_chinese_quality_proxy"] == 1
    output = pq.read_table(report["files"][0]["path"]).to_pylist()
    assert output[0]["text"] == history
    assert output[0]["upstream_row"] == 0
    assert output[0]["upstream_file"] == "3_4/000000.parquet"
    assert output[0]["upstream_file_sha256"] == upstream_row["sha256"]
    assert output[0]["upstream_text_sha256"] == hashlib.sha256(
        history.encode("utf-8")
    ).hexdigest()
    assert output[0]["humanities_taxonomy"] == "humanities_history"

    monkeypatch.setattr(
        mod,
        "_load_chinese_quality_proxy",
        lambda **_kwargs: _FailIfScoredQualityProxy(),
    )
    resumed = mod.filter_humanities_derivative(
        acquisition_report=str(acquisition),
        source_name="upstream",
        admission_policy=str(policy),
        output_root=str(derivative_root),
        report_path=str(derivative_report),
        batch_size=128,
    )
    assert resumed["files"][0]["sha256"] == report["files"][0]["sha256"]

    bundle = bundle_mod.build_bundle(
        hf_acquisition_reports=[],
        wikimedia_extraction_reports=[],
        derivative_reports=[str(derivative_report)],
        output_root=str(tmp_path),
        inventory_output=str(tmp_path / "bundle_inventory.json"),
        acquisition_output=str(tmp_path / "bundle_acquisition.json"),
    )
    assert bundle["status"] == "complete"
    assert bundle["sources"][0]["files"][0]["rows"] == 1
    assert bundle["sources"][0]["upstream_source_name"] == "upstream"
