from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ml.tooling.scripts.data import build_tokenizer_training_sample as mod


def test_integer_byte_budgets_are_exact_and_follow_token_quotas() -> None:
    budgets = mod._integer_byte_budgets(
        total_bytes=10, source_quotas={"zh": 7, "en": 3}
    )

    assert budgets == {"en": 3, "zh": 7}
    assert sum(budgets.values()) == 10


def test_evenly_spaced_files_include_both_ends(tmp_path: Path) -> None:
    paths = [tmp_path / f"part-{index:02d}.parquet" for index in range(10)]

    selected = mod._evenly_spaced_files(paths, 3)

    assert selected == [paths[0], paths[4], paths[9]]


def _write_clean_file(path: Path, *, source: str, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "text": str(row["text"]),
                    "source": source,
                    "legacy_exact_overlap": bool(row.get("legacy_exact_overlap", False)),
                }
                for row in rows
            ]
        ),
        path,
    )


def test_build_sample_applies_admission_and_writes_exact_mix(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    _write_clean_file(
        clean / "zh" / "train" / "part.parquet",
        source="zh",
        rows=[
            {"text": "drop-me", "legacy_exact_overlap": True},
            {"text": "abcdefghij"},
        ],
    )
    _write_clean_file(
        clean / "en" / "train" / "part.parquet",
        source="en",
        rows=[{"text": "klmnopqrst"}],
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "sophia_resolved_pretrain_mix_plan_v1",
                "status": "ready",
                "source_train_quotas": {"zh": 7, "en": 3},
            }
        ),
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "post_clean_document_admission": {
                    "exclude_legacy_exact_overlap": True,
                    "exclude_legacy_near_overlap": False,
                    "repeated_sentence_rules": {},
                }
            }
        ),
        encoding="utf-8",
    )

    report = mod.build_sample(
        clean_root=clean,
        resolved_mix_plan_path=plan_path,
        admission_policy_path=policy_path,
        output_root=tmp_path / "sample",
        sample_bytes=10,
        max_text_chars=100,
        max_files_per_source=10,
        batch_size=2,
    )

    assert report["status"] == "complete"
    assert report["sample_utf8_bytes"] == 10
    assert report["source_byte_budgets"] == {"en": 3, "zh": 7}
    assert report["sources"]["zh"]["documents_excluded_by"] == {
        "legacy_exact_overlap": 1
    }
    assert pq.read_table(
        tmp_path / "sample" / "zh" / "train" / "part-00000.parquet"
    ).column("text").to_pylist() == ["abcdefg"]
