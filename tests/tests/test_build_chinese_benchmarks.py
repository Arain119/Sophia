from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ml.tooling.scripts.data import build_chinese_benchmarks as mod


def test_normalize_row_formats_choices() -> None:
    row = mod.normalize_row(
        benchmark="ceval",
        subject="math",
        split="val",
        row_id=3,
        row={"question": "一加一等于多少？", "A": "一", "B": "二", "answer": "b"},
        revision="a" * 40,
        license_name="test",
    )
    assert row is not None
    assert row["answer"] == "B"
    assert row["prompt"] == "一加一等于多少？\nA. 一\nB. 二"


def test_read_pinned_benchmark_layouts(tmp_path: Path) -> None:
    ceval = tmp_path / "ceval" / "math"
    ceval.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{"id": 1, "question": "题目", "A": "甲", "B": "乙", "answer": "A"}]
        ),
        ceval / "val-00000-of-00001.parquet",
    )
    cmmlu = tmp_path / "cmmlu" / "dev"
    cmmlu.mkdir(parents=True)
    with (cmmlu / "physics_dev.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Question", "A", "B", "Answer"])
        writer.writeheader()
        writer.writerow({"Question": "问题", "A": "甲", "B": "乙", "Answer": "B"})

    ceval_rows = list(mod.read_ceval(tmp_path / "ceval"))
    cmmlu_rows = list(mod.read_cmmlu(tmp_path / "cmmlu"))

    assert ceval_rows[0]["id"] == "ceval:math:val:1"
    assert cmmlu_rows[0]["id"] == "cmmlu:physics:dev:0"


def test_ceval_inventory_derives_pinned_source_paths(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "parquet_files": [
                    {"config": f"subject_{index}", "split": split}
                    for index in range(52)
                    for split in ("dev", "test", "val")
                ]
            }

    monkeypatch.setattr(mod.requests, "get", lambda *args, **kwargs: Response())

    paths = mod.ceval_file_inventory()

    assert len(paths) == 156
    assert "subject_0/dev/0000.parquet" in paths
