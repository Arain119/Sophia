from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ml.tooling.scripts.data import build_pretrain_decontamination_benchmarks as mod
from ml.tooling.scripts.data.download_hf_source_inventory import sha256_file


class _Response:
    def __init__(self, *, payload=None, content: bytes = b"", status_code: int = 200):
        self.payload = payload
        self.content = content
        self.status_code = int(status_code)

    def json(self):
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def iter_content(self, chunk_size: int):
        for offset in range(0, len(self.content), int(chunk_size)):
            yield self.content[offset : offset + int(chunk_size)]

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


class _Session:
    def __init__(self, *, revision: str, parquet_bytes: bytes):
        self.revision = revision
        self.parquet_bytes = parquet_bytes

    def get(self, url, **_kwargs):
        if "/api/datasets/" in str(url):
            return _Response(
                payload={
                    "sha": self.revision,
                    "private": False,
                    "disabled": False,
                    "gated": False,
                }
            )
        if str(url) == "https://datasets.example/parquet":
            return _Response(
                payload={
                    "parquet_files": [
                        {
                            "config": "main",
                            "split": "test",
                            "url": "https://files.example/test.parquet",
                            "size": len(self.parquet_bytes),
                        }
                    ],
                    "failed": [],
                }
            )
        return _Response(content=self.parquet_bytes)


def _parquet_bytes(rows: list[dict[str, object]]) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer)
    return buffer.getvalue()


def test_snapshot_downloads_pinned_benchmark_and_records_hash(tmp_path: Path) -> None:
    revision = "a" * 40
    content = _parquet_bytes([{"question": "A sufficiently long benchmark question?"}])
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": mod.SELECTION_SCHEMA,
                "metadata_endpoint": "https://metadata.example",
                "parquet_api": "https://datasets.example/parquet",
                "sources": [
                    {
                        "name": "gsm8k",
                        "repo_id": "owner/repo",
                        "revision": revision,
                        "license": "mit",
                        "selections": [{"config": "main", "splits": ["test"]}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    inventory_path = tmp_path / "inventory.json"
    inventory = mod.snapshot_and_download(
        selection_path=str(selection),
        raw_root=str(tmp_path / "raw"),
        inventory_output=str(inventory_path),
        session=_Session(revision=revision, parquet_bytes=content),
    )

    file_info = inventory["sources"][0]["files"][0]
    assert inventory["files"] == 1
    assert file_info["sha256"] == sha256_file(file_info["path"])
    assert Path(file_info["path"]).read_bytes() == content

    with pytest.raises(RuntimeError, match="revision drift"):
        mod.snapshot_and_download(
            selection_path=str(selection),
            raw_root=str(tmp_path / "drift_raw"),
            inventory_output=str(tmp_path / "drift.json"),
            session=_Session(revision="b" * 40, parquet_bytes=content),
        )


@pytest.mark.parametrize(
    ("name", "row", "expected"),
    [
        (
            "gsm8k",
            {"question": "Janet has sixteen eggs and sells the remaining eggs every day. How much does she earn?"},
            "Janet has sixteen eggs",
        ),
        (
            "math",
            {"problem": "Find all real values of x satisfying the following sufficiently long polynomial equation.", "type": "Algebra"},
            "Find all real values",
        ),
        (
            "mmlu",
            {"question": "Which answer best explains this sufficiently detailed scientific observation?", "choices": ["First choice", "Second choice"]},
            "A. First choice",
        ),
        (
            "arc",
            {"question": "Which answer best explains this sufficiently detailed scientific observation?", "choices": {"label": ["A", "B"], "text": ["First choice", "Second choice"]}},
            "B. Second choice",
        ),
        (
            "hellaswag",
            {"ctx": "A person begins a detailed multi-step household activity and then", "endings": ["finishes the first plausible action", "does something unrelated"]},
            "A. finishes",
        ),
        (
            "humaneval",
            {"prompt": "def solve(values):\n    \"\"\"Return the correct result for this sufficiently detailed programming task.\"\"\"\n", "entry_point": "solve"},
            "def solve",
        ),
    ],
)
def test_benchmark_signature_normalizes_supported_suites(
    name: str, row: dict[str, object], expected: str
) -> None:
    result = mod.benchmark_signature(name=name, row=row)
    assert result is not None
    assert expected in result[0]


def test_build_signatures_merges_existing_and_deduplicates(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rows_by_source = {
        "gsm8k": [{"question": "A long unique grade school mathematics word problem asks for the final numerical answer."}],
        "math": [{"problem": "A long unique competition mathematics problem asks for a rigorous symbolic solution.", "level": "Level 3", "type": "Algebra"}],
        "mmlu": [{"question": "A long unique knowledge question asks which explanation is correct.", "choices": ["Choice one", "Choice two"], "subject": "physics"}],
        "arc": [{"question": "A long unique science question asks which observation is correct.", "choices": {"label": ["A", "B"], "text": ["Choice one", "Choice two"]}}],
        "hellaswag": [{"ctx": "A long unique commonsense activity begins with a person preparing dinner.", "endings": ["They finish cooking.", "They fly away."]}],
        "humaneval": [{"prompt": "def unique_programming_task(values):\n    \"\"\"Solve this long unique programming benchmark task correctly.\"\"\"\n", "entry_point": "unique_programming_task"}],
    }
    sources = []
    for name, rows in rows_by_source.items():
        path = raw / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        sources.append(
            {
                "name": name,
                "revision": name.ljust(40, "0")[:40],
                "license": "test-license",
                "files": [
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "config": "default",
                        "split": "test",
                    }
                ],
            }
        )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps({"schema": mod.INVENTORY_SCHEMA, "sources": sources}),
        encoding="utf-8",
    )
    existing = tmp_path / "existing.jsonl"
    existing.write_text(
        json.dumps(
            {
                "id": "ceval:one",
                "benchmark": "ceval",
                "prompt": "这是一条长度足够且唯一的中文评测题目，用于验证训练语料去污染流程是否完整，并确保不会被错误忽略。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = mod.build_signatures(
        inventory_path=str(inventory),
        existing_signatures=[str(existing)],
        output_root=str(tmp_path / "output"),
    )

    assert report["status"] == "complete"
    assert report["signatures"] == 7
    assert set(report["counts"]) == {*rows_by_source, "ceval"}
    assert sha256_file(report["signature_path"]) == report["signature_sha256"]
