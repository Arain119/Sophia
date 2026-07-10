from __future__ import annotations

import json
import sys
from pathlib import Path

from ml.tooling.scripts.data.production.profile_posttrain_lengths import main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_profile_posttrain_lengths_supports_sft(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    output = tmp_path / "profile.json"
    row = {
        "messages": [
            {"role": "user", "content": "Say this more warmly."},
            {"role": "assistant", "content": "I really appreciate it. Thank you."},
        ],
        "metadata": {
            "lang": "en",
            "capability_family": "humanistic",
            "bucket": "en_chat",
        },
    }
    _write_jsonl(dataset_root / "sft" / "train.jsonl", [row])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_posttrain_lengths.py",
            "--dataset_root",
            str(dataset_root),
            "--output",
            str(output),
            "--mode",
            "chars",
            "--datasets",
            "sft",
            "--splits",
            "train",
        ],
    )
    main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["datasets"]["sft"]["train"]["rows"] == 1
    assert report["datasets"]["sft"]["train"]["avg"] > 0
