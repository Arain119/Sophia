from __future__ import annotations

import json

from ml.core.engine.types import MetricsRow

def append_metrics_row(output_dir: str, row: MetricsRow) -> None:
    path = f"{str(output_dir)}/metrics.jsonl"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


__all__ = ["append_metrics_row"]
