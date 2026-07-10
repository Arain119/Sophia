from __future__ import annotations

import json
from pathlib import Path

from ml.tooling.scripts.pretrain_stability_probe import summarize_metrics


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_summarize_metrics_passes_stable_run(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics,
        [
            {"type": "start", "max_steps": 3},
            {"type": "train", "step": 1, "max_steps": 3, "loss": 11.0, "grad_norm": 6.0, "tok_s": 100.0},
            {"type": "train", "step": 2, "max_steps": 3, "loss": 10.8, "grad_norm": 5.0, "tok_s": 110.0},
            {"type": "train", "step": 3, "max_steps": 3, "loss": 10.7, "grad_norm": 4.0, "tok_s": 120.0},
            {"type": "end", "step": 3, "max_steps": 3},
        ],
    )

    summary = summarize_metrics(metrics)

    assert summary["passed"] is True
    assert summary["max_loss"] == 11.0
    assert summary["max_grad_norm"] == 6.0


def test_summarize_metrics_flags_loss_and_grad_thresholds(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics,
        [
            {"type": "start", "max_steps": 4},
            {"type": "train", "step": 1, "max_steps": 4, "loss": 11.0, "grad_norm": 21.0},
            {"type": "train", "step": 2, "max_steps": 4, "loss": 13.5, "grad_norm": 22.0},
            {"type": "train", "step": 3, "max_steps": 4, "loss": 12.0, "grad_norm": 23.0},
            {"type": "train", "step": 4, "max_steps": 4, "loss": 12.0, "grad_norm": 51.0},
            {"type": "end", "step": 4, "max_steps": 4},
        ],
    )

    summary = summarize_metrics(metrics)

    assert summary["passed"] is False
    assert summary["loss_gt_threshold_steps"] == [2]
    assert summary["grad_gt_threshold_steps"] == [4]
    assert summary["sustained_grad_threshold_steps"] == [1, 2]
