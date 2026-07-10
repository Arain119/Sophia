from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.tooling.core.json_io import load_json_dict
from ml.data.jsonl_stream import iter_jsonl_objects
from ml.training.posttrain.defaults import RELEASE_SFT_CURRICULUM_STAGE


def _report_rows(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("rows")
    if isinstance(rows, dict):
        return rows
    dataset = report.get("dataset")
    if isinstance(dataset, dict):
        dataset_rows = dataset.get("rows")
        if isinstance(dataset_rows, dict):
            return dataset_rows
    return {}


def _split_rows(dataset_dir: Path) -> tuple[dict[str, int], str]:
    counts: dict[str, int] = {}
    saw_any_split = False
    for split in ("train", "val", "test"):
        split_path = dataset_dir / f"{split}.jsonl"
        if not split_path.is_file():
            continue
        saw_any_split = True
        counts[split] = sum(1 for _ in iter_jsonl_objects(str(split_path)))
    if not saw_any_split:
        return {}, "report_json"
    counts["total"] = int(sum(counts.values()))
    return counts, "split_jsonl"


def build_posttrain_length_curriculum(
    *,
    sft_dir: Path,
) -> dict[str, Any]:
    sft_rows, sft_rows_source = _split_rows(sft_dir)
    if not sft_rows:
        sft_report = load_json_dict(
            sft_dir / "report.json",
            required=True,
            label="posttrain report",
        )
        assert sft_report is not None
        sft_rows = _report_rows(sft_report)
    return {
        "kind": "posttrain_length_curriculum",
        "file_role": "generated_curriculum_plan",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal": (
            "Keep post-training faithful to the original SFT examples while "
            "shaping the decoder around its configured context window."
        ),
        "model_context": {
            "max_seq_len": 4096,
            "context_required": True,
            "source": "canonical pretrain model config",
        },
        "datasets": {
            "sft": {
                "dir": str(sft_dir),
                "rows": sft_rows,
                "rows_source": sft_rows_source,
      "role": "original supervised conversations aligned to the configured context window",
            },
        },
        "sft_curriculum": [
            {
                "stage": RELEASE_SFT_CURRICULUM_STAGE,
                "max_seq_len": 4096,
                "data": "dataset/sft/train.jsonl",
                "selection": "all_examples",
                "crop_policy": "tail_tokens",
                "recommended_share_of_updates": 0.92,
                "purpose": "main supervised behavior shaping under the decoder context budget",
            },
            {
                "stage": "sft_long_tail",
                "max_seq_len": 4096,
                "data": "dataset/sft/train.jsonl",
                "selection": "long_examples_only",
                "crop_policy": "preserve_final_turn",
                "recommended_share_of_updates": 0.08,
                "purpose": "soft long-sample stage that preserves the final supervised turn when examples exceed the context window",
            },
        ],
        "training_policy": {
            "context_policy": "context_window",
            "reason": (
                "The dataset policy prioritizes original data fidelity. The decoder is "
                "trained at the configured context length; post-training reinforces that "
                "budget directly while the soft long-SFT stage preserves the final "
                "supervised turn on overflow examples."
            ),
            "minimum_acceptance": {},
        },
    }


def write_posttrain_length_curriculum(
    *,
    sft_dir: Path,
    output: Path,
) -> dict[str, Any]:
    plan = build_posttrain_length_curriculum(
        sft_dir=sft_dir,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan
