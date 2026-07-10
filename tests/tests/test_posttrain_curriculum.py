from __future__ import annotations

import json
from pathlib import Path

from ml.training.posttrain.curriculum import (
    default_posttrain_curriculum_stage,
    resolve_posttrain_max_seq_len,
    resolve_posttrain_stage_spec,
)
from ml.training.posttrain.defaults import DEFAULT_POSTTRAIN_CURRICULUM_PATH
from ml.tooling.pipelines.posttrain_length_curriculum import (
    write_posttrain_length_curriculum,
)


def test_resolve_posttrain_stage_spec_reads_stage_payload(tmp_path: Path) -> None:
    path = tmp_path / "curriculum.json"
    path.write_text(
        json.dumps(
            {
                "sft_curriculum": [
                    {
                        "stage": "sft_core",
                        "max_seq_len": 4096,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    stage = resolve_posttrain_stage_spec(
        stage_kind="sft",
        curriculum_path=str(path),
        curriculum_stage="sft_core",
    )

    assert stage is not None
    assert int(stage.max_seq_len) == 4096
    assert stage.selection == "all_examples"
    assert stage.crop_policy == "tail_tokens"


def test_resolve_posttrain_max_seq_len_prefers_curriculum_stage(tmp_path: Path) -> None:
    path = tmp_path / "curriculum.json"
    path.write_text(
        json.dumps(
            {
                "sft_curriculum": [
                    {"stage": "sft_core", "max_seq_len": 4096}
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    resolved = resolve_posttrain_max_seq_len(
        stage_kind="sft",
        requested_max_seq_len=2048,
        curriculum_path=str(path),
        curriculum_stage="sft_core",
    )

    assert int(resolved) == 4096

def test_default_posttrain_curriculum_helpers_match_release_chain() -> None:
    assert str(DEFAULT_POSTTRAIN_CURRICULUM_PATH) == "dataset/posttrain_length_curriculum.json"
    assert default_posttrain_curriculum_stage(stage_kind="sft") == "sft_core"


def test_write_posttrain_length_curriculum_builds_from_split_jsonl(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    sft_dir = dataset_root / "sft"
    sft_dir.mkdir(parents=True)
    for path in (
        sft_dir / "train.jsonl",
        sft_dir / "val.jsonl",
        sft_dir / "test.jsonl",
    ):
        path.write_text('{"messages":[{"role":"user","content":"hi"}]}\n', encoding="utf-8")

    generated_path = dataset_root / "posttrain_length_curriculum.json"
    write_posttrain_length_curriculum(
        sft_dir=sft_dir,
        output=generated_path,
    )

    stage = resolve_posttrain_stage_spec(
        stage_kind="sft",
        curriculum_path=str(generated_path),
        curriculum_stage=default_posttrain_curriculum_stage(stage_kind="sft"),
    )

    payload = json.loads(generated_path.read_text(encoding="utf-8"))
    assert stage is not None
    assert int(stage.max_seq_len) == 4096
    assert stage.selection == "all_examples"
    assert stage.crop_policy == "tail_tokens"
    assert payload["datasets"]["sft"]["rows_source"] == "split_jsonl"
    assert int(payload["datasets"]["sft"]["rows"]["total"]) == 3
    assert payload["sft_curriculum"][1]["stage"] == "sft_long_tail"
    assert payload["sft_curriculum"][1]["crop_policy"] == "preserve_final_turn"
