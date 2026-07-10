from __future__ import annotations

import json
import sys

import pytest

from ml.tooling.scripts.machine_recipes import export_machine_recipe as mod
from ml.errors import SophiaUsageError


def _write_export_dir(path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "chat_template.jinja").write_text("", encoding="utf-8")


def _write_run_args(
    path,
    *,
    export_dir,
    train_data,
    eval_data,
    test_data,
) -> None:
    path.write_text(
        json.dumps(
            {
                "_sophia_machine_signature": {
                    "schema": "posttrain_machine_adaptive_v1",
                    "device_type": "cuda",
                    "torch": "2.8.0",
                    "cuda": "12.8",
                    "platform": "linux",
                    "machine": "x86_64",
                    "cuda_device_name": "NVIDIA Test",
                    "cuda_capability": [9, 0],
                    "cuda_total_memory_gb": 80.0,
                },
                "export_dir": str(export_dir),
                "train_data": str(train_data),
                "eval_data": str(eval_data),
                "test_data": str(test_data),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_export_sft_machine_recipe_from_summary(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    out = tmp_path / "recipe.json"
    export_dir = tmp_path / "export"
    run_dir = tmp_path / "release_sft"
    train_data = tmp_path / "train.jsonl"
    eval_data = tmp_path / "eval.jsonl"
    test_data = tmp_path / "test.jsonl"
    _write_export_dir(export_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for path in (train_data, eval_data, test_data):
        path.write_text('{"messages":[{"role":"user","content":"hi"}]}\n', encoding="utf-8")
    _write_run_args(
        run_dir / "run_args.json",
        export_dir=export_dir,
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
    )
    summary.write_text(
        json.dumps(
            {
                "kind": "sft_machine_selection",
                "selected_runs": [
                    {
                        "name": "release_sft",
                        "release_semantics": {"target_examples_per_update": 16},
                        "machine_selected": {
                            "batch_size": 4,
                            "accumulation_steps": 4,
                            "gradient_checkpointing": 1,
                        },
                        "output_dir": str(run_dir),
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    sys.argv = [
        "export_machine_recipe.py",
        "--summary_json",
        str(summary),
        "--output_json",
        str(out),
        "--stage",
        "sft",
    ]
    mod.main()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["stage"] == "sft"
    assert payload["release_semantics"] == {"target_examples_per_update": 16}
    assert payload["machine_recipe"] == {
        "batch_size": 4,
        "accumulation_steps": 4,
        "gradient_checkpointing": 1,
    }
    assert payload["machine_signature"]["schema"] == "posttrain_machine_adaptive_v1"
    assert payload["export_signature"]["export_dir"] == str(export_dir.resolve())
    assert payload["data_signature"]["train"]["path"] == str(train_data.resolve())


def test_export_machine_recipe_requires_signed_run_args(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    out = tmp_path / "recipe.json"
    run_dir = tmp_path / "release_sft"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "kind": "sft_machine_selection",
                "selected_runs": [
                    {
                        "name": "release_sft",
                        "release_semantics": {"target_examples_per_update": 16},
                        "machine_selected": {
                            "batch_size": 4,
                            "accumulation_steps": 4,
                            "gradient_checkpointing": 1,
                        },
                        "output_dir": str(run_dir),
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    sys.argv = [
        "export_machine_recipe.py",
        "--summary_json",
        str(summary),
        "--output_json",
        str(out),
        "--stage",
        "sft",
    ]

    with pytest.raises(SophiaUsageError, match="run_args.json"):
        mod.main()
