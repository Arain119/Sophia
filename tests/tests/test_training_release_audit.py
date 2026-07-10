from __future__ import annotations

import json
from pathlib import Path

from ml.tooling.scripts.release_gate import audit_training_release as mod


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix:
        path.write_text("", encoding="utf-8")
    else:
        path.mkdir(parents=True, exist_ok=True)


def _prepare_complete_release(tmp_path: Path, *, sft_metrics: str | None = None) -> dict[str, Path]:
    pretrain_recipe_root = tmp_path / "pretrain_recipe"
    sft_recipe_root = tmp_path / "sft_recipe"
    pretrain_output = tmp_path / "release_pretrain"
    sft_output = tmp_path / "release_sft"

    _write_json(
        pretrain_recipe_root / "report.json",
        {
            "machine_baseline_table": [
                {
                    "seq_len": 4096,
                    "backend": "inductor",
                    "batch_size": 2,
                    "accumulation_steps": 4,
                    "gradient_checkpointing": 1,
                    "loss_chunk_size": 256,
                    "tokens_per_s": 1000.0,
                    "step_time_s": 1.0,
                    "max_mem_gb": 20.0,
                }
            ]
        },
    )
    _write_json(pretrain_recipe_root / "release_pretrain_machine_recipe.json", {"ok": True})
    _write_json(
        sft_recipe_root / "summary.json",
        {
            "machine_baseline_table": [
                {
                    "batch_size": 16,
                    "accumulation_steps": 1,
                    "gradient_checkpointing": 0,
                    "loss": 1.0,
                    "val_test": 1.1,
                    "speed": 10.0,
                }
            ]
        },
    )
    _write_json(sft_recipe_root / "release_sft_recipe.json", {"ok": True})

    for path, stage in (
        (tmp_path / "pretrain_resume_drill.json", "pretrain"),
        (tmp_path / "sft_resume_drill.json", "sft"),
    ):
        _write_json(
            path,
            {
                "kind": "resume_drill_audit",
                "stage": stage,
                "summary": {"all_passed": True, "passed": 3, "failed": 0},
            },
        )
    _write_json(
        tmp_path / "training_soak_test.json",
        {
            "kind": "soak_test_audit",
            "summary": {"all_passed": True, "passed": 2, "failed": 0},
        },
    )

    for rel in (
        "metrics.jsonl",
        "run_args.json",
        "run_meta.json",
        "machine_recipe_summary.json",
        "machine_recipe.json",
        "machine_runtime.json",
        "update_profile.json",
    ):
        _touch(pretrain_output / rel)
    _touch(pretrain_output / "tensorboard")
    _write_json(pretrain_output / "external_eval" / "step_00000010" / "contamination_scan.json", {})
    (pretrain_output / "metrics.jsonl").write_text(
        '{"name":"pretrain_contamination_safe_rate","value":0.9}\n',
        encoding="utf-8",
    )

    for rel in ("metrics.jsonl", "run_args.json", "run_meta.json", "machine_recipe_summary.json"):
        _touch(sft_output / rel)
    _touch(sft_output / "tensorboard")
    _write_json(sft_output / "sft_capability_report.json", {"ok": True})
    _write_json(
        sft_output / "checkpoint_capability" / "step_00000010" / "sft_capability_report.json",
        {"ok": True},
    )
    (sft_output / "metrics.jsonl").write_text(
        sft_metrics or '{"stage":"sft","eval_loss":1.0,"test_loss":1.1}\n',
        encoding="utf-8",
    )

    return {
        "pretrain_recipe_root": pretrain_recipe_root,
        "sft_recipe_root": sft_recipe_root,
        "pretrain_output": pretrain_output,
        "sft_output": sft_output,
        "pretrain_resume_report": tmp_path / "pretrain_resume_drill.json",
        "sft_resume_report": tmp_path / "sft_resume_drill.json",
        "soak_report": tmp_path / "training_soak_test.json",
    }


def _build_release_audit(paths: dict[str, Path]) -> dict[str, object]:
    return mod.build_release_audit(
        pretrain_recipe_root=paths["pretrain_recipe_root"],
        pretrain_output=paths["pretrain_output"],
        pretrain_resume_report=paths["pretrain_resume_report"],
        soak_report=paths["soak_report"],
        sft_recipe_root=paths["sft_recipe_root"],
        sft_output=paths["sft_output"],
        sft_resume_report=paths["sft_resume_report"],
    )


def test_build_release_audit_passes_for_complete_artifacts(tmp_path: Path) -> None:
    payload = _build_release_audit(_prepare_complete_release(tmp_path))

    assert payload["summary"]["all_passed"] is True
    assert payload["summary"]["failed"] == 0


def test_build_release_audit_supports_pretrain_only_gate(tmp_path: Path) -> None:
    paths = _prepare_complete_release(tmp_path)

    payload = mod.build_release_audit(
        pretrain_recipe_root=paths["pretrain_recipe_root"],
        pretrain_output=paths["pretrain_output"],
        pretrain_resume_report=paths["pretrain_resume_report"],
        soak_report=paths["soak_report"],
        require_sft=False,
    )

    names = {item["name"] for item in payload["checks"]}
    assert payload["summary"]["all_passed"] is True
    assert "pretrain_machine_recipe_generated" in names
    assert "sft_machine_recipe_generated" not in names


def test_build_release_audit_fails_when_external_eval_is_missing(tmp_path: Path) -> None:
    payload = mod.build_release_audit(
        pretrain_recipe_root=tmp_path / "missing_pretrain_recipe",
        pretrain_output=tmp_path / "missing_pretrain_output",
        pretrain_resume_report=tmp_path / "missing_pretrain_resume.json",
        soak_report=tmp_path / "missing_soak.json",
        require_sft=False,
    )

    assert payload["summary"]["all_passed"] is False
    names = {item["name"]: item for item in payload["checks"]}
    assert names["pretrain_external_eval_reports"]["passed"] is False


def test_build_release_audit_requires_sft_eval_metrics(tmp_path: Path) -> None:
    paths = _prepare_complete_release(tmp_path, sft_metrics='{"stage":"sft","test_loss":1.1}\n')

    payload = _build_release_audit(paths)

    names = {item["name"]: item for item in payload["checks"]}
    assert names["sft_eval_metric_present"]["passed"] is False


def test_build_release_audit_rejects_stale_tmp_artifacts(tmp_path: Path) -> None:
    paths = _prepare_complete_release(tmp_path)
    tmp_artifact = paths["sft_output"] / "checkpoints" / "ckpt_step10.pt.tmp.123"
    tmp_artifact.parent.mkdir(parents=True, exist_ok=True)
    tmp_artifact.write_text("", encoding="utf-8")

    payload = _build_release_audit(paths)

    names = {item["name"]: item for item in payload["checks"]}
    assert names["sft_no_stale_tmp"]["passed"] is False
