from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.tooling.scripts.audit_selected_muon_probe import (
    EXPECTED_SEMANTICS,
    audit_selected_muon_probe,
    validate_selected_muon_probe_audit,
)
from ml.training.pretrain.implementation_fingerprint import (
    current_pretrain_implementation_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _probe_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    recipe = tmp_path / "recipe.json"
    _write_json(
        recipe,
        {
            "kind": "pretrain_machine_recipe",
            "stage": "pretrain",
            "machine_signature": {"cuda_device_name": "NVIDIA GeForce RTX 5090"},
            "data_signature": {"train_manifest_sha1": "fresh_data"},
            "machine_recipe": {"batch_size": 1, "accumulation_steps": 320},
            "machine_runtime": {"step_execution_backend": "sm120_graph"},
            "release_semantics": dict(EXPECTED_SEMANTICS),
        },
    )
    recipe_sha256 = hashlib.sha256(recipe.read_bytes()).hexdigest()
    implementation_sha256 = current_pretrain_implementation_sha256()
    stability_output_dir = tmp_path / "stability_run"
    checkpoint = stability_output_dir / "checkpoints" / "ckpt_step153.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"warmup-boundary-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    common = {
        "kind": "pretrain_stability_probe",
        "status": "pass",
        "passed": True,
        "completed": True,
        "machine_recipe_sha256": recipe_sha256,
        "pretrain_implementation_sha256": implementation_sha256,
        "last_window_mean_loss": 7.5,
        "avg_tok_s_after_20": 20_000.0,
        "last_val_loss": 7.6,
        "lr_schedule_horizon_steps": 15_259,
        "nonfinite_loss_steps": [],
        "nonfinite_grad_norm_steps": [],
        "attention_logit_telemetry_complete": True,
    }
    stability1620 = tmp_path / "stability1620.json"
    resume1620 = tmp_path / "resume1620.json"
    _write_json(
        stability1620,
        {
            **common,
            "output_dir": str(stability_output_dir),
            "first_step": 1,
            "last_step": 154,
            "resume_from_checkpoint": "",
        },
    )
    _write_json(
        resume1620,
        {
            **common,
            "first_step": 154,
            "last_step": 154,
            "resume_from_checkpoint": str(checkpoint),
            "resume_checkpoint_path": str(checkpoint.resolve()),
            "resume_checkpoint_sha256": checkpoint_sha256,
        },
    )
    recapture = tmp_path / "recapture.json"
    _write_json(
        recapture,
        {
            "kind": "pretrain_graph_recapture_probe",
            "status": "pass",
            "passed": True,
            "completed": True,
            "machine_recipe_sha256": recipe_sha256,
            "pretrain_implementation_sha256": implementation_sha256,
            "first_step": 1,
            "last_step": 12,
            "graph_recapture_covered": True,
            "nonfinite_loss_steps": [],
            "nonfinite_grad_norm_steps": [],
            "eval_count": 1,
            "attention_logit_telemetry_complete": True,
            "graph_recapture_peak_allocated_bytes": 1,
            "graph_recapture_peak_reserved_bytes": 2,
        },
    )
    return recipe, stability1620, resume1620, recapture


def test_declared_muon_config_matches_executable_audit() -> None:
    payload = json.loads(
        (REPO_ROOT / "configs/pretrain/muon_recipe.json").read_text(
            encoding="utf-8"
        )
    )
    declared = {
        **payload["selected_semantics"],
        "target_tokens_per_update": payload["semantic_target_tokens_per_update"],
    }
    for field_name, value in EXPECTED_SEMANTICS.items():
        assert declared[field_name] == value


def test_selected_muon_probe_requires_stability_validation_and_resume(
    tmp_path: Path,
) -> None:
    recipe, stability1620, resume1620, recapture = _probe_files(tmp_path)

    result = audit_selected_muon_probe(
        recipe_path=recipe,
        stability_path=stability1620,
        resume_path=resume1620,
        recapture_path=recapture,
    )

    assert result["status"] == "ready"
    assert result["selected_optimizer"] == "torch_muon_hybrid"
    assert result["errors"] == []
    assert result["pretrain_ready"] is True


def test_formal_audit_validation_recomputes_all_bound_evidence(tmp_path: Path) -> None:
    recipe, stability1620, resume1620, recapture = _probe_files(tmp_path)
    audit = audit_selected_muon_probe(
        recipe_path=recipe,
        stability_path=stability1620,
        resume_path=resume1620,
        recapture_path=recapture,
    )
    audit_path = tmp_path / "audit.json"
    _write_json(audit_path, audit)

    validated = validate_selected_muon_probe_audit(
        audit_path=audit_path,
        recipe_path=recipe,
    )
    assert validated == audit

    stability = json.loads(stability1620.read_text(encoding="utf-8"))
    stability["last_val_loss"] = 99.0
    _write_json(stability1620, stability)
    with pytest.raises(ValueError, match="evidence changed"):
        validate_selected_muon_probe_audit(
            audit_path=audit_path,
            recipe_path=recipe,
        )


def test_selected_muon_probe_rejects_missing_validation_and_fake_resume(
    tmp_path: Path,
) -> None:
    recipe, stability1620, resume1620, recapture = _probe_files(tmp_path)
    step_payload = json.loads(stability1620.read_text(encoding="utf-8"))
    step_payload["last_val_loss"] = None
    _write_json(stability1620, step_payload)
    resume_payload = json.loads(resume1620.read_text(encoding="utf-8"))
    resume_payload["resume_from_checkpoint"] = ""
    _write_json(resume1620, resume_payload)

    result = audit_selected_muon_probe(
        recipe_path=recipe,
        stability_path=stability1620,
        resume_path=resume1620,
        recapture_path=recapture,
    )

    assert result["status"] == "blocked"
    assert "stability_missing_finite_last_val_loss" in result["errors"]
    assert "resume_resume_request_mismatch" in result["errors"]


def test_selected_muon_probe_rejects_checkpoint_identity_mismatch(
    tmp_path: Path,
) -> None:
    recipe, stability1620, resume1620, recapture = _probe_files(tmp_path)
    resume_payload = json.loads(resume1620.read_text(encoding="utf-8"))
    resume_payload["resume_checkpoint_sha256"] = "0" * 64
    _write_json(resume1620, resume_payload)

    result = audit_selected_muon_probe(
        recipe_path=recipe,
        stability_path=stability1620,
        resume_path=resume1620,
        recapture_path=recapture,
    )

    assert result["status"] == "blocked"
    assert "resume_resume_checkpoint_sha256_mismatch" in result["errors"]


def test_selected_muon_probe_rejects_uncovered_graph_recapture(
    tmp_path: Path,
) -> None:
    recipe, stability1620, resume1620, recapture = _probe_files(tmp_path)
    recapture_payload = json.loads(recapture.read_text(encoding="utf-8"))
    recapture_payload["graph_recapture_covered"] = False
    _write_json(recapture, recapture_payload)

    result = audit_selected_muon_probe(
        recipe_path=recipe,
        stability_path=stability1620,
        resume_path=resume1620,
        recapture_path=recapture,
    )

    assert result["status"] == "blocked"
    assert "recapture_graph_recapture_not_covered" in result["errors"]


def test_selected_muon_probe_rejects_nonfinite_recapture_report(
    tmp_path: Path,
) -> None:
    recipe, stability1620, resume1620, recapture = _probe_files(tmp_path)
    recapture_payload = json.loads(recapture.read_text(encoding="utf-8"))
    recapture_payload["nonfinite_loss_steps"] = [11]
    _write_json(recapture, recapture_payload)

    result = audit_selected_muon_probe(
        recipe_path=recipe,
        stability_path=stability1620,
        resume_path=resume1620,
        recapture_path=recapture,
    )

    assert result["status"] == "blocked"
    assert "recapture_nonfinite_loss" in result["errors"]
