from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.core.spec import ModelSpec
from ml.tooling.scripts import eval_pretrain_base_capability_gate as mod

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "configs/eval/pretrain_base_checkpoint_policy.json"
PROTOCOL_PATH = REPO_ROOT / "configs/eval/pretrain_release_protocol.json"
MODEL_SPEC_SHA256 = "c605697ccdcd92b11095e04fe5a3cbad30d6388ec450568996c45db1afa4ba6a"


def test_protocol_uses_canonical_model_spec_hash() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    encoded = json.dumps(ModelSpec.default().to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == MODEL_SPEC_SHA256
    assert protocol["model"]["spec_sha256"] == MODEL_SPEC_SHA256


def test_protocol_binds_exact_checkpoint_policy(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["policy"]["sha256"] = "0" * 64
    protocol_path = tmp_path / PROTOCOL_PATH.name
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="policy sha256 mismatch"):
        mod.validate_final_checkpoint(
            protocol_path=protocol_path,
            evidence_path=tmp_path / "unused.json",
        )


def test_alternate_protocol_is_explicitly_selected(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    alternate = tmp_path / "alternate_protocol.json"
    alternate.write_text(json.dumps(protocol), encoding="utf-8")
    candidate = _candidate(tmp_path / "candidate", protocol_path=alternate)

    report = mod.validate_final_checkpoint(
        protocol_path=alternate,
        evidence_path=candidate,
    )
    assert report["protocol_path"] == str(alternate.resolve())
    assert report["evidence"]["checks"]


def _write_json(path: Path, payload: object) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _candidate(
    tmp_path: Path,
    *,
    seed: int = 42,
    answer_accuracy: float = 0.80,
    overrides: dict[str, object] | None = None,
    protocol_path: Path = PROTOCOL_PATH,
    stability_last_step: int = 154,
) -> Path:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / f"ckpt_seed{seed}.pt"
    checkpoint.write_bytes(f"checkpoint-{seed}".encode())
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    eval_model = tmp_path / f"eval_model_seed{seed}.safetensors"
    eval_model.write_bytes(f"eval-model-{seed}".encode())
    eval_model_sha = hashlib.sha256(eval_model.read_bytes()).hexdigest()
    binding = {
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": checkpoint_sha,
        "eval_export_model_path": str(eval_model.resolve()),
        "eval_export_model_sha256": eval_model_sha,
    }
    stability_binding = {
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": checkpoint_sha,
    }
    generation = {
        "schema": "sophia_generation_quality_v1",
        "suite_sha256": policy["evaluation_assets"]["full_generation_suite"]["sha256"],
        "prompt_format": "raw",
        "cases": [{}] * 2000,
        "evaluation_binding": binding,
        "summary": {
            "overall": {
                "answer_accuracy": answer_accuracy,
                "empty_rate": 0.0,
                "first_token_eos_rate": 0.0,
                "language_match_rate": 1.0,
                "prompt_echo_rate": 0.0,
                "abnormal_markup_or_path_rate": 0.0,
                "high_repetition_rate": 0.0,
            },
            "by_capability": {
                "math": {"answer_accuracy": 0.80},
                "logic": {"answer_accuracy": 0.85},
                "code": {"answer_accuracy": 0.50},
                "fact": {"answer_accuracy": 0.60},
                "science": {"answer_accuracy": 0.65},
            },
            "by_language": {"zh": {"answer_accuracy": 0.75}, "en": {"answer_accuracy": 0.75}},
        },
    }
    chinese_mc = {
        "schema": "sophia_chinese_multiple_choice_eval_v1",
        "benchmark_sha256": policy["evaluation_assets"]["chinese_multiple_choice"]["sha256"],
        "evaluation_binding": binding,
        "summary": {"overall": {"accuracy": 0.30}},
    }
    ppl_manifests = {
        "zh_val": tmp_path / "zh_val_manifest.json",
        "en_val": tmp_path / "en_val_manifest.json",
    }
    for name, manifest_path in ppl_manifests.items():
        manifest_path.write_text(
            json.dumps({"source": name}, sort_keys=True), encoding="utf-8"
        )
    ppl_fingerprints = {
        name: hashlib.sha1(path.read_bytes()).hexdigest()
        for name, path in ppl_manifests.items()
    }
    perplexity_binding = {**binding, "dataset_fingerprints": ppl_fingerprints}
    perplexity = {
        "schema": "sophia_pretrain_perplexity_eval_v1",
        "evaluation_binding": perplexity_binding,
        "datasets": {
            "zh_val": {
                "manifest_sha1": ppl_fingerprints["zh_val"],
                "perplexity": 12.0,
            },
            "en_val": {
                "manifest_sha1": ppl_fingerprints["en_val"],
                "perplexity": 10.0,
            },
        },
    }
    stability = {
        "kind": "pretrain_stability_probe",
        "status": "pass",
        "passed": True,
        "completed": True,
        "last_step": stability_last_step,
        "evaluation_binding": stability_binding,
    }
    executable = {
        "schema": "sophia_executable_capability_eval_v1",
        "suite_sha256": policy["evaluation_assets"]["full_generation_suite"]["sha256"],
        "evaluation_binding": binding,
        "summary": {"math_exact_match_rate": 0.80, "code_pass_at_1": 0.50},
    }
    contamination_report = _write_json(
        tmp_path / f"contamination_{seed}.json",
        {
            "kind": "pretrain_contamination_scan_report",
            "measure_kind": "generated_text_health",
            "step": 15_259,
            "safe_rate_mean": 1.0,
            "splits": [],
        },
    )
    reports = {
        "generation": _write_json(tmp_path / f"generation_{seed}.json", generation),
        "executable": _write_json(tmp_path / f"executable_{seed}.json", executable),
        "chinese_multiple_choice": _write_json(tmp_path / f"mc_{seed}.json", chinese_mc),
        "perplexity": _write_json(tmp_path / f"ppl_{seed}.json", perplexity),
        "stability": _write_json(tmp_path / f"stability_{seed}.json", stability),
    }
    candidate: dict[str, object] = {
        "schema": mod.EVIDENCE_SCHEMA,
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        "model": {
            "parameter_count": 1012630480,
            "max_seq_len": 4096,
            "spec_sha256": MODEL_SPEC_SHA256,
        },
        "training": {
            "target_tokens": 20_000_000_000,
            "tokens_per_update": 1_310_720,
            "release_steps": 15_259,
            "consumed_tokens": 20_000_276_480,
            "seed": seed,
        },
        "lineage": {
            "tokenizer_sha1": "a" * 40,
            "train_manifest_sha1": "b" * 40,
            "val_manifest_sha1": "c" * 40,
            "test_manifest_sha1": "d" * 40,
            "machine_recipe_sha256": "f" * 64,
            "data_admission_sha256": "0" * 64,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "step": 15_259,
            "consumed_tokens": 20_000_276_480,
        },
        "eval_export_model": {
            "path": str(eval_model),
            "sha256": eval_model_sha,
        },
        "reports": reports,
        "ppl_datasets": {
            name: {"path": str(path), "sha1": ppl_fingerprints[name]}
            for name, path in ppl_manifests.items()
        },
        "ppl_language_groups": {"zh": ["zh_val"], "en": ["en_val"]},
        "generated_text_health_safe_rate": 1.0,
        "contamination_report": contamination_report,
        "decontamination_assets": {
            name: {
                "path": str((REPO_ROOT / item["path"]).resolve()),
                "sha256": item["sha256"],
            }
            for name, item in protocol["evaluation_assets"].items()
            if name
            in {
                "generation_decontamination_signatures",
                "full_benchmark_decontamination",
            }
        },
    }
    candidate["provenance"] = {
        "source": "sophia-pretraining",
        "version": "1",
        "license": "research",
        "parameter_count": 1012630480,
        "training_tokens": 20_000_000_000,
    }
    if overrides:
        candidate.update(overrides)
    path = tmp_path / f"candidate_{seed}.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    return path


def _evaluate(tmp_path: Path, evidence: Path, *, protocol_path: Path = PROTOCOL_PATH) -> dict[str, object]:
    del tmp_path
    return mod.validate_final_checkpoint(
        protocol_path=protocol_path,
        policy_path=POLICY_PATH,
        evidence_path=evidence,
    )


@pytest.mark.parametrize("stability_last_step", [0, 153])
def test_stability_before_minimum_fails(tmp_path: Path, stability_last_step: int) -> None:
    report = _evaluate(tmp_path, _candidate(tmp_path, stability_last_step=stability_last_step))
    stability_checks = report["evidence"]["checks"]
    assert report["valid"] is False
    assert any(check["name"] == "stability.last_step" and not check["passed"] for check in stability_checks)


def test_gate_rejects_ppl_manifest_binding_drift(tmp_path: Path) -> None:
    candidates = [_candidate(tmp_path / "42", seed=42)]
    candidate = json.loads(candidates[0].read_text(encoding="utf-8"))
    report_path = Path(candidate["reports"]["perplexity"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["datasets"]["zh_val"]["manifest_sha1"] = "0" * 40
    report_path.write_text(json.dumps(report), encoding="utf-8")
    candidate["reports"]["perplexity"]["sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    candidates[0].write_text(json.dumps(candidate), encoding="utf-8")

    gate = _evaluate(tmp_path, candidates[0])

    assert gate["valid"] is False
    row = gate["evidence"]
    check = next(
        item
        for item in row["checks"]
        if item["name"] == "ppl_dataset.zh_val.manifest_sha1"
    )
    assert check["passed"] is False


def test_gate_rejects_tampered_decontamination_asset_binding(tmp_path: Path) -> None:
    candidates = [_candidate(tmp_path / "42", seed=42)]
    candidate = json.loads(candidates[0].read_text(encoding="utf-8"))
    candidate["decontamination_assets"]["full_benchmark_decontamination"][
        "sha256"
    ] = "0" * 64
    candidates[0].write_text(json.dumps(candidate), encoding="utf-8")

    gate = _evaluate(tmp_path, candidates[0])

    assert gate["valid"] is False
    invalid = gate["evidence"]
    assert "full_benchmark_decontamination sha256 mismatch" in " ".join(
        invalid["failure_reasons"]
    )


def test_gate_rejects_contamination_scalar_not_bound_to_report(tmp_path: Path) -> None:
    candidates = [
        _candidate(
            tmp_path / str(seed),
            seed=seed,
            overrides=(
                {"generated_text_health_safe_rate": 0.99}
                if seed == 42
                else None
            ),
        )
        for seed in (42,)
    ]

    gate = _evaluate(tmp_path, candidates[0])

    assert gate["valid"] is False
    row = gate["evidence"]
    check = next(
        item
        for item in row["checks"]
        if item["name"] == "generated_text_health_safe_rate.binding"
    )
    assert check["passed"] is False


def test_gate_rejects_tampered_eval_export_model_binding(tmp_path: Path) -> None:
    candidate_paths = [_candidate(tmp_path / "42", seed=42)]
    candidate = json.loads(candidate_paths[0].read_text(encoding="utf-8"))
    report_path = Path(candidate["reports"]["generation"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["evaluation_binding"]["eval_export_model_sha256"] = "0" * 64
    _write_json(report_path, report)
    candidate["reports"]["generation"]["sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    _write_json(candidate_paths[0], candidate)

    gate = _evaluate(tmp_path, candidate_paths[0])

    assert gate["valid"] is False
    row = gate["evidence"]
    binding_check = next(
        item
        for item in row["checks"]
        if item["name"] == "report.generation.evaluation_binding"
    )
    assert binding_check["passed"] is False


def test_single_formal_seed_passes_with_exact_token_overshoot(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "42", seed=42)
    report = _evaluate(tmp_path, candidate)

    assert report["status"] == "pass"
    assert report["valid"] is True
    assert report["evidence"]["candidate_path"] == str(candidate.resolve())
    assert report["evidence"]["checkpoint"]["consumed_tokens"] == 20_000_276_480


def test_checkpoint_step_must_match_release_step(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["checkpoint"]["step"] = 800
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    report = _evaluate(tmp_path, candidate)
    assert report["valid"] is False
    assert any(check["name"] == "checkpoint.step" and not check["passed"] for check in report["evidence"]["checks"])


def test_nonformal_seed_fails_closed(tmp_path: Path) -> None:
    report = _evaluate(tmp_path, _candidate(tmp_path, seed=43))

    assert report["status"] == "fail"
    assert report["valid"] is False
    seed_check = next(
        check for check in report["evidence"]["checks"] if check["name"] == "training.seed"
    )
    assert seed_check["passed"] is False


def test_binding_and_lineage_mismatches_fail(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, overrides={"model": {"parameter_count": 1, "max_seq_len": 2048, "spec_sha256": "0" * 64}})
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["lineage"]["train_manifest_sha1"] = "not-a-hash"
    payload["reports"]["stability"]["sha256"] = "0" * 64
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    report = _evaluate(tmp_path, candidate)

    assert report["valid"] is False
    reasons = str(report["evidence"])
    assert "model.parameter_count" in reasons or "evidence_validation" in reasons
