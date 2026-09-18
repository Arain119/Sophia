from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ml.training.pretrain.release_config import (
    RELEASE_PRETRAIN_DEFAULTS,
    RELEASE_PRETRAIN_STABILITY_STEPS,
    RELEASE_PRETRAIN_STEPS,
)
from ml.training.pretrain.implementation_fingerprint import (
    current_pretrain_implementation_sha256,
)


EXPECTED_SEMANTICS: dict[str, object] = {
    "target_tokens_per_update": RELEASE_PRETRAIN_DEFAULTS.target_tokens_per_update,
    "optimizer_kind": "torch_muon_hybrid",
    "learning_rate": RELEASE_PRETRAIN_DEFAULTS.learning_rate,
    "muon_ns_steps": 5,
    "weight_decay": RELEASE_PRETRAIN_DEFAULTS.weight_decay,
    "beta1": 0.9,
    "beta2": 0.95,
    "adam_eps": 1e-8,
    "warmup_steps": RELEASE_PRETRAIN_DEFAULTS.warmup_steps,
    "warmup_ratio": 0.0,
    "min_lr_ratio": RELEASE_PRETRAIN_DEFAULTS.min_lr_ratio,
    "lr_schedule": RELEASE_PRETRAIN_DEFAULTS.lr_schedule,
}


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"unable to read JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=0.0)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _finite(report: dict[str, Any], field_name: str) -> bool:
    value = report.get(field_name)
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _report_errors(
    report: dict[str, Any],
    *,
    recipe_sha256: str,
    implementation_sha256: str,
    expected_step: int,
    expected_first_step: int,
    require_resume: bool,
    require_validation_and_throughput: bool,
    expected_resume_checkpoint: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    if report.get("kind") != "pretrain_stability_probe":
        errors.append("wrong_report_kind")
    if str(report.get("machine_recipe_sha256", "")).lower() != recipe_sha256:
        errors.append("recipe_sha256_mismatch")
    if str(report.get("pretrain_implementation_sha256", "")).lower() != implementation_sha256:
        errors.append("implementation_sha256_mismatch")
    if int(report.get("last_step", 0) or 0) != int(expected_step):
        errors.append(f"last_step_must_equal_{expected_step}")
    if int(report.get("first_step", 0) or 0) != int(expected_first_step):
        errors.append(f"first_step_must_equal_{expected_first_step}")
    if int(report.get("lr_schedule_horizon_steps", 0) or 0) != RELEASE_PRETRAIN_STEPS:
        errors.append("lr_schedule_horizon_steps_must_equal_15259")
    if not bool(report.get("completed")):
        errors.append("run_not_completed")
    if not bool(report.get("passed")) or report.get("status") != "pass":
        errors.append("stability_probe_failed")
    if report.get("nonfinite_loss_steps") != []:
        errors.append("nonfinite_loss")
    if report.get("nonfinite_grad_norm_steps") != []:
        errors.append("nonfinite_grad_norm")
    if not bool(report.get("attention_logit_telemetry_complete")):
        errors.append("attention_logit_telemetry_incomplete")
    if bool(str(report.get("resume_from_checkpoint", "")).strip()) != bool(
        require_resume
    ):
        errors.append("resume_request_mismatch")
    if require_resume and not str(report.get("resume_from_checkpoint", "")).endswith(
        f"ckpt_step{RELEASE_PRETRAIN_DEFAULTS.warmup_steps}.pt"
    ):
        errors.append("resume_checkpoint_must_be_warmup_boundary")
    if require_resume:
        reported_checkpoint = str(report.get("resume_checkpoint_path", "")).strip()
        reported_sha256 = str(report.get("resume_checkpoint_sha256", "")).lower()
        if expected_resume_checkpoint is None:
            errors.append("missing_expected_resume_checkpoint")
        elif not expected_resume_checkpoint.is_file():
            errors.append("expected_resume_checkpoint_missing")
        else:
            try:
                reported_path = Path(reported_checkpoint).expanduser().resolve()
            except (OSError, RuntimeError):
                reported_path = None
            if reported_path != expected_resume_checkpoint.resolve():
                errors.append("resume_checkpoint_path_mismatch")
            expected_sha256 = _sha256(expected_resume_checkpoint)
            if reported_sha256 != expected_sha256:
                errors.append("resume_checkpoint_sha256_mismatch")
    required_finite_fields = ["last_window_mean_loss"]
    if require_validation_and_throughput:
        required_finite_fields.extend(("avg_tok_s_after_20", "last_val_loss"))
    for field_name in required_finite_fields:
        if not _finite(report, field_name):
            errors.append(f"missing_finite_{field_name}")
    return errors


def _graph_recapture_report_errors(
    report: dict[str, Any],
    *,
    recipe_sha256: str,
    implementation_sha256: str,
) -> list[str]:
    errors: list[str] = []
    if report.get("kind") != "pretrain_graph_recapture_probe":
        errors.append("wrong_report_kind")
    if str(report.get("machine_recipe_sha256", "")).lower() != recipe_sha256:
        errors.append("recipe_sha256_mismatch")
    if str(report.get("pretrain_implementation_sha256", "")).lower() != implementation_sha256:
        errors.append("implementation_sha256_mismatch")
    if int(report.get("first_step", 0) or 0) != 1:
        errors.append("first_step_must_equal_1")
    if int(report.get("last_step", 0) or 0) != 12:
        errors.append("last_step_must_equal_12")
    if not bool(report.get("completed")):
        errors.append("run_not_completed")
    if not bool(report.get("passed")) or report.get("status") != "pass":
        errors.append("recapture_probe_failed")
    if not bool(report.get("graph_recapture_covered")):
        errors.append("graph_recapture_not_covered")
    if report.get("nonfinite_loss_steps") != []:
        errors.append("nonfinite_loss")
    if report.get("nonfinite_grad_norm_steps") != []:
        errors.append("nonfinite_grad_norm")
    if int(report.get("eval_count", 0) or 0) <= 0:
        errors.append("eval_not_observed")
    for field_name in (
        "graph_recapture_peak_allocated_bytes",
        "graph_recapture_peak_reserved_bytes",
    ):
        value = report.get(field_name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"missing_finite_{field_name}")
        elif int(value) <= 0:
            errors.append(f"{field_name}_must_be_positive")
    return errors


def audit_selected_muon_probe(
    *,
    recipe_path: str | Path,
    stability_path: str | Path,
    resume_path: str | Path,
    recapture_path: str | Path,
) -> dict[str, Any]:
    recipe_source = Path(recipe_path).expanduser().resolve()
    stability_source = Path(stability_path).expanduser().resolve()
    resume_source = Path(resume_path).expanduser().resolve()
    recapture_source = Path(recapture_path).expanduser().resolve()
    recipe = _read_json(recipe_source)
    stability = _read_json(stability_source)
    resume = _read_json(resume_source)
    recapture = _read_json(recapture_source)
    recipe_sha256 = _sha256(recipe_source)
    implementation_sha256 = current_pretrain_implementation_sha256()
    errors: list[str] = []

    errors.extend(
        f"recapture_{item}"
        for item in _graph_recapture_report_errors(
            recapture,
            recipe_sha256=recipe_sha256,
            implementation_sha256=implementation_sha256,
        )
    )

    if recipe.get("kind") != "pretrain_machine_recipe":
        errors.append("wrong_recipe_kind")
    if recipe.get("stage") != "pretrain":
        errors.append("wrong_recipe_stage")
    for section_name in (
        "machine_signature",
        "data_signature",
        "machine_recipe",
        "machine_runtime",
    ):
        section = recipe.get(section_name)
        if not isinstance(section, dict) or not section:
            errors.append(f"missing_{section_name}")

    semantics = recipe.get("release_semantics")
    if not isinstance(semantics, dict):
        errors.append("missing_release_semantics")
        semantics = {}
    for field_name, expected in EXPECTED_SEMANTICS.items():
        actual = semantics.get(field_name)
        if not _matches(actual, expected):
            errors.append(f"semantic_{field_name}_expected_{expected!r}_got_{actual!r}")

    stability_output_dir = str(stability.get("output_dir", "")).strip()
    expected_resume_checkpoint = (
        Path(stability_output_dir).expanduser().resolve()
        / "checkpoints"
        / f"ckpt_step{RELEASE_PRETRAIN_DEFAULTS.warmup_steps}.pt"
        if stability_output_dir
        else None
    )

    errors.extend(
        f"stability_{item}"
        for item in _report_errors(
            stability,
            recipe_sha256=recipe_sha256,
            implementation_sha256=implementation_sha256,
            expected_step=RELEASE_PRETRAIN_STABILITY_STEPS,
            expected_first_step=1,
            require_resume=False,
            require_validation_and_throughput=True,
        )
    )
    errors.extend(
        f"resume_{item}"
        for item in _report_errors(
            resume,
            recipe_sha256=recipe_sha256,
            implementation_sha256=implementation_sha256,
            expected_step=RELEASE_PRETRAIN_STABILITY_STEPS,
            expected_first_step=RELEASE_PRETRAIN_STABILITY_STEPS,
            require_resume=True,
            require_validation_and_throughput=False,
            expected_resume_checkpoint=expected_resume_checkpoint,
        )
    )

    ready = not errors
    return {
        "schema": "sophia_selected_muon_probe_audit_v4",
        "status": "ready" if ready else "blocked",
        "selected_optimizer": "torch_muon_hybrid",
        "recipe_path": str(recipe_source),
        "recipe_sha256": recipe_sha256,
        "pretrain_implementation_sha256": implementation_sha256,
        "stability_report_path": str(stability_source),
        "stability_report_sha256": _sha256(stability_source),
        "resume_report_path": str(resume_source),
        "resume_report_sha256": _sha256(resume_source),
        "recapture_report_path": str(recapture_source),
        "recapture_report_sha256": _sha256(recapture_source),
        "errors": errors,
        "pretrain_ready": bool(ready),
    }


def validate_selected_muon_probe_audit(
    *,
    audit_path: str | Path,
    recipe_path: str | Path,
) -> dict[str, Any]:
    raw_audit_path = str(audit_path or "").strip()
    if not raw_audit_path:
        raise ValueError(
            "formal pretraining requires the selected Muon warmup/resume audit"
        )
    recorded = _read_json(raw_audit_path)
    if recorded.get("schema") != "sophia_selected_muon_probe_audit_v4":
        raise ValueError("unsupported selected Muon probe audit schema")
    if recorded.get("status") != "ready" or recorded.get("errors") != []:
        raise ValueError("selected Muon probe audit is not ready")
    if recorded.get("pretrain_implementation_sha256") != current_pretrain_implementation_sha256():
        raise ValueError("selected Muon probe audit implementation changed; rerun the audit")

    expected_recipe = Path(recipe_path).expanduser().resolve()
    recorded_recipe = Path(str(recorded.get("recipe_path") or "")).expanduser().resolve()
    if recorded_recipe != expected_recipe:
        raise ValueError("selected Muon probe audit recipe path mismatch")
    recomputed = audit_selected_muon_probe(
        recipe_path=expected_recipe,
        stability_path=str(recorded.get("stability_report_path") or ""),
        resume_path=str(recorded.get("resume_report_path") or ""),
        recapture_path=str(recorded.get("recapture_report_path") or ""),
    )
    if recomputed != recorded:
        raise ValueError(
            "selected Muon probe audit evidence changed; rerun the audit"
        )
    return recorded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the selected Muon target-GPU stability and resume probes."
    )
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--stability", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--recapture", required=True)
    parser.add_argument("--output_json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_selected_muon_probe(
        recipe_path=args.recipe,
        stability_path=args.stability,
        resume_path=args.resume,
        recapture_path=args.recapture,
    )
    output_path = Path(str(args.output_json)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
