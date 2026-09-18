from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ml.core.common.io import write_json_atomic
EVIDENCE_SCHEMA = "sophia_pretrain_final_checkpoint_evidence_v1"
OUTPUT_SCHEMA = "sophia_pretrain_final_checkpoint_validation_v1"
REPORT_NAMES = ("generation", "executable", "chinese_multiple_choice", "perplexity", "stability")
FORMAL_SEED = 42
SHA1_FIELDS = ("tokenizer_sha1", "train_manifest_sha1", "val_manifest_sha1", "test_manifest_sha1")
SHA256_FIELDS = ("machine_recipe_sha256", "data_admission_sha256")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_path(value: object, *, base: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _verified_artifact(payload: object, *, label: str, base: Path) -> tuple[Path, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"candidate is missing {label} artifact metadata")
    path = _resolve_path(payload.get("path"), base=base)
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    expected = str(payload.get("sha256") or "").lower()
    actual = _sha256(path)
    if len(expected) != 64 or expected != actual:
        raise ValueError(f"{label} sha256 mismatch: expected={expected} actual={actual}")
    return path, actual


def _nested(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def _finite_number(value: object, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _finite_metric(*, name: str, value: object, rate: bool = True) -> dict[str, Any]:
    try:
        number = _finite_number(value, label=name)
    except ValueError as exc:
        return _failure(name, reason=str(exc))
    passed = not rate or 0.0 <= number <= 1.0
    return {
        "name": name,
        "passed": passed,
        "value": number,
        "expectation": "finite rate in [0, 1]" if rate else "finite number",
    }


def _mean_ppl(report: dict[str, Any], *, names: list[str], label: str) -> float:
    datasets = report.get("datasets")
    if not isinstance(datasets, dict) or not names:
        raise ValueError(f"candidate must declare at least one {label} PPL dataset")
    values = []
    for name in names:
        row = datasets.get(str(name))
        if not isinstance(row, dict):
            raise ValueError(f"missing PPL dataset {name!r} for {label}")
        value = _finite_number(row.get("perplexity"), label=f"ppl.{name}")
        if value <= 0:
            raise ValueError(f"ppl.{name} must be positive")
        values.append(value)
    return sum(values) / len(values)


def _hash(value: object, length: int) -> bool:
    text = str(value or "")
    return len(text) == length and all(char in "0123456789abcdef" for char in text.lower())


def _failure(name: str, value: object = None, reason: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name, "passed": False, "value": value}
    if reason:
        row["reason"] = reason
    return row


def _binding(
    report: dict[str, Any],
    *,
    protocol_sha256: str,
    checkpoint_sha256: str,
    eval_export_model: dict[str, str],
    name: str,
    base: Path,
) -> dict[str, Any]:
    binding = report.get("evaluation_binding")
    if not isinstance(binding, dict):
        return _failure(f"report.{name}.evaluation_binding", reason="missing evaluation_binding")
    passed = (
        binding.get("protocol_sha256") == protocol_sha256
        and binding.get("checkpoint_sha256") == checkpoint_sha256
    )
    threshold: dict[str, Any] = {
        "protocol_sha256": protocol_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    model_path = binding.get("eval_export_model_path")
    model_sha256 = binding.get("eval_export_model_sha256")
    if name == "stability" and model_path in (None, "") and model_sha256 in (None, ""):
        pass
    else:
        try:
            passed = passed and _resolve_path(model_path, base=base) == Path(
                eval_export_model["path"]
            ).resolve()
        except (TypeError, ValueError):
            passed = False
        passed = passed and model_sha256 == eval_export_model["sha256"]
        threshold.update(eval_export_model)
    return {
        "name": f"report.{name}.evaluation_binding",
        "passed": passed,
        "value": binding,
        "threshold": threshold,
    }


def _protocol_asset(
    protocol: dict[str, Any], *, name: str
) -> tuple[Path, str]:
    assets = protocol.get("evaluation_assets")
    item = assets.get(name) if isinstance(assets, dict) else None
    if not isinstance(item, dict):
        raise ValueError(f"protocol evaluation asset is missing: {name}")
    path = _resolve_path(item.get("path"), base=_REPO_ROOT)
    expected_sha256 = str(item.get("sha256") or "")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"protocol evaluation asset sha256 mismatch: {name}")
    return path, actual_sha256


def _evaluate_v2(  # noqa: C901 - fail-closed audit keeps all candidate checks together
    candidate_path: Path,
    policy: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    candidate = _load_json(candidate_path)
    base = candidate_path.parent
    checks: list[dict[str, Any]] = []
    protocol_path, protocol_sha = _verified_artifact(candidate.get("protocol"), label="protocol", base=base)
    expected_protocol_path = Path(str(protocol["_path"])).resolve()
    if protocol_path.resolve() != expected_protocol_path:
        checks.append(_failure("protocol.path", reason="candidate protocol is not the evaluated protocol"))
    checks.append({"name": "protocol.sha256", "passed": protocol_sha == _sha256(expected_protocol_path), "value": protocol_sha, "threshold": _sha256(expected_protocol_path)})
    candidate_decontamination = candidate.get("decontamination_assets")
    if not isinstance(candidate_decontamination, dict):
        raise ValueError("candidate decontamination_assets must be an object")
    for asset_name in (
        "generation_decontamination_signatures",
        "full_benchmark_decontamination",
    ):
        expected_path, expected_sha256 = _protocol_asset(protocol, name=asset_name)
        candidate_asset_path, candidate_asset_sha256 = _verified_artifact(
            candidate_decontamination.get(asset_name),
            label=f"decontamination_assets.{asset_name}",
            base=base,
        )
        checks.append(
            {
                "name": f"decontamination_assets.{asset_name}",
                "passed": candidate_asset_path == expected_path
                and candidate_asset_sha256 == expected_sha256,
                "value": {
                    "path": str(candidate_asset_path),
                    "sha256": candidate_asset_sha256,
                },
                "threshold": {
                    "path": str(expected_path),
                    "sha256": expected_sha256,
                },
            }
        )
    model = candidate.get("model")
    if not isinstance(model, dict):
        raise ValueError("candidate model must be an object")
    expected_model = protocol["model"]
    checks += [
        {"name": "model.parameter_count", "passed": model.get("parameter_count") == expected_model["parameter_count"], "value": model.get("parameter_count"), "threshold": expected_model["parameter_count"]},
        {"name": "model.max_seq_len", "passed": model.get("max_seq_len") == expected_model["max_seq_len"], "value": model.get("max_seq_len"), "threshold": expected_model["max_seq_len"]},
        {"name": "model.spec_sha256", "passed": model.get("spec_sha256") == expected_model["spec_sha256"], "value": model.get("spec_sha256"), "threshold": expected_model["spec_sha256"]},
    ]
    training = candidate.get("training")
    if not isinstance(training, dict):
        raise ValueError("candidate training must be an object")
    expected_training = protocol["training"]
    for key in ("target_tokens", "tokens_per_update", "release_steps", "consumed_tokens"):
        checks.append({"name": f"training.{key}", "passed": training.get(key) == expected_training[key], "value": training.get(key), "threshold": expected_training[key]})
    checks.append({"name": "training.overshoot_tokens", "passed": training.get("consumed_tokens", 0) - training.get("target_tokens", 0) == expected_training["overshoot_tokens"], "value": training.get("consumed_tokens", 0) - training.get("target_tokens", 0), "threshold": expected_training["overshoot_tokens"]})
    seed = training.get("seed")
    checks.append({"name": "training.seed", "passed": seed == FORMAL_SEED, "value": seed, "threshold": FORMAL_SEED})

    lineage = candidate.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("candidate lineage must be an object")
    for key in SHA1_FIELDS:
        checks.append({"name": f"lineage.{key}", "passed": _hash(lineage.get(key), 40), "value": lineage.get(key)})
    for key in SHA256_FIELDS:
        checks.append({"name": f"lineage.{key}", "passed": _hash(lineage.get(key), 64), "value": lineage.get(key)})

    checkpoint = candidate.get("checkpoint")
    checkpoint_path, checkpoint_sha = _verified_artifact(checkpoint, label="checkpoint", base=base)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be an object")
    checkpoint_tokens = checkpoint.get("consumed_tokens", 0)
    expected_checkpoint_tokens = protocol["training"]["consumed_tokens"]
    checks += [
        {"name": "checkpoint.step", "passed": checkpoint.get("step") == protocol["training"]["release_steps"] and checkpoint.get("step") == training.get("release_steps"), "value": checkpoint.get("step"), "threshold": protocol["training"]["release_steps"]},
        {"name": "checkpoint.consumed_tokens", "passed": checkpoint_tokens == expected_checkpoint_tokens, "value": checkpoint.get("consumed_tokens"), "threshold": expected_checkpoint_tokens},
        {"name": "checkpoint.training_binding", "passed": checkpoint.get("consumed_tokens") == training.get("consumed_tokens"), "value": checkpoint.get("consumed_tokens")},
    ]
    eval_model_path, eval_model_sha256 = _verified_artifact(
        candidate.get("eval_export_model"), label="eval_export_model", base=base
    )
    eval_export_model = {
        "path": str(eval_model_path),
        "sha256": eval_model_sha256,
    }
    reports = candidate.get("reports")
    if not isinstance(reports, dict):
        raise ValueError("candidate reports must be an object")
    loaded: dict[str, dict[str, Any]] = {}
    report_artifacts: dict[str, dict[str, str]] = {}
    for name in REPORT_NAMES:
        path, digest = _verified_artifact(reports.get(name), label=f"report.{name}", base=base)
        loaded[name] = _load_json(path)
        report_artifacts[name] = {"path": str(path), "sha256": digest}
        checks.append(
            _binding(
                loaded[name],
                protocol_sha256=protocol_sha,
                checkpoint_sha256=checkpoint_sha,
                eval_export_model=eval_export_model,
                name=name,
                base=path.parent,
            )
        )

    generation = loaded["generation"]
    asset = policy["evaluation_assets"]["full_generation_suite"]
    checks += [
        {"name": "generation_schema", "passed": generation.get("schema") == "sophia_generation_quality_v1", "value": generation.get("schema")},
        {"name": "generation_suite_sha256", "passed": generation.get("suite_sha256") == asset["sha256"], "value": generation.get("suite_sha256"), "threshold": asset["sha256"]},
        {"name": "generation_case_count", "passed": len(generation.get("cases") or []) == 2000, "value": len(generation.get("cases") or []), "threshold": 2000},
        {"name": "generation_prompt_format", "passed": generation.get("prompt_format") == "raw", "value": generation.get("prompt_format")},
    ]
    summary = generation.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("generation report is missing summary")
    generation_metrics = (
        "overall.answer_accuracy",
        "overall.empty_rate",
        "overall.first_token_eos_rate",
        "overall.language_match_rate",
        "overall.prompt_echo_rate",
        "overall.abnormal_markup_or_path_rate",
        "overall.high_repetition_rate",
        "by_capability.math.answer_accuracy",
        "by_capability.logic.answer_accuracy",
        "by_capability.code.answer_accuracy",
        "by_capability.fact.answer_accuracy",
        "by_capability.science.answer_accuracy",
        "by_language.zh.answer_accuracy",
        "by_language.en.answer_accuracy",
    )
    for metric_key in generation_metrics:
        name = f"generation.{metric_key}"
        try:
            value = _nested(summary, metric_key)
        except KeyError as exc:
            checks.append(_failure(name, reason=str(exc)))
        else:
            checks.append(_finite_metric(name=name, value=value))

    mc = loaded["chinese_multiple_choice"]
    mc_accuracy = _finite_number(_nested(mc, "summary.overall.accuracy"), label="chinese_multiple_choice.overall.accuracy")
    checks += [
        {"name": "chinese_multiple_choice_schema", "passed": mc.get("schema") == "sophia_chinese_multiple_choice_eval_v1", "value": mc.get("schema")},
        {"name": "chinese_multiple_choice_benchmark_sha256", "passed": mc.get("benchmark_sha256") == policy["evaluation_assets"]["chinese_multiple_choice"]["sha256"], "value": mc.get("benchmark_sha256")},
        _finite_metric(name="chinese_multiple_choice.overall_accuracy", value=mc_accuracy),
    ]
    stability = loaded["stability"]
    stability_contract = protocol.get("stability_contract")
    if not isinstance(stability_contract, dict):
        raise ValueError("protocol is missing stability_contract")
    stability_last_step = int(stability.get("last_step", 0) or 0)
    stability_checks = [
        {"name": "stability.kind", "passed": stability.get("kind") == stability_contract["kind"], "value": stability.get("kind"), "threshold": stability_contract["kind"]},
        {"name": "stability.status", "passed": stability.get("status") == stability_contract["status"], "value": stability.get("status"), "threshold": stability_contract["status"]},
        {"name": "stability.passed", "passed": stability.get("passed") is stability_contract["passed"], "value": stability.get("passed"), "threshold": stability_contract["passed"]},
        {"name": "stability.completed", "passed": stability.get("completed") is stability_contract["completed"], "value": stability.get("completed"), "threshold": stability_contract["completed"]},
        {"name": "stability.last_step", "passed": stability_last_step >= int(stability_contract["minimum_last_step"]), "value": stability_last_step, "threshold": stability_contract["minimum_last_step"]},
    ]
    checks.extend(stability_checks)
    executable = loaded["executable"]
    executable_summary = executable.get("summary")
    if not isinstance(executable_summary, dict):
        raise ValueError("executable report is missing summary")
    checks += [
        {"name": "executable_schema", "passed": executable.get("schema") == "sophia_executable_capability_eval_v1", "value": executable.get("schema")},
        {"name": "executable_suite_sha256", "passed": executable.get("suite_sha256") == asset["sha256"], "value": executable.get("suite_sha256")},
        _finite_metric(name="executable.math_exact_match_rate", value=executable_summary.get("math_exact_match_rate")),
        _finite_metric(name="executable.code_pass_at_1", value=executable_summary.get("code_pass_at_1")),
    ]
    contamination_path, _contamination_sha256 = _verified_artifact(
        candidate.get("contamination_report"),
        label="contamination_report",
        base=base,
    )
    contamination_report = _load_json(contamination_path)
    contamination = _finite_number(
        contamination_report.get("safe_rate_mean"),
        label="contamination_report.safe_rate_mean",
    )
    checks += [
        {
            "name": "contamination_report.kind",
            "passed": contamination_report.get("kind")
            == "pretrain_contamination_scan_report",
            "value": contamination_report.get("kind"),
            "threshold": "pretrain_contamination_scan_report",
        },
        {
            "name": "contamination_report.step",
            "passed": contamination_report.get("step") == checkpoint.get("step"),
            "value": contamination_report.get("step"),
            "threshold": checkpoint.get("step"),
        },
        {
            "name": "contamination_report.measure_kind",
            "passed": contamination_report.get("measure_kind")
            == "generated_text_health",
            "value": contamination_report.get("measure_kind"),
            "threshold": "generated_text_health",
        },
        {
            "name": "generated_text_health_safe_rate.binding",
            "passed": candidate.get("generated_text_health_safe_rate")
            == contamination,
            "value": candidate.get("generated_text_health_safe_rate"),
            "threshold": contamination,
        },
        _finite_metric(name="generated_text_health_safe_rate", value=contamination),
    ]
    ppl = loaded["perplexity"]
    checks.append({"name": "perplexity_schema", "passed": ppl.get("schema") == "sophia_pretrain_perplexity_eval_v1", "value": ppl.get("schema")})
    ppl_datasets = candidate.get("ppl_datasets")
    if not isinstance(ppl_datasets, dict) or not ppl_datasets:
        raise ValueError("candidate must declare ppl_datasets")
    report_datasets = ppl.get("datasets")
    ppl_binding = ppl.get("evaluation_binding")
    dataset_fingerprints = (
        ppl_binding.get("dataset_fingerprints")
        if isinstance(ppl_binding, dict)
        else None
    )
    if not isinstance(report_datasets, dict) or not isinstance(
        dataset_fingerprints, dict
    ):
        raise ValueError("PPL report is missing dataset fingerprints")
    canonical_ppl_datasets: dict[str, dict[str, str]] = {}
    for name, artifact in sorted(ppl_datasets.items()):
        if not isinstance(artifact, dict):
            raise ValueError(f"PPL dataset artifact is invalid: {name}")
        manifest_path = _resolve_path(artifact.get("path"), base=base)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing PPL dataset manifest: {manifest_path}")
        expected_sha1 = str(artifact.get("sha1") or "").lower()
        actual_sha1 = _sha1(manifest_path)
        row = report_datasets.get(name)
        report_sha1 = row.get("manifest_sha1") if isinstance(row, dict) else None
        binding_sha1 = dataset_fingerprints.get(name)
        checks.append(
            {
                "name": f"ppl_dataset.{name}.manifest_sha1",
                "passed": len(expected_sha1) == 40
                and expected_sha1 == actual_sha1
                and report_sha1 == actual_sha1
                and binding_sha1 == actual_sha1,
                "value": {
                    "candidate": expected_sha1,
                    "report": report_sha1,
                    "binding": binding_sha1,
                },
                "threshold": actual_sha1,
            }
        )
        canonical_ppl_datasets[name] = {
            "path": str(manifest_path),
            "sha1": actual_sha1,
        }
    groups = candidate.get("ppl_language_groups")
    if not isinstance(groups, dict):
        raise ValueError("candidate must declare ppl_language_groups")
    zh_ppl, en_ppl = _mean_ppl(ppl, names=list(groups.get("zh") or []), label="zh"), _mean_ppl(ppl, names=list(groups.get("en") or []), label="en")
    metrics = {"generation_overall": _finite_number(_nested(summary, "overall.answer_accuracy"), label="answer_accuracy"), "math": _finite_number(executable_summary["math_exact_match_rate"], label="math"), "logic": _finite_number(_nested(summary, "by_capability.logic.answer_accuracy"), label="logic"), "fact": _finite_number(_nested(summary, "by_capability.fact.answer_accuracy"), label="fact"), "science": _finite_number(_nested(summary, "by_capability.science.answer_accuracy"), label="science"), "code": _finite_number(executable_summary["code_pass_at_1"], label="code"), "chinese_multiple_choice": mc_accuracy, "zh_ppl": zh_ppl, "en_ppl": en_ppl, "pathology": sum(_finite_number(_nested(summary, f"overall.{key}"), label=key) for key in ("empty_rate", "first_token_eos_rate", "prompt_echo_rate", "abnormal_markup_or_path_rate", "high_repetition_rate")) / 5}
    provenance = candidate.get("provenance")
    if not isinstance(provenance, dict):
        checks.append(_failure("provenance", reason="missing provenance"))
    else:
        for key in ("source", "version", "license", "parameter_count", "training_tokens"):
            checks.append({"name": f"provenance.{key}", "passed": key in provenance and provenance[key] not in (None, ""), "value": provenance.get(key)})
    passed = all(bool(check.get("passed")) for check in checks)
    return {"candidate_path": str(candidate_path.resolve()), "candidate_sha256": _sha256(candidate_path), "seed": seed, "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha, "step": checkpoint.get("step"), "consumed_tokens": checkpoint.get("consumed_tokens")}, "eval_export_model": eval_export_model, "reports": report_artifacts, "checks": checks, "hard_gates_passed": passed, "metrics": metrics}


def _evaluate_candidate(candidate_path: Path, policy: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    try:
        candidate = _load_json(candidate_path)
        if candidate.get("schema") != EVIDENCE_SCHEMA:
            raise ValueError("unsupported final checkpoint evidence schema")
        return _evaluate_v2(candidate_path, policy, protocol)
    except Exception as exc:
        return {"candidate_path": str(candidate_path.resolve()), "hard_gates_passed": False, "checks": [_failure("evidence_validation", reason=str(exc))], "failure_reasons": [str(exc)]}


def validate_final_checkpoint(*, protocol_path: Path, evidence_path: Path, policy_path: Path | None = None) -> dict[str, Any]:
    protocol_path = protocol_path.expanduser().resolve()
    protocol = _load_json(protocol_path)
    policy_binding = protocol.get("policy")
    if not isinstance(policy_binding, dict):
        raise ValueError("release protocol is missing the checkpoint policy binding")
    bound_policy_path = _resolve_path(policy_binding.get("path"), base=_REPO_ROOT)
    if policy_path is not None and bound_policy_path != policy_path.expanduser().resolve():
        raise ValueError("release protocol checkpoint policy path mismatch")
    policy_path = bound_policy_path
    policy = _load_json(policy_path)
    if policy.get("schema") != "sophia_pretrain_base_checkpoint_policy_v1":
        raise ValueError("unsupported checkpoint policy")
    bound_policy_sha256 = str(policy_binding.get("sha256") or "").lower()
    actual_policy_sha256 = _sha256(policy_path)
    if bound_policy_sha256 != actual_policy_sha256:
        raise ValueError(
            "release protocol checkpoint policy sha256 mismatch: "
            f"expected={bound_policy_sha256} actual={actual_policy_sha256}"
        )
    protocol["_path"] = str(protocol_path)
    protocol_sha = _sha256(protocol_path)
    row = _evaluate_candidate(evidence_path.expanduser().resolve(), policy, protocol)
    valid = bool(
        row.get("hard_gates_passed") and row.get("seed") == FORMAL_SEED
    )
    row["valid"] = valid
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "pass" if valid else "fail",
        "valid": valid,
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": _sha256(policy_path),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha,
        "evidence": row,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the final seed-42 checkpoint and its evaluation evidence.")
    parser.add_argument("--protocol", default="configs/eval/pretrain_release_protocol.json")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_final_checkpoint(
        protocol_path=Path(args.protocol),
        policy_path=Path(args.policy) if args.policy else None,
        evidence_path=Path(args.evidence),
    )
    write_json_atomic(args.output, report, ensure_ascii=False, sort_keys=True, make_parents=True)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
