from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from ml.core.common.io import write_json_atomic


REPORT_SCHEMA = "sophia_tokenizer_quality_gate_report_v1"
_BOUNDARY_TOLERANCE = 1e-12


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    actual: object,
    expectation: object,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "actual": actual,
            "expectation": expectation,
        }
    )


def evaluate_tokenizer_gates(
    *,
    gates: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    candidate_model = candidate.get("model")
    candidate_summary = candidate.get("summary")
    baseline_domains = baseline.get("domains")
    candidate_domains = candidate.get("domains")
    if not all(
        isinstance(value, dict)
        for value in (
            candidate_model,
            candidate_summary,
            baseline_domains,
            candidate_domains,
        )
    ):
        raise ValueError("tokenizer audit reports are missing required objects")

    scalar_checks = (
        (
            "protocol.vocab_size",
            int(candidate_model.get("vocab_size", 0)),
            int(gates["required_vocab_size"]),
            "eq",
        ),
        (
            "protocol.model_max_length",
            int(candidate_model.get("model_max_length") or 0),
            int(gates["required_model_max_length"]),
            "ge",
        ),
        (
            "quality.unk_token_rate",
            float(candidate_summary.get("unk_token_rate", math.inf)),
            float(gates["max_unk_token_rate"]),
            "le",
        ),
        (
            "quality.byte_fallback_rate",
            float(candidate_summary.get("byte_fallback_rate", math.inf)),
            float(gates["max_byte_fallback_rate"]),
            "le",
        ),
        (
            "quality.roundtrip_failures",
            int(candidate_summary.get("roundtrip_failures", -1)),
            int(gates["max_roundtrip_failures"]),
            "le",
        ),
        (
            "protocol.missing_special_tokens",
            int(candidate_summary.get("missing_special_token_count", -1)),
            int(gates["max_missing_special_tokens"]),
            "le",
        ),
    )
    for name, actual, expected, comparison in scalar_checks:
        passed = actual == expected if comparison == "eq" else (
            actual >= expected if comparison == "ge" else actual <= expected
        )
        _check(
            checks,
            name=name,
            passed=passed,
            actual=actual,
            expectation={comparison: expected},
        )

    required_bundle_sha1 = str(gates.get("required_bundle_sha1") or "").strip()
    if required_bundle_sha1:
        bundle = candidate.get("bundle")
        actual_bundle_sha1 = (
            str(bundle.get("sha1") or "").strip()
            if isinstance(bundle, dict)
            else ""
        )
        _check(
            checks,
            name="protocol.bundle_sha1",
            passed=actual_bundle_sha1 == required_bundle_sha1,
            actual=actual_bundle_sha1,
            expectation={"eq": required_bundle_sha1},
        )

    minimum_samples = int(gates.get("minimum_sample_count", 0) or 0)
    if minimum_samples > 0:
        suite = candidate.get("suite")
        actual_samples = int(suite.get("sample_count", 0) or 0) if isinstance(suite, dict) else 0
        _check(
            checks,
            name="protocol.minimum_sample_count",
            passed=actual_samples >= minimum_samples,
            actual=actual_samples,
            expectation={"ge": minimum_samples},
        )

    baseline_suite = baseline.get("suite")
    candidate_suite = candidate.get("suite")
    baseline_suite_sha256 = (
        str(baseline_suite.get("sha256") or "").strip()
        if isinstance(baseline_suite, dict)
        else ""
    )
    candidate_suite_sha256 = (
        str(candidate_suite.get("sha256") or "").strip()
        if isinstance(candidate_suite, dict)
        else ""
    )
    required_suite_sha256 = str(gates.get("required_suite_sha256") or "").strip()
    if required_suite_sha256:
        _check(
            checks,
            name="protocol.suite_sha256",
            passed=candidate_suite_sha256 == required_suite_sha256,
            actual=candidate_suite_sha256,
            expectation={"eq": required_suite_sha256},
        )
    if bool(gates.get("require_same_suite", False)):
        _check(
            checks,
            name="protocol.baseline_candidate_same_suite",
            passed=(
                bool(baseline_suite_sha256)
                and baseline_suite_sha256 == candidate_suite_sha256
            ),
            actual={
                "baseline": baseline_suite_sha256,
                "candidate": candidate_suite_sha256,
            },
            expectation={"equal": True},
        )
    boolean_checks = (
        (
            "protocol.byte_fallback_enabled",
            bool(candidate_model.get("byte_fallback", False)),
            bool(gates.get("require_byte_fallback", False)),
        ),
        (
            "provenance.present",
            bool(candidate_summary.get("provenance_present", False)),
            bool(gates.get("require_provenance", False)),
        ),
        (
            "training_report.present",
            bool(candidate_summary.get("training_report_present", False)),
            bool(gates.get("require_training_report", False)),
        ),
    )
    for name, actual, required in boolean_checks:
        _check(
            checks,
            name=name,
            passed=actual or not required,
            actual=actual,
            expectation={"required": required},
        )

    fertility_gates = gates.get("fertility")
    if not isinstance(fertility_gates, dict):
        raise ValueError("gates.fertility must be an object")
    for domain, domain_gate in fertility_gates.items():
        baseline_row = baseline_domains.get(domain)
        candidate_row = candidate_domains.get(domain)
        if not isinstance(baseline_row, dict) or not isinstance(candidate_row, dict):
            raise ValueError(f"missing tokenizer audit domain: {domain}")
        baseline_value = float(baseline_row["tokens_per_char"])
        candidate_value = float(candidate_row["tokens_per_char"])
        relative_change = (candidate_value - baseline_value) / baseline_value
        if "minimum_relative_improvement" in domain_gate:
            threshold = float(domain_gate["minimum_relative_improvement"])
            actual = -relative_change
            passed = actual >= threshold or math.isclose(
                actual,
                threshold,
                rel_tol=_BOUNDARY_TOLERANCE,
                abs_tol=_BOUNDARY_TOLERANCE,
            )
            expectation = {"minimum_relative_improvement": threshold}
        else:
            threshold = float(domain_gate["maximum_relative_regression"])
            actual = relative_change
            passed = actual <= threshold or math.isclose(
                actual,
                threshold,
                rel_tol=_BOUNDARY_TOLERANCE,
                abs_tol=_BOUNDARY_TOLERANCE,
            )
            expectation = {"maximum_relative_regression": threshold}
        _check(
            checks,
            name=f"fertility.{domain}",
            passed=passed,
            actual={
                "baseline_tokens_per_char": baseline_value,
                "candidate_tokens_per_char": candidate_value,
                "relative_change": relative_change,
                "gate_value": actual,
            },
            expectation=expectation,
        )

    baseline_throughput = float(baseline["throughput"]["chars_per_second"])
    candidate_throughput = float(candidate["throughput"]["chars_per_second"])
    throughput_ratio = candidate_throughput / baseline_throughput
    minimum_ratio = float(gates["minimum_throughput_ratio"])
    _check(
        checks,
        name="throughput.same_machine_ratio",
        passed=throughput_ratio >= minimum_ratio,
        actual=throughput_ratio,
        expectation={"minimum": minimum_ratio},
    )
    failed = [row["name"] for row in checks if not row["passed"]]
    return {
        "schema": REPORT_SCHEMA,
        "status": "pass" if not failed else "fail",
        "passed": not failed,
        "summary": {
            "check_count": len(checks),
            "passed_count": len(checks) - len(failed),
            "failed_count": len(failed),
            "failed_checks": failed,
        },
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare tokenizer audits against gates.")
    parser.add_argument("--gates", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = evaluate_tokenizer_gates(
        gates=_load(Path(args.gates)),
        baseline=_load(Path(args.baseline)),
        candidate=_load(Path(args.candidate)),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(str(output), report)
    print(
        f"tokenizer gate status={report['status']} "
        f"passed={report['summary']['passed_count']}/"
        f"{report['summary']['check_count']} output={output}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
