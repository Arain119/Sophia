from __future__ import annotations

from ml.tooling.scripts import eval_tokenizer_quality_gates as mod


def _report(*, tokens_per_char: float, provenance: bool = True) -> dict:
    return {
        "model": {
            "vocab_size": 57344,
            "model_max_length": 16384,
            "byte_fallback": True,
        },
        "summary": {
            "unk_token_rate": 0.0,
            "byte_fallback_rate": 0.0,
            "roundtrip_failures": 0,
            "missing_special_token_count": 0,
            "provenance_present": provenance,
            "training_report_present": True,
        },
        "domains": {"native_chinese": {"tokens_per_char": tokens_per_char}},
        "throughput": {"chars_per_second": 100.0},
    }


def test_gate_requires_improved_fertility_and_provenance() -> None:
    gates = {
        "required_vocab_size": 57344,
        "required_model_max_length": 16384,
        "require_byte_fallback": True,
        "require_provenance": True,
        "require_training_report": True,
        "max_unk_token_rate": 0.0,
        "max_byte_fallback_rate": 0.02,
        "max_roundtrip_failures": 0,
        "max_missing_special_tokens": 0,
        "minimum_throughput_ratio": 0.7,
        "fertility": {
            "native_chinese": {"minimum_relative_improvement": 0.02}
        },
    }

    passed = mod.evaluate_tokenizer_gates(
        gates=gates,
        baseline=_report(tokens_per_char=0.6),
        candidate=_report(tokens_per_char=0.55),
    )
    failed = mod.evaluate_tokenizer_gates(
        gates=gates,
        baseline=_report(tokens_per_char=0.6),
        candidate=_report(tokens_per_char=0.6, provenance=False),
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert "provenance.present" in failed["summary"]["failed_checks"]
    assert "fertility.native_chinese" in failed["summary"]["failed_checks"]


def test_gate_accepts_float_roundoff_at_exact_regression_boundary() -> None:
    gates = {
        "required_vocab_size": 57344,
        "required_model_max_length": 16384,
        "require_byte_fallback": True,
        "require_provenance": True,
        "require_training_report": True,
        "max_unk_token_rate": 0.0,
        "max_byte_fallback_rate": 0.02,
        "max_roundtrip_failures": 0,
        "max_missing_special_tokens": 0,
        "minimum_throughput_ratio": 0.7,
        "fertility": {
            "native_chinese": {"maximum_relative_regression": 0.1}
        },
    }

    report = mod.evaluate_tokenizer_gates(
        gates=gates,
        baseline=_report(tokens_per_char=10.0),
        candidate=_report(tokens_per_char=11.0),
    )

    assert report["passed"] is True


def test_gate_binds_candidate_bundle_and_sample_count() -> None:
    candidate = _report(tokens_per_char=0.5)
    candidate["bundle"] = {"sha1": "a" * 40}
    candidate["suite"] = {"sample_count": 120}
    gates = {
        "required_vocab_size": 57344,
        "required_model_max_length": 16384,
        "require_byte_fallback": True,
        "require_provenance": True,
        "require_training_report": True,
        "max_unk_token_rate": 0.0,
        "max_byte_fallback_rate": 0.02,
        "max_roundtrip_failures": 0,
        "max_missing_special_tokens": 0,
        "minimum_throughput_ratio": 0.7,
        "required_bundle_sha1": "a" * 40,
        "minimum_sample_count": 100,
        "fertility": {"native_chinese": {"minimum_relative_improvement": 0.02}},
    }
    report = mod.evaluate_tokenizer_gates(
        gates=gates,
        baseline=_report(tokens_per_char=0.6),
        candidate=candidate,
    )
    assert report["passed"] is True

    candidate["bundle"]["sha1"] = "b" * 40
    failed = mod.evaluate_tokenizer_gates(
        gates=gates,
        baseline=_report(tokens_per_char=0.6),
        candidate=candidate,
    )
    assert failed["passed"] is False
    assert "protocol.bundle_sha1" in failed["summary"]["failed_checks"]
