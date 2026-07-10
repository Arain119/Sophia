from __future__ import annotations

from ml.tooling.scripts import audit_acceleration as mod


def test_acceleration_audit_exposes_production_gate() -> None:
    report = mod.build_acceleration_report()

    assert report["kind"] == "acceleration_audit"
    gate = report["production_gate"]
    assert gate["release_pretrain_safe"] is False
    assert gate["release_pretrain_precision"] == "bf16"
    assert gate["release_pretrain_precision_stack"] == "bf16_sdpa_flash_liger"
    assert gate["selected_runtime_validation_status"] == "bf16_canary_pending_full_run"
    assert gate["selected_runtime_validation_requirements"] == [
        "runtime_context",
        "release_warmup",
    ]
    assert gate["required_actions"]


def test_acceleration_audit_recommends_bf16_sdpa_flash_liger_stack() -> None:
    report = mod.build_acceleration_report()

    assert report["recommendation"]["release_pretrain_precision"] == "bf16"
    assert (
        report["recommendation"]["release_pretrain_precision_stack"]
        == "bf16_sdpa_flash_liger"
    )
    assert (
        report["recommendation"]["selected_runtime_validation_status"]
        == "bf16_canary_pending_full_run"
    )
    assert "pending" in report["recommendation"]["reason"]
