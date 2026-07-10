from __future__ import annotations

from ml.tooling.scripts import audit_precision_probe as mod


def test_precision_probe_report_records_bf16_gate() -> None:
    report = mod.build_precision_probe_report(
        acceleration_report={"capabilities": {"bf16_training_stack": True}},
        bf16_result={
            "mode": "bf16",
            "status": "passed",
            "reason": "finite_forward_backward",
            "loss": 1.0,
            "grad_norm": 2.0,
        },
    )

    gate = report["production_gate"]
    assert gate["release_pretrain_precision"] == "bf16"
    assert gate["release_pretrain_precision_stack"] == "bf16_sdpa_flash_liger"
    assert gate["selected_runtime_validation_status"] == "bf16_canary_pending_full_run"
    assert gate["selected_runtime_validation_requirements"] == [
        "runtime_context",
        "release_warmup",
    ]
    assert gate["bf16_baseline_passed"] is True
    assert gate["release_pretrain_safe"] is False
    assert gate["required_before_release"] == gate["selected_runtime_validation_requirements"]
    modes = {probe["mode"]: probe for probe in report["probes"]}
    assert modes["bf16"]["status"] == "passed"
    assert report["acceleration"]["capabilities"]["bf16_training_stack"] is True


def test_precision_probe_cuda_unavailable_skips_requested_cuda(monkeypatch) -> None:
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        mod,
        "build_acceleration_report",
        lambda: {"capabilities": {}},
    )

    assert mod.main(["--device", "cuda:0"]) == 0
