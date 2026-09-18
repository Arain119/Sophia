from __future__ import annotations

from ml.tooling.scripts import audit_precision_probe as mod


def test_precision_probe_report_records_bf16_gate() -> None:
    report = mod.build_precision_probe_report(
        acceleration_report={"capabilities": {"cuda_bf16": True}},
        bf16_result={
            "mode": "bf16",
            "status": "passed",
            "reason": "finite_forward_backward",
            "loss": 1.0,
            "grad_norm": 2.0,
        },
    )

    assert report["contract"]["precision"] == "bf16"
    assert report["contract"]["precision_stack"] == "bf16_fla_kda_sdpa_mla_liger"
    assert report["passed"] is True
    modes = {probe["mode"]: probe for probe in report["probes"]}
    assert modes["bf16"]["status"] == "passed"
    assert report["acceleration"]["capabilities"]["cuda_bf16"] is True


def test_precision_probe_cuda_unavailable_skips_requested_cuda(monkeypatch) -> None:
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        mod,
        "build_acceleration_report",
        lambda: {"capabilities": {}},
    )

    assert mod.main(["--device", "cuda:0"]) == 1
