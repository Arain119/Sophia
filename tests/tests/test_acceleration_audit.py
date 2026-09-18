from __future__ import annotations

from ml.tooling.scripts import audit_acceleration as mod


def test_module_available_handles_missing_parent_package(monkeypatch) -> None:
    def _raise_missing(_name: str):
        raise ModuleNotFoundError("missing parent package")

    monkeypatch.setattr(mod.importlib.util, "find_spec", _raise_missing)

    assert mod._module_available("missing_parent.child") is False


def test_acceleration_audit_reports_contract_and_measured_capabilities() -> None:
    report = mod.build_acceleration_report()

    assert report["kind"] == "acceleration_audit"
    assert report["contract"]["precision"] == "bf16"
    assert report["contract"]["precision_stack"] == "bf16_fla_kda_sdpa_mla_liger"
    assert isinstance(report["capabilities"]["cuda_bf16"], bool)
    assert isinstance(report["capabilities"]["flash_sdp_enabled"], bool)
    assert isinstance(report["capabilities"]["sdpa_non_math_backend"], bool)
    assert isinstance(report["capabilities"]["sdpa_backend_check"], dict)
