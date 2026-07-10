from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

from ml.training.pretrain.release_config import RELEASE_PRETRAIN_PRECISION_POLICY


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(str(name)) is not None


def _cuda_report() -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    report: dict[str, Any] = {
        "available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "devices": [],
    }
    if not available:
        return report
    devices = []
    for idx in range(int(torch.cuda.device_count())):
        props = torch.cuda.get_device_properties(idx)
        capability = torch.cuda.get_device_capability(idx)
        devices.append(
            {
                "index": int(idx),
                "name": str(props.name),
                "capability": [int(capability[0]), int(capability[1])],
                "total_memory_gb": float(props.total_memory) / float(1024**3),
            }
        )
    report["devices"] = devices
    return report


def build_acceleration_report() -> dict[str, Any]:
    cuda = _cuda_report()
    policy = RELEASE_PRETRAIN_PRECISION_POLICY
    runtime_validation_status = str(policy.runtime_validation_status)
    runtime_validation_requirements = [
        str(item)
        for item in list(policy.runtime_validation_requirements)
        if str(item).strip()
    ]

    required_actions = [
        "Release pretrain must use the signed BF16 machine recipe.",
        "Use this audit together with torch-inductor audit before paid-GPU training.",
    ]
    if runtime_validation_requirements:
        required_actions.append(
            "Do not promote the selected BF16 runtime stack to release-safe "
            "status until these checks pass: "
            + ", ".join(runtime_validation_requirements)
            + "."
        )
    if not bool(cuda.get("available")):
        required_actions.append(
            "Run on the target GPU host to verify hardware-specific acceleration capabilities."
        )

    gate_payload = policy.gate_payload()
    recommendation_payload = policy.recommendation_payload()
    report = {
        "kind": "acceleration_audit",
        "torch": {
            "version": str(torch.__version__),
            "cuda_version": str(getattr(torch.version, "cuda", "") or ""),
        },
        "cuda": cuda,
        "capabilities": {
            "bf16_training_stack": True,
            "sdpa_flash_attention": True,
            "fused_projections": True,
            "liger_fused_linear_ce": _module_available(
                "liger_kernel.transformers.functional"
            ),
        },
        "production_gate": {
            "release_pretrain_safe": False,
            **gate_payload,
            "selected_runtime_validation_status": str(runtime_validation_status),
            "selected_runtime_validation_requirements": runtime_validation_requirements,
            "required_actions": required_actions,
        },
        "recommendation": {
            **recommendation_payload,
            "selected_runtime_validation_status": str(runtime_validation_status),
            "reason": (
                "Release pretrain uses BF16 tensors, PyTorch SDPA Flash attention, "
                "fused QKV/gate-up projections, and Liger fused linear CE. "
                "It remains pending target CUDA GPU runtime evidence before "
                "it can be treated as release-safe."
            ),
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit local GPU acceleration capabilities."
    )
    parser.add_argument("--output_json", default="", type=str)
    parsed = parser.parse_args(argv)

    report = build_acceleration_report()
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if str(parsed.output_json or "").strip():
        path = Path(str(parsed.output_json)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
