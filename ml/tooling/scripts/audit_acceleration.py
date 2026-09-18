from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from ml.training.pretrain.release_config import release_pretrain_runtime_metadata


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(str(name)) is not None
    except (ImportError, AttributeError, ValueError):
        return False


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


def _sdpa_backend_report(*, cuda_available: bool) -> dict[str, Any]:
    if not cuda_available:
        return {
            "checked": False,
            "non_math_backend": False,
            "error": "CUDA is unavailable",
        }
    query: torch.Tensor | None = None
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        query = torch.empty(
            (1, 16, 4096, 128), device="cuda", dtype=torch.bfloat16
        )
        with sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        ):
            functional.scaled_dot_product_attention(
                query,
                query,
                query,
                is_causal=True,
            )
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - target CUDA only
        return {
            "checked": True,
            "non_math_backend": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if query is not None:
            del query
    return {"checked": True, "non_math_backend": True, "error": ""}


def build_acceleration_report() -> dict[str, Any]:
    cuda = _cuda_report()
    cuda_available = bool(cuda.get("available"))
    sdpa = _sdpa_backend_report(cuda_available=cuda_available)
    return {
        "kind": "acceleration_audit",
        "contract": release_pretrain_runtime_metadata(),
        "torch": {
            "version": str(torch.__version__),
            "cuda_version": str(getattr(torch.version, "cuda", "") or ""),
        },
        "cuda": cuda,
        "capabilities": {
            "cuda_bf16": bool(
                cuda_available and torch.cuda.is_bf16_supported()
            ),
            "flash_sdp_enabled": bool(
                cuda_available and torch.backends.cuda.flash_sdp_enabled()
            ),
            "scaled_dot_product_attention": callable(
                torch.nn.functional.scaled_dot_product_attention
            ),
            "sdpa_non_math_backend": bool(sdpa["non_math_backend"]),
            "sdpa_backend_check": sdpa,
            "liger_fused_linear_ce": _module_available(
                "liger_kernel.transformers.functional"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit local GPU acceleration capabilities."
    )
    parser.add_argument("--output_json", default="", type=str)
    parsed = parser.parse_args(argv)

    report = build_acceleration_report()
    if bool(report["cuda"].get("available")) and not bool(
        report["capabilities"].get("sdpa_non_math_backend")
    ):
        raise SystemExit(
            "FATAL: scaled_dot_product_attention did not pass the non-math backend check."
        )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if str(parsed.output_json or "").strip():
        path = Path(str(parsed.output_json)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
