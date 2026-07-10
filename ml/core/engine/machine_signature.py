from __future__ import annotations

import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ml.core.common.mapping import object_mapping


@dataclass(frozen=True)
class MachineAdaptiveSignature:
    schema: str
    device_type: str
    torch_version: str
    cuda_version: str
    platform_name: str
    machine: str
    cuda_device_name: str | None = None
    cuda_capability: tuple[int, int] | None = None
    cuda_total_memory_gb: float | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": str(self.schema),
            "device_type": str(self.device_type),
            "torch": str(self.torch_version),
            "cuda": str(self.cuda_version),
            "platform": str(self.platform_name),
            "machine": str(self.machine),
        }
        if self.cuda_device_name is not None:
            payload["cuda_device_name"] = str(self.cuda_device_name)
        if self.cuda_capability is not None:
            payload["cuda_capability"] = [
                int(self.cuda_capability[0]),
                int(self.cuda_capability[1]),
            ]
        if self.cuda_total_memory_gb is not None:
            payload["cuda_total_memory_gb"] = float(self.cuda_total_memory_gb)
        return payload

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        expected_schema: str | None = None,
    ) -> MachineAdaptiveSignature | None:
        raw_schema = str(payload.get("schema", "") or "")
        if not raw_schema:
            return None
        if expected_schema is not None and raw_schema != str(expected_schema):
            return None

        capability: tuple[int, int] | None = None
        raw_capability = payload.get("cuda_capability")
        if isinstance(raw_capability, (list, tuple)) and len(raw_capability) == 2:
            try:
                capability = (int(raw_capability[0]), int(raw_capability[1]))
            except (TypeError, ValueError):
                capability = None

        total_memory_gb: float | None = None
        raw_total_memory = payload.get("cuda_total_memory_gb")
        if raw_total_memory is not None:
            try:
                total_memory_gb = float(raw_total_memory)
            except (TypeError, ValueError):
                total_memory_gb = None

        raw_cuda_name = payload.get("cuda_device_name")
        cuda_device_name = None
        if raw_cuda_name is not None:
            text = str(raw_cuda_name or "").strip()
            cuda_device_name = text or None

        return cls(
            schema=raw_schema,
            device_type=str(payload.get("device_type", "") or ""),
            torch_version=str(payload.get("torch", "") or ""),
            cuda_version=(
                ""
                if "cuda" not in payload and payload.get("cuda") is None
                else str(payload.get("cuda"))
            ),
            platform_name=str(payload.get("platform", "") or ""),
            machine=str(payload.get("machine", "") or ""),
            cuda_device_name=cuda_device_name,
            cuda_capability=capability,
            cuda_total_memory_gb=total_memory_gb,
        )


def machine_signature_payload(
    signature: MachineAdaptiveSignature | Mapping[str, object] | None,
) -> dict[str, object]:
    if isinstance(signature, MachineAdaptiveSignature):
        return signature.to_payload()
    if isinstance(signature, Mapping):
        return {str(key): value for key, value in signature.items()}
    return {}


def _coerce_machine_signature(
    raw: object,
    *,
    expected_schema: str,
) -> MachineAdaptiveSignature | None:
    if isinstance(raw, MachineAdaptiveSignature):
        return raw if raw.schema == str(expected_schema) else None
    if isinstance(raw, Mapping):
        return MachineAdaptiveSignature.from_mapping(
            raw,
            expected_schema=expected_schema,
        )
    return None


def build_machine_adaptive_signature(
    *,
    schema: str,
    device: torch.device,
) -> MachineAdaptiveSignature:
    device_type = str(getattr(device, "type", "") or "")
    signature = MachineAdaptiveSignature(
        schema=str(schema),
        device_type=device_type,
        torch_version=str(getattr(torch, "__version__", "")),
        cuda_version=str(getattr(torch.version, "cuda", None)),
        platform_name=str(getattr(sys, "platform", "") or ""),
        machine=str(platform.machine() or ""),
    )
    if str(getattr(device, "type", "") or "") != "cuda":
        return signature
    try:
        idx = device.index
        if idx is None:
            idx = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(int(idx))
        cuda_device_name = str(getattr(props, "name", "") or "") or None
        cuda_capability = (
            int(getattr(props, "major", 0) or 0),
            int(getattr(props, "minor", 0) or 0),
        )
        total_mem = float(getattr(props, "total_memory", 0.0) or 0.0)
        return MachineAdaptiveSignature(
            schema=signature.schema,
            device_type=signature.device_type,
            torch_version=signature.torch_version,
            cuda_version=signature.cuda_version,
            platform_name=signature.platform_name,
            machine=signature.machine,
            cuda_device_name=cuda_device_name,
            cuda_capability=cuda_capability,
            cuda_total_memory_gb=round(total_mem / float(1024**3), 6),
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to read CUDA machine properties for device={device}"
        ) from exc


def ensure_machine_adaptive_signature(
    *,
    args: object,
    device: torch.device,
    arg_key: str,
    schema: str,
) -> MachineAdaptiveSignature:
    existing = _coerce_machine_signature(
        object_mapping(args).get(arg_key),
        expected_schema=str(schema),
    )
    if existing is not None:
        return existing
    signature = build_machine_adaptive_signature(schema=str(schema), device=device)
    setattr(args, arg_key, signature.to_payload())
    return signature


def resume_machine_signature(
    *,
    resume_args: Mapping[str, object] | None,
    arg_key: str,
    schema: str,
) -> MachineAdaptiveSignature | None:
    if not isinstance(resume_args, Mapping):
        return None
    return _coerce_machine_signature(
        resume_args.get(arg_key),
        expected_schema=str(schema),
    )


__all__ = [
    "MachineAdaptiveSignature",
    "build_machine_adaptive_signature",
    "ensure_machine_adaptive_signature",
    "machine_signature_payload",
    "resume_machine_signature",
]
