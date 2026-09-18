"""SFT artifact integrity checks."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

SFT_MACHINE_RECIPE_SCHEMA = "sophia_sft_machine_recipe_v1"
SFT_PROBE_AUDIT_SCHEMA = "sophia_sft_probe_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sft_export_artifacts(report: Mapping[str, object]) -> Path:
    output_dir = Path(str(report.get("output_dir", "") or "")).resolve()
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("export.artifacts is not a JSON object")
    required = {"config.json", "model.safetensors", "tokenizer.json"}
    if not required.issubset(artifacts):
        raise RuntimeError(
            "SFT export is missing required artifacts: "
            + ", ".join(sorted(required - set(artifacts)))
        )
    for relative, expected_hash in artifacts.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe SFT export artifact path: {relative!r}")
        path = (output_dir / relative_path).resolve()
        try:
            path.relative_to(output_dir)
        except ValueError as exc:
            raise RuntimeError(f"SFT export artifact escapes output_dir: {relative!r}") from exc
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise RuntimeError(f"SFT export artifact bytes changed: {relative}")
    return output_dir


__all__ = [
    "SFT_MACHINE_RECIPE_SCHEMA",
    "SFT_PROBE_AUDIT_SCHEMA",
    "sha256_file",
    "validate_sft_export_artifacts",
]
