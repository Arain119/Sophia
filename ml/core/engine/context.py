"""Runtime-bound execution context contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from ml.core.engine.machine_signature import (
    machine_signature_payload,
)
from ml.core.engine.types import RuntimeMetadata


@dataclass(frozen=True)
class RunContext:
    """Resolved runtime environment for a task run."""

    output_dir: str
    device: torch.device
    base_dtype: torch.dtype
    tokenizer: object | None = None
    runtime_metadata: RuntimeMetadata = field(default_factory=dict)
    machine_signature: dict[str, object] = field(default_factory=dict)
    resume_path: str | None = None
    resolved_device: str = ""

    def __post_init__(self) -> None:
        output_dir = str(self.output_dir or "").strip()
        if not output_dir:
            raise ValueError("output_dir must be non-empty")
        object.__setattr__(self, "output_dir", output_dir)

        resume = self.resume_path
        resume = None if resume is None else str(resume).strip() or None
        object.__setattr__(self, "resume_path", resume)
        object.__setattr__(self, "runtime_metadata", dict(self.runtime_metadata))
        object.__setattr__(
            self,
            "machine_signature",
            machine_signature_payload(self.machine_signature),
        )

        resolved_device = str(self.resolved_device or "").strip()
        if not resolved_device:
            resolved_device = str(self.device)
        object.__setattr__(self, "resolved_device", resolved_device)
