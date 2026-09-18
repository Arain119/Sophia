"""Immutable engine run specifications."""

from __future__ import annotations

from dataclasses import dataclass

from ml.core.spec import ModelSpec


@dataclass(frozen=True)
class RunSpec:
    """Static intent for a task run.

    This contract replaces passing raw ``argparse.Namespace`` objects across
    package boundaries.
    """

    run_kind: str
    model: ModelSpec
    seed: int = 1337
    max_steps: int | None = None
    output_dir: str = ""
    resume_from_checkpoint: str | None = None
    safe_serialization: bool = True

    def __post_init__(self) -> None:
        normalized_kind = str(self.run_kind or "").strip().lower()
        if not normalized_kind:
            raise ValueError("run_kind must be non-empty")
        object.__setattr__(self, "run_kind", normalized_kind)

        if int(self.seed) < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.max_steps is not None and int(self.max_steps) <= 0:
            raise ValueError(
                f"max_steps must be > 0 when provided, got {self.max_steps}"
            )

        output_dir = str(self.output_dir or "").strip()
        object.__setattr__(self, "output_dir", output_dir)

        resume = self.resume_from_checkpoint
        resume = None if resume is None else str(resume).strip() or None
        object.__setattr__(self, "resume_from_checkpoint", resume)
