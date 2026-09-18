"""The pipeline snapshot the runtime bootstrap reads.

It lives beside the runtime that consumes it rather than beside the session
that fills it: every field it carries is a ``ml.training.pretrain`` type, and
having the bootstrap reach up into ``ml.tasks`` for it was the one import that
made the two packages depend on each other.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from ml.training.pretrain.resources import (
    PretrainManifestLike,
    PretrainTokenizerLike,
)
from ml.training.pretrain.value_semantics import coerce_int, coerce_str

if TYPE_CHECKING:
    from ml.training.pretrain.resume_loader import PretrainResumeState
    from ml.training.pretrain.engine.step_runner_impl import StepRunner
else:
    PretrainResumeState = object
    StepRunner = object


def _coerce_optional_str(value: object) -> object:
    return None if value is None else str(value)


_SNAPSHOT_FIELD_NAMES = frozenset(
    {
        "output_dir",
        "resume_path",
        "manifest_path",
        "manifest",
        "tokenizer",
        "tokenizer_path",
        "seq_len",
        "vocab_size",
        "model",
        "resume",
        "max_steps",
        "runner",
        "target_tokens_per_update",
    }
)
_SNAPSHOT_RUNTIME_STATE_OVERRIDE_FIELDS = (
    "seq_len",
    "max_steps",
    "target_tokens_per_update",
)


@dataclass(frozen=True)
class PretrainPipelineSnapshot:
    output_dir: str = ""
    resume_path: str | None = None
    manifest_path: str = ""
    manifest: PretrainManifestLike | None = None
    tokenizer: PretrainTokenizerLike | None = None
    tokenizer_path: str = ""
    seq_len: int = 0
    vocab_size: int = 0
    model: torch.nn.Module | None = None
    resume: PretrainResumeState | None = None
    max_steps: int = 0
    runner: StepRunner | None = None
    target_tokens_per_update: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", coerce_str(self.output_dir))
        object.__setattr__(self, "resume_path", _coerce_optional_str(self.resume_path))
        object.__setattr__(self, "manifest_path", coerce_str(self.manifest_path))
        object.__setattr__(self, "tokenizer_path", coerce_str(self.tokenizer_path))
        object.__setattr__(self, "seq_len", coerce_int(self.seq_len))
        object.__setattr__(self, "vocab_size", coerce_int(self.vocab_size))
        object.__setattr__(self, "max_steps", coerce_int(self.max_steps))
        object.__setattr__(
            self,
            "target_tokens_per_update",
            coerce_int(self.target_tokens_per_update),
        )

    def updated(self, field_name: str, value: object) -> PretrainPipelineSnapshot:
        resolved_field_name = str(field_name)
        if resolved_field_name not in _SNAPSHOT_FIELD_NAMES:
            raise AttributeError(
                f"unknown pretrain pipeline snapshot field: {resolved_field_name!r}"
            )
        return replace(self, **{resolved_field_name: value})

    def runtime_state_overrides(self) -> dict[str, int]:
        overrides: dict[str, int] = {}
        for field_name in _SNAPSHOT_RUNTIME_STATE_OVERRIDE_FIELDS:
            value = int(getattr(self, field_name))
            if value > 0:
                overrides[field_name] = value
        return overrides


__all__ = ["PretrainPipelineSnapshot"]
