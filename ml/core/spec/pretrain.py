from __future__ import annotations

from dataclasses import dataclass

from .model import ModelSpec


@dataclass(frozen=True)
class PretrainScheduleSpec:
    max_seq_len: int
    train_seq_len: int


def build_release_pretrain_schedule_spec(
    *,
    model: ModelSpec | None = None,
) -> PretrainScheduleSpec:
    active_model = model or ModelSpec.default()
    max_seq_len = int(active_model.max_seq_len)
    return PretrainScheduleSpec(
        max_seq_len=max_seq_len,
        train_seq_len=4096,
    )


__all__ = [
    "PretrainScheduleSpec",
    "build_release_pretrain_schedule_spec",
]
