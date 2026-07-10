from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LrScheduleStateConfig:
    """Persisted WSD decay boundary.

    The live LR scheduler recomputes its decay boundary from warmup/stable
    ratio every run; this value is round-tripped through the checkpoint so a
    resume reports the same boundary it was launched with (see
    ``build_lr_schedule_state_config``).
    """

    lr_decay_start_step: int = 0


__all__ = ["LrScheduleStateConfig"]
