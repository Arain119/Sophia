"""Mutable run state contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from ml.core.engine.types import StatePayload


@dataclass
class RunState:
    """Mutable resumable execution state.

    The state is generic enough to wrap task-owned train state without forcing
    task-specific fields into the shared engine contract.
    """

    step: int = 0
    seen_tokens: int = 0
    train_state: StatePayload = field(default_factory=dict)
    optimizer_state: StatePayload = field(default_factory=dict)
    scheduler_state: StatePayload = field(default_factory=dict)
    rng_state: StatePayload = field(default_factory=dict)
    ema_state: StatePayload | None = None
    iterator_states: StatePayload = field(default_factory=dict)
    controller_states: StatePayload = field(default_factory=dict)
    best_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.step = int(self.step)
        self.seen_tokens = int(self.seen_tokens)
        if self.step < 0:
            raise ValueError(f"step must be >= 0, got {self.step}")
        if self.seen_tokens < 0:
            raise ValueError(
                f"seen_tokens must be >= 0, got {self.seen_tokens}"
            )

        self.train_state = dict(self.train_state)
        self.optimizer_state = dict(self.optimizer_state)
        self.scheduler_state = dict(self.scheduler_state)
        self.rng_state = dict(self.rng_state)
        self.iterator_states = dict(self.iterator_states)
        self.controller_states = dict(self.controller_states)
        self.best_metrics = {
            str(key): float(value) for key, value in dict(self.best_metrics).items()
        }
        if self.ema_state is not None:
            self.ema_state = dict(self.ema_state)
