from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ml.tasks.pretrain.planner import CurriculumStage, StagePlan

if TYPE_CHECKING:
    from ml.training.pretrain.runtime_state import PretrainRuntimeState


@dataclass
class PretrainRuntimePlan:
    curriculum_stages: list[CurriculumStage] = field(default_factory=list)
    stage_execution_plans: list[StagePlan] = field(default_factory=list)
    _runtime_state: PretrainRuntimeState | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.curriculum_stages = list(self.curriculum_stages)
        self.stage_execution_plans = list(self.stage_execution_plans)

    @classmethod
    def create(
        cls,
        *,
        curriculum_stages: list[CurriculumStage] | None = None,
        stage_execution_plans: list[StagePlan] | None = None,
    ) -> PretrainRuntimePlan:
        return cls(
            curriculum_stages=(
                [] if curriculum_stages is None else list(curriculum_stages)
            ),
            stage_execution_plans=(
                [] if stage_execution_plans is None else list(stage_execution_plans)
            ),
        )

    def bind_runtime_state(self, runtime_state: PretrainRuntimeState) -> None:
        self._runtime_state = runtime_state

    @property
    def max_steps(self) -> int:
        if self._runtime_state is not None:
            return int(self._runtime_state.max_steps)
        return 0

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        if self._runtime_state is not None:
            self._runtime_state.max_steps = int(value)

    @property
    def target_tokens_per_update(self) -> int:
        if self._runtime_state is not None:
            return int(self._runtime_state.target_tokens_per_update)
        return 0

    @target_tokens_per_update.setter
    def target_tokens_per_update(self, value: int) -> None:
        if self._runtime_state is not None:
            self._runtime_state.target_tokens_per_update = int(value)


__all__ = ["PretrainRuntimePlan"]
