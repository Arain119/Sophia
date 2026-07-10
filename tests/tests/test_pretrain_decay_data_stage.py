from __future__ import annotations

import pytest

from ml.tasks.pretrain.planner import (
    CurriculumStage,
    RuntimeRecipe,
    StagePlan,
    apply_decay_data_stage,
    wsd_decay_start_step,
)
from ml.training.pretrain.run_config import PretrainRunConfig


def _single_stage_plans(*, total_steps: int, tokens_per_update: int) -> list[StagePlan]:
    stage = CurriculumStage(
        stage_index=0,
        seq_len=4096,
        start_tokens=0,
        end_tokens=int(total_steps) * int(tokens_per_update),
    )
    recipe = RuntimeRecipe(
        seq_len=4096,
        batch_size=1,
        accumulation_steps=64,
        tokens_per_update=int(tokens_per_update),
        machine_recipe_tuned=1,
    )
    return [
        StagePlan(
            stage=stage,
            recipe=recipe,
            start_step=0,
            end_step=int(total_steps),
        )
    ]


def _args(**overrides: object) -> PretrainRunConfig:
    payload: dict[str, object] = {
        "data_path": "/data/train",
        "decay_data_path": "/data/decay_train",
        "lr_schedule": "wsd",
        "warmup_steps": 800,
        "warmup_ratio": 0.0,
        "wsd_stable_ratio": 0.9,
    }
    payload.update(overrides)
    return PretrainRunConfig(**payload)  # type: ignore[arg-type]


def test_wsd_decay_start_step_matches_scheduler_math() -> None:
    # warmup 800 + stable 0.9*10000=9000 -> capped at total-warmup=9200
    assert (
        wsd_decay_start_step(
            total_steps=10_000,
            warmup_steps=800,
            warmup_ratio=0.0,
            wsd_stable_ratio=0.9,
        )
        == 9_800
    )
    assert (
        wsd_decay_start_step(
            total_steps=60_000,
            warmup_steps=800,
            warmup_ratio=0.0,
            wsd_stable_ratio=0.9,
        )
        == 800 + 54_000
    )


def test_apply_decay_data_stage_splits_final_plan() -> None:
    plans = _single_stage_plans(total_steps=60_000, tokens_per_update=262_144)
    out = apply_decay_data_stage(stage_execution_plans=plans, args=_args())

    assert len(out) == 2
    stable, decay = out
    assert stable.start_step == 0
    assert stable.end_step == 54_800
    assert stable.data_path == ""
    assert decay.start_step == 54_800
    assert decay.end_step == 60_000
    assert decay.data_path == "/data/decay_train"
    # Step geometry and stage token accounting stay contiguous.
    assert stable.stage.end_tokens == decay.stage.start_tokens
    assert decay.stage.end_tokens == plans[0].stage.end_tokens
    # Recipes are inherited unchanged (including machine tuning).
    assert decay.recipe == plans[0].recipe
    # Stage indices are re-sequenced.
    assert [plan.stage.stage_index for plan in out] == [0, 1]


def test_apply_decay_data_stage_noop_without_decay_path() -> None:
    plans = _single_stage_plans(total_steps=60_000, tokens_per_update=262_144)
    out = apply_decay_data_stage(
        stage_execution_plans=plans,
        args=_args(decay_data_path=""),
    )
    assert out == plans


def test_apply_decay_data_stage_requires_wsd() -> None:
    plans = _single_stage_plans(total_steps=60_000, tokens_per_update=262_144)
    with pytest.raises(RuntimeError, match="requires lr_schedule=wsd"):
        apply_decay_data_stage(
            stage_execution_plans=plans,
            args=_args(lr_schedule="cosine"),
        )


def test_apply_decay_data_stage_rejects_empty_decay_window() -> None:
    plans = _single_stage_plans(total_steps=1_000, tokens_per_update=262_144)
    with pytest.raises(RuntimeError, match="no decay steps"):
        apply_decay_data_stage(
            stage_execution_plans=plans,
            args=_args(warmup_steps=0, wsd_stable_ratio=1.0),
        )
