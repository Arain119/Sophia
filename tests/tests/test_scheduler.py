import math

import pytest
import torch

from ml.training.scheduler import build_lr_scheduler, cosine_lr_lambda


def test_release_cosine_trajectory() -> None:
    total_steps = 15_259
    warmup_steps = 89

    assert cosine_lr_lambda(
        step=0,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=0.1,
    ) == 0.0
    assert cosine_lr_lambda(
        step=44,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=0.1,
    ) == pytest.approx(44 / 89)
    assert cosine_lr_lambda(
        step=warmup_steps,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=0.1,
    ) == 1.0

    midpoint = warmup_steps + (total_steps - warmup_steps) // 2
    progress = (midpoint - warmup_steps) / (total_steps - warmup_steps)
    expected = 0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))
    assert cosine_lr_lambda(
        step=midpoint,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=0.1,
    ) == pytest.approx(expected)
    assert cosine_lr_lambda(
        step=total_steps,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=0.1,
    ) == pytest.approx(0.1)


def test_resume_restores_saved_lr_to_optimizer() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=8.825e-4)
    initial = build_lr_scheduler(
        optimizer=optimizer,
        max_steps=15_259,
        warmup_steps=89,
        warmup_ratio=0.0,
        min_lr_ratio=0.1,
        schedule="cosine",
        resume_state=None,
    )
    state = initial.state_dict()
    state["last_epoch"] = 7_000
    state["_step_count"] = 7_001
    state["_last_lr"] = [4e-4]

    resumed_optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(()))], lr=8.825e-4
    )
    resumed_optimizer.param_groups[0]["initial_lr"] = 8.825e-4
    resumed = build_lr_scheduler(
        optimizer=resumed_optimizer,
        max_steps=15_259,
        warmup_steps=89,
        warmup_ratio=0.0,
        min_lr_ratio=0.1,
        schedule="cosine",
        resume_state=state,
    )

    assert resumed.last_epoch == 7_000
    assert resumed.get_last_lr() == pytest.approx([4e-4])
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(4e-4)


def test_resume_rejects_checkpoint_step_mismatch() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=8.825e-4)
    initial = build_lr_scheduler(
        optimizer=optimizer,
        max_steps=15_259,
        warmup_steps=89,
        warmup_ratio=0.0,
        min_lr_ratio=0.1,
        schedule="cosine",
        resume_state=None,
    )
    state = initial.state_dict()
    state["last_epoch"] = 7_000
    state["_step_count"] = 7_001
    state["_last_lr"] = [4e-4]

    resumed_optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(()))], lr=8.825e-4
    )
    resumed_optimizer.param_groups[0]["initial_lr"] = 8.825e-4
    with pytest.raises(ValueError, match="does not match checkpoint step"):
        build_lr_scheduler(
            optimizer=resumed_optimizer,
            max_steps=15_259,
            warmup_steps=89,
            warmup_ratio=0.0,
            min_lr_ratio=0.1,
            schedule="cosine",
            resume_state=state,
            resume_step=7_001,
        )


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_scheduler_rejects_invalid_ratios(value: float) -> None:
    with pytest.raises(ValueError, match="min_lr_ratio"):
        cosine_lr_lambda(
            step=1,
            total_steps=10,
            warmup_steps=1,
            min_lr_ratio=value,
        )


@pytest.mark.parametrize("step", [-1, 11])
def test_scheduler_rejects_step_outside_declared_run(step: int) -> None:
    with pytest.raises(ValueError, match="step must be"):
        cosine_lr_lambda(
            step=step,
            total_steps=10,
            warmup_steps=1,
            min_lr_ratio=0.1,
        )


def test_scheduler_rejects_non_cosine_schedule() -> None:
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
    with pytest.raises(ValueError, match="lr_schedule must be 'cosine'"):
        build_lr_scheduler(
            optimizer=optimizer,
            max_steps=10,
            warmup_steps=1,
            warmup_ratio=0.0,
            min_lr_ratio=0.1,
            schedule="linear",
            resume_state=None,
        )
