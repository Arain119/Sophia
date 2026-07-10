from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from ml.training.pretrain.resources import PretrainDataIter
from ml.training.pretrain.engine.grad_clip import (
    clip_gradients_,
    grad_global_norm_,
    scale_grads_by_token_count_,
)
from ml.training.pretrain.engine.state_support import _capture_data_iter_state_or_raise
from ml.training.pretrain.train_state import PretrainTrainState
from ml.training.pretrain.engine.lr_schedule_state import LrScheduleStateConfig
from ml.training.pretrain.engine.step_runner_impl import MicroStepResult

if TYPE_CHECKING:
    from ml.training.ema import ModelEMA
    from ml.training.pretrain.engine.step_runner_impl import StepRunner


class LoopConfigLike(Protocol):
    accumulation_steps: int
    max_grad_norm: float
    grad_clip_mode: str
    agc_clip: float
    agc_eps: float
    agc_exclude_bias_and_norm: int


@dataclass(frozen=True)
class LoopControllers:
    lr_schedule_state_cfg: LrScheduleStateConfig


@dataclass
class LoopAccumulators:
    running_loss: torch.Tensor
    running_loss_sum: torch.Tensor
    running_supervised_tokens: torch.Tensor
    seen_supervised_tokens: torch.Tensor
    seen_supervised_tokens_value: int

    @classmethod
    def create(
        cls,
        *,
        device: torch.device,
        resume_seen_tokens: int,
    ) -> LoopAccumulators:
        return cls(
            running_loss=torch.zeros((), device=device, dtype=torch.float32),
            running_loss_sum=torch.zeros((), device=device, dtype=torch.float32),
            running_supervised_tokens=torch.zeros((), device=device, dtype=torch.int64),
            seen_supervised_tokens=torch.tensor(
                int(resume_seen_tokens),
                device=device,
                dtype=torch.int64,
            ),
            seen_supervised_tokens_value=int(resume_seen_tokens),
        )

    def apply_update(
        self,
        *,
        token_weighted_active: bool,
        micro_loss_sum: torch.Tensor,
        update_loss_sum: torch.Tensor,
        update_supervised_tokens: torch.Tensor,
        accumulation_steps: int,
    ) -> int:
        if token_weighted_active:
            self.running_loss_sum.add_(update_loss_sum)
            self.running_supervised_tokens.add_(update_supervised_tokens)
            self.seen_supervised_tokens.add_(update_supervised_tokens)
            update_supervised_tokens_value = int(update_supervised_tokens.detach().item())
            self.seen_supervised_tokens_value += int(update_supervised_tokens_value)
            return int(update_supervised_tokens_value)
        loss_tensor = micro_loss_sum / float(max(int(accumulation_steps), 1))
        self.running_loss.add_(loss_tensor)
        return 0

@dataclass(frozen=True)
class AccumulatedUpdate:
    token_weighted_active: bool
    micro_loss_sum: torch.Tensor
    update_loss_sum: torch.Tensor
    update_supervised_tokens: torch.Tensor


@dataclass(frozen=True)
class TrainUpdateResult:
    next_step: int
    loss_value: float
    grad_norm: torch.Tensor | None
    token_weighted_active: bool
    train_state: PretrainTrainState

    @property
    def global_step(self) -> int:
        return int(self.next_step)


@dataclass(frozen=True)
class ResolvedGradClipConfig:
    grad_clip_mode: str
    max_grad_norm: float
    agc_clip: float
    agc_eps: float
    agc_exclude_bias_and_norm: bool


def resolve_grad_clip_config(cfg: LoopConfigLike) -> ResolvedGradClipConfig:
    grad_clip_mode = str(cfg.grad_clip_mode).strip()
    if grad_clip_mode == "":
        grad_clip_mode = "norm"
    return ResolvedGradClipConfig(
        grad_clip_mode=grad_clip_mode,
        max_grad_norm=float(cfg.max_grad_norm),
        agc_clip=float(cfg.agc_clip),
        agc_eps=float(cfg.agc_eps),
        agc_exclude_bias_and_norm=bool(int(cfg.agc_exclude_bias_and_norm) == 1),
    )


def scale_grads_by_token_count(
    *,
    model: torch.nn.Module,
    supervised_tokens: torch.Tensor,
) -> None:
    scale_grads_by_token_count_(model=model, supervised_tokens=supervised_tokens)


def clip_gradients(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    cfg: LoopConfigLike,
) -> torch.Tensor | None:
    resolved_cfg = resolve_grad_clip_config(cfg)
    return clip_gradients_(
        model=model,
        optimizer=optimizer,
        grad_clip_mode=resolved_cfg.grad_clip_mode,
        max_grad_norm=resolved_cfg.max_grad_norm,
        agc_clip=resolved_cfg.agc_clip,
        agc_eps=resolved_cfg.agc_eps,
        agc_exclude_bias_and_norm=resolved_cfg.agc_exclude_bias_and_norm,
    )


def loss_value_for_update(
    *,
    token_weighted_active: bool,
    micro_loss_sum: torch.Tensor,
    update_loss_sum: torch.Tensor,
    update_supervised_tokens: torch.Tensor,
    accumulation_steps: int,
) -> float:
    if bool(token_weighted_active) and int(update_supervised_tokens.detach().item()) > 0:
        return float(
            (
                update_loss_sum / update_supervised_tokens.to(dtype=update_loss_sum.dtype)
            )
            .detach()
            .item()
        )
    return float(
        (micro_loss_sum / float(max(int(accumulation_steps), 1))).detach().item()
    )


def build_train_state(
    *,
    controllers: LoopControllers,
    seen_tokens: int,
    best_eval_loss: float | None,
    data_iter: PretrainDataIter | None = None,
) -> PretrainTrainState:
    data_iter_state = None
    if data_iter is not None:
        data_iter_state = _capture_data_iter_state_or_raise(
            data_iter,
            feature_name="checkpointable training data stream",
        )
    payload: dict[str, object] = {
        "seen_supervised_tokens": int(seen_tokens),
        "lr_decay_start_step": int(controllers.lr_schedule_state_cfg.lr_decay_start_step),
    }
    if best_eval_loss is not None:
        payload["best_eval_loss"] = float(best_eval_loss)
    if data_iter_state is not None:
        payload["data_iter_state"] = dict(data_iter_state)
    return PretrainTrainState.from_payload(payload)


def _accumulate_gradients_for_update(
    *,
    model: torch.nn.Module,
    data_iter: PretrainDataIter,
    runner: StepRunner,
    device: torch.device,
    accumulation_steps: int,
    token_weighted_cfg: bool,
) -> AccumulatedUpdate:
    micro_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    update_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    update_supervised_tokens = torch.zeros((), device=device, dtype=torch.int64)
    token_weighted_active = bool(token_weighted_cfg)

    for _micro_index in range(int(accumulation_steps)):
        batch = next(data_iter)
        micro_out = runner.run_micro(batch, accum_steps=int(accumulation_steps))
        if isinstance(micro_out, MicroStepResult):
            token_weighted_active = True
            loss_sum_det = micro_out.loss_sum_detached
            supervised_tokens = micro_out.supervised_tokens
            if loss_sum_det.device != device:
                loss_sum_det = loss_sum_det.to(device=device)
            if supervised_tokens.device != device:
                supervised_tokens = supervised_tokens.to(device=device)
            update_loss_sum.add_(loss_sum_det.to(dtype=torch.float32))
            update_supervised_tokens.add_(supervised_tokens.to(dtype=torch.int64))
            continue
        if token_weighted_active:
            raise RuntimeError(
                "token_weighted_loss is enabled but StepRunner returned a raw loss tensor. "
                "Use a token-weighted StepRunner implementation."
            )
        micro_loss_sum.add_(micro_out.to(dtype=torch.float32))

    return AccumulatedUpdate(
        token_weighted_active=bool(token_weighted_active),
        micro_loss_sum=micro_loss_sum,
        update_loss_sum=update_loss_sum,
        update_supervised_tokens=update_supervised_tokens,
    )


def run_train_update(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    data_iter: PretrainDataIter,
    runner: StepRunner,
    device: torch.device,
    cfg: LoopConfigLike,
    step: int,
    best_eval_loss: float | None,
    token_weighted_cfg: bool,
    ema: ModelEMA | None,
    controllers: LoopControllers,
    accumulators: LoopAccumulators,
    stability_check_fn: Callable[
        [float, torch.Tensor | None, torch.Tensor | None], None
    ]
    | None = None,
    build_train_state_fn: Callable[..., PretrainTrainState] = build_train_state,
) -> TrainUpdateResult:
    optimizer.zero_grad(set_to_none=bool(runner.zero_grad_set_to_none))
    runner.begin_update(accum_steps=int(cfg.accumulation_steps))
    accumulated = _accumulate_gradients_for_update(
        model=model,
        data_iter=data_iter,
        runner=runner,
        device=device,
        accumulation_steps=int(cfg.accumulation_steps),
        token_weighted_cfg=bool(token_weighted_cfg),
    )
    if accumulated.token_weighted_active:
        scale_grads_by_token_count(
            model=model,
            supervised_tokens=accumulated.update_supervised_tokens,
        )
    grad_norm = clip_gradients(model=model, optimizer=optimizer, cfg=cfg)
    post_clip_grad_norm = grad_global_norm_(model.parameters())
    loss_value = loss_value_for_update(
        token_weighted_active=bool(accumulated.token_weighted_active),
        micro_loss_sum=accumulated.micro_loss_sum,
        update_loss_sum=accumulated.update_loss_sum,
        update_supervised_tokens=accumulated.update_supervised_tokens,
        accumulation_steps=int(cfg.accumulation_steps),
    )
    if stability_check_fn is not None:
        stability_check_fn(float(loss_value), grad_norm, post_clip_grad_norm)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if ema is not None:
        ema.update()
    accumulators.apply_update(
        token_weighted_active=bool(accumulated.token_weighted_active),
        micro_loss_sum=accumulated.micro_loss_sum,
        update_loss_sum=accumulated.update_loss_sum,
        update_supervised_tokens=accumulated.update_supervised_tokens,
        accumulation_steps=int(cfg.accumulation_steps),
    )

    global_step = int(step) + 1
    return TrainUpdateResult(
        next_step=int(global_step),
        loss_value=float(loss_value),
        grad_norm=grad_norm,
        token_weighted_active=bool(accumulated.token_weighted_active),
        train_state=build_train_state_fn(
            controllers=controllers,
            seen_tokens=int(accumulators.seen_supervised_tokens_value),
            best_eval_loss=best_eval_loss,
        ),
    )


__all__ = [
    "LoopAccumulators",
    "LoopControllers",
    "TrainUpdateResult",
    "build_train_state",
    "run_train_update",
]
