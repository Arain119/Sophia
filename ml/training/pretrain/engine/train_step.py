from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from ml.training.pretrain.resources import PretrainDataIter
from ml.training.pretrain.engine.gradients import (
    global_grad_norm,
    scale_gradients_by_token_count,
)
from ml.training.pretrain.engine.state_support import _capture_data_iter_state_or_raise
from ml.training.pretrain.train_state import PretrainTrainState
if TYPE_CHECKING:
    from ml.training.pretrain.engine.step_runner_impl import StepRunner


class LoopConfigLike(Protocol):
    accumulation_steps: int


def _iter_tensor_values(value: object) -> Iterator[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensor_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensor_values(item)


@torch.no_grad()
def check_finite_update_state(
    *, model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    finite_checker = getattr(optimizer, "check_finite_state", None)
    if callable(finite_checker):
        # Custom optimizers own model/master pairs and nested inner-optimizer
        # state. Let them perform one complete check rather than scanning the
        # same state through the generic optimizer.state mapping.
        finite_checker()
        return

    entries: list[tuple[str, torch.Tensor]] = [
        (f"parameter[{index}]", parameter.data)
        for index, parameter in enumerate(model.parameters())
        if parameter.is_floating_point() or parameter.is_complex()
    ]
    for index, state in enumerate(optimizer.state.values()):
        entries.extend(
            (f"optimizer.state[{index}].tensor[{tensor_index}]", tensor)
            for tensor_index, tensor in enumerate(_iter_tensor_values(state))
            if tensor.is_floating_point() or tensor.is_complex()
        )

    if not entries:
        return
    device = entries[0][1].device
    all_finite = torch.ones((), device=device, dtype=torch.bool)
    for _name, tensor in entries:
        all_finite = all_finite & torch.isfinite(tensor).all().to(device=device)
    if bool(all_finite.item()):
        return
    for name, tensor in entries:
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise RuntimeError(f"non-finite update state: {name}")
    raise RuntimeError("non-finite update state")


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
class TrainUpdateResult:
    next_step: int
    loss_value: float
    grad_norm: torch.Tensor | None
    token_weighted_active: bool
    train_state: PretrainTrainState

    @property
    def global_step(self) -> int:
        return int(self.next_step)


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
    }
    if best_eval_loss is not None:
        payload["best_eval_loss"] = float(best_eval_loss)
    if data_iter_state is not None:
        payload["data_iter_state"] = dict(data_iter_state)
    return PretrainTrainState.from_payload(payload)


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
    accumulators: LoopAccumulators,
    stability_check_fn: Callable[[float, torch.Tensor | None], None]
    | None = None,
    post_update_stability_check_fn: Callable[[float, torch.Tensor | None], None]
    | None = None,
    build_train_state_fn: Callable[..., PretrainTrainState] = build_train_state,
) -> TrainUpdateResult:
    optimizer.zero_grad(set_to_none=bool(runner.zero_grad_set_to_none))
    accumulated = runner.run_update(
        data_iter=data_iter,
        device=device,
        accumulation_steps=int(cfg.accumulation_steps),
        token_weighted_cfg=bool(token_weighted_cfg),
    )
    gradient_pairs = runner.gradient_pairs()
    gradient_buffers = None
    if gradient_pairs is not None:
        gradient_buffers = tuple(gradient for _parameter, gradient in gradient_pairs)
        set_gradient_buffers = getattr(optimizer, "set_gradient_buffers", None)
        if not callable(set_gradient_buffers):
            raise RuntimeError(
                "the selected step runner requires an optimizer with explicit "
                "gradient-buffer support"
            )
        set_gradient_buffers(gradient_pairs)
    if accumulated.token_weighted_active:
        scale_gradients_by_token_count(
            model=model,
            supervised_tokens=accumulated.update_supervised_tokens,
            gradients=gradient_buffers,
        )
    grad_norm = global_grad_norm(model.parameters(), gradients=gradient_buffers)
    loss_value = loss_value_for_update(
        token_weighted_active=bool(accumulated.token_weighted_active),
        micro_loss_sum=accumulated.micro_loss_sum,
        update_loss_sum=accumulated.update_loss_sum,
        update_supervised_tokens=accumulated.update_supervised_tokens,
        accumulation_steps=int(cfg.accumulation_steps),
    )
    if stability_check_fn is not None:
        stability_check_fn(float(loss_value), grad_norm)

    runner.prepare_for_optimizer_step(optimizer)
    optimizer.step()
    runner.post_optimizer_step(optimizer)
    if scheduler is not None:
        scheduler.step()
    if post_update_stability_check_fn is not None:
        post_update_stability_check_fn(float(loss_value), grad_norm)
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
            seen_tokens=int(accumulators.seen_supervised_tokens_value),
            best_eval_loss=best_eval_loss,
        ),
    )


__all__ = [
    "LoopAccumulators",
    "TrainUpdateResult",
    "build_train_state",
    "check_finite_update_state",
    "run_train_update",
]
