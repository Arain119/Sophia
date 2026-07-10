from __future__ import annotations

from dataclasses import dataclass
import json

import torch

from ml.errors import SophiaUsageError
from ml.core.engine.checkpointing import EngineCheckpoint, load_checkpoint
from ml.core.engine.machine_signature import (
    ensure_machine_adaptive_signature,
    machine_signature_payload,
    resume_machine_signature,
)
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.training.pretrain.resume_restore import (
    apply_resume_machine_recipe_settings,
    apply_resume_recipe_settings,
    apply_resume_runtime_settings,
    validate_resume_data_path,
)
from ml.core.common.rng import restore_rng_state
from ml.training.pretrain.model_setup import sync_runtime_batch_capacity
from ml.training.pretrain.model_runtime_controls import (
    require_runtime_recipe_control,
)
from ml.training.pretrain.train_state import PretrainTrainState
from ml.training.pretrain.semantic_defaults import (
    learning_rate_for_tokens_per_update,
)
from ml.training.pretrain.batch_size_autofit import (
    apply_gradient_checkpointing,
    create_optimizer,
    move_optimizer_state_to_device,
)

PRETRAIN_MACHINE_SIGNATURE_SCHEMA = "pretrain_machine_adaptive_v1"


def _require_matching_machine_signature(
    *,
    args: PretrainRunConfig,
    resume_args: dict[str, object] | None,
    device: torch.device,
    machine_signature_arg_key: str,
) -> None:
    current_signature = ensure_machine_adaptive_signature(
        args=args,
        device=device,
        arg_key=str(machine_signature_arg_key),
        schema=PRETRAIN_MACHINE_SIGNATURE_SCHEMA,
    )
    checkpoint_signature = resume_machine_signature(
        resume_args=resume_args,
        arg_key=str(machine_signature_arg_key),
        schema=PRETRAIN_MACHINE_SIGNATURE_SCHEMA,
    )
    if checkpoint_signature is None:
        raise SophiaUsageError(
            "[ERR] resume checkpoint has no machine signature. "
            "Refuse to infer target-machine settings."
        )
    if checkpoint_signature == current_signature:
        return
    raise SophiaUsageError(
        "[ERR] resume checkpoint machine signature mismatch.\n"
        f"checkpoint_machine={json.dumps(machine_signature_payload(checkpoint_signature), ensure_ascii=False, sort_keys=True)}\n"
        f"current_machine={json.dumps(machine_signature_payload(current_signature), ensure_ascii=False, sort_keys=True)}"
    )


@dataclass(frozen=True)
class PretrainResumeState:
    ckpt: EngineCheckpoint | None
    optimizer: torch.optim.Optimizer | None
    start_step: int
    resume_scheduler_state: dict[str, object] | None
    resume_args: dict[str, object] | None
    resume_train_state: PretrainTrainState
    resolved_args: PretrainRunConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resume_train_state",
            PretrainTrainState.resolve(self.resume_train_state),
        )


def maybe_resume_from_checkpoint(
    *,
    args: PretrainRunConfig,
    model: torch.nn.Module,
    resume_path: str | None,
    device: torch.device,
    machine_signature_arg_key: str,
    runtime_state: PretrainRuntimeState | None = None,
    ckpt: EngineCheckpoint | None = None,
    load_checkpoint_fn=None,
    create_optimizer_fn=None,
    move_optimizer_state_to_device_fn=None,
    restore_rng_state_fn=None,
    apply_gradient_checkpointing_fn=None,
    sync_runtime_batch_capacity_fn=None,
) -> PretrainResumeState:
    if resume_path is None:
        return PretrainResumeState(
            ckpt=None,
            optimizer=None,
            start_step=0,
            resume_scheduler_state=None,
            resume_args=None,
            resume_train_state=PretrainTrainState(),
            resolved_args=None,
        )

    if load_checkpoint_fn is None:
        load_checkpoint_fn = load_checkpoint
    if create_optimizer_fn is None:
        create_optimizer_fn = create_optimizer
    if move_optimizer_state_to_device_fn is None:
        move_optimizer_state_to_device_fn = move_optimizer_state_to_device
    if restore_rng_state_fn is None:
        restore_rng_state_fn = restore_rng_state
    if apply_gradient_checkpointing_fn is None:
        apply_gradient_checkpointing_fn = apply_gradient_checkpointing
    if sync_runtime_batch_capacity_fn is None:
        sync_runtime_batch_capacity_fn = sync_runtime_batch_capacity

    ckpt = load_checkpoint_fn(str(resume_path)) if ckpt is None else ckpt
    start_step = int(ckpt.step)
    model.load_state_dict(ckpt.model, strict=True)
    resume_args = ckpt.args
    if not isinstance(resume_args, dict):
        raise SophiaUsageError("[ERR] resume checkpoint args must be a dictionary.")
    _require_matching_machine_signature(
        args=args,
        resume_args=resume_args if isinstance(resume_args, dict) else None,
        device=device,
        machine_signature_arg_key=machine_signature_arg_key,
    )
    resolved_args = apply_resume_recipe_settings(
        args=args,
        resume_args=resume_args,
        runtime_state=runtime_state,
    )
    resolved_args = apply_resume_machine_recipe_settings(
        args=resolved_args,
        resume_args=resume_args,
        runtime_state=runtime_state,
    )
    args = resolved_args
    state = (
        runtime_state
        if runtime_state is not None
        else PretrainRuntimeState.from_config(args)
    )
    tokens_per_update = (
        int(state.seq_len)
        * max(int(state.batch_size), 1)
        * max(int(state.accumulation_steps), 1)
    )
    optimizer = create_optimizer_fn(
        model,
        lr=float(
            learning_rate_for_tokens_per_update(
                semantic_learning_rate=float(state.learning_rate),
                tokens_per_update=int(tokens_per_update),
                target_tokens_per_update=int(state.target_tokens_per_update),
            )
        ),
        weight_decay=float(args.weight_decay),
        betas=(float(args.beta1), float(args.beta2)),
        eps=float(args.adam_eps),
        layerwise_lr_decay=float(args.layerwise_lr_decay),
        muon_ns_steps=int(args.muon_ns_steps),
        muon_target_rms=(
            None if args.muon_target_rms is None else float(args.muon_target_rms)
        ),
    )
    try:
        optimizer.load_state_dict(ckpt.optimizer)
    except Exception as exc:
        raise SophiaUsageError(
            f"[ERR] failed to load optimizer state for torch_muon_hybrid checkpoint: {exc}"
        ) from exc
    move_optimizer_state_to_device_fn(optimizer, str(device))
    print("[OPT] optimizer=torch_muon_hybrid (resumed)", flush=True)

    resume_scheduler_state = ckpt.scheduler
    resume_rng_state = ckpt.rng
    resume_train_state = PretrainTrainState.from_payload(ckpt.train_state)

    validate_resume_data_path(
        args=args,
        resume_args=resume_args if isinstance(resume_args, dict) else None,
    )

    if resume_rng_state is not None:
        restore_rng_state_fn(resume_rng_state)

    resolved_args = apply_resume_runtime_settings(
        args=resolved_args,
        resume_args=resume_args,
        runtime_state=runtime_state,
    )
    args = resolved_args

    require_runtime_recipe_control(model).apply_runtime_recipe_knobs(
        loss_chunk_size=int(state.loss_chunk_size),
        gradient_checkpointing_exclude_first=int(
            state.gradient_checkpointing_exclude_first
        ),
        gradient_checkpointing_exclude_last=int(
            state.gradient_checkpointing_exclude_last
        ),
    )

    apply_gradient_checkpointing_fn(
        model, enabled=bool(int(state.gradient_checkpointing) == 1)
    )
    sync_runtime_batch_capacity_fn(model=model, args=args)
    print(f"[INFO] resumed from checkpoint: {resume_path} (step={start_step})")

    return PretrainResumeState(
        ckpt=ckpt,
        optimizer=optimizer,
        start_step=int(start_step),
        resume_scheduler_state=resume_scheduler_state,
        resume_args=resume_args if isinstance(resume_args, dict) else None,
        resume_train_state=resume_train_state,
        resolved_args=resolved_args,
    )


__all__ = [
    "PretrainResumeState",
    "maybe_resume_from_checkpoint",
]
