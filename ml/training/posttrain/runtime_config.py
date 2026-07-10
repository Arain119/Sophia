from __future__ import annotations

import json
from collections.abc import Callable

import torch

from ml.core.engine.checkpointing import EngineCheckpoint
from ml.core.engine.machine_signature import (
    MachineAdaptiveSignature,
    machine_signature_payload,
    resume_machine_signature as load_resume_machine_signature,
)
from ml.training.posttrain.types import PosttrainStageArgs
from ml.training.runtime_tools import apply_gradient_checkpointing

MACHINE_SIGNATURE_ARG_KEY = "_sophia_machine_signature"
POSTTRAIN_MACHINE_SIGNATURE_SCHEMA = "posttrain_machine_adaptive_v1"
POSTTRAIN_MACHINE_RECIPE_KIND = "posttrain_machine_recipe"


def maybe_apply_resume_machine_settings(
    *,
    args: PosttrainStageArgs,
    resume_checkpoint: EngineCheckpoint | None,
    current_machine_signature: MachineAdaptiveSignature | dict[str, object],
    model: torch.nn.Module | None = None,
    apply_gradient_checkpointing_fn: Callable[..., None] = apply_gradient_checkpointing,
) -> bool:
    current_signature = machine_signature_payload(current_machine_signature)
    if resume_checkpoint is None or not isinstance(resume_checkpoint.args, dict):
        return False
    resume_args = dict(resume_checkpoint.args)
    checkpoint_signature = load_resume_machine_signature(
        resume_args=resume_args,
        arg_key=MACHINE_SIGNATURE_ARG_KEY,
        schema=POSTTRAIN_MACHINE_SIGNATURE_SCHEMA,
    )
    if checkpoint_signature is None:
        print(
            "[INFO] resume checkpoint has no post-train machine signature; "
            "keeping current machine-layer settings.",
            flush=True,
        )
        return False
    if machine_signature_payload(checkpoint_signature) != current_signature:
        print(
            "[INFO] resume checkpoint machine signature mismatch; "
            "keeping current machine-layer settings.\n"
            f"checkpoint_machine={json.dumps(machine_signature_payload(checkpoint_signature), ensure_ascii=False, sort_keys=True)}\n"
            f"current_machine={json.dumps(current_signature, ensure_ascii=False, sort_keys=True)}",
            flush=True,
        )
        return False

    applied: dict[str, int] = {}
    for field in ("batch_size", "accumulation_steps", "gradient_checkpointing"):
        raw = resume_args.get(field)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        setattr(args, field, int(value))
        applied[field] = int(value)
    if model is not None and "gradient_checkpointing" in applied:
        apply_gradient_checkpointing_fn(
            model,
            enabled=bool(int(applied["gradient_checkpointing"]) == 1),
        )
    if applied:
        print(
            "[INFO] resume checkpoint machine signature matched; "
            f"reusing machine-layer settings {json.dumps(applied, ensure_ascii=False, sort_keys=True)}",
            flush=True,
        )
    return bool(applied)


__all__ = [
    "MACHINE_SIGNATURE_ARG_KEY",
    "POSTTRAIN_MACHINE_RECIPE_KIND",
    "POSTTRAIN_MACHINE_SIGNATURE_SCHEMA",
    "maybe_apply_resume_machine_settings",
]
