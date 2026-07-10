from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace

from ml.errors import SophiaUsageError
from ml.training.pretrain.machine_runtime import PretrainMachineRuntime
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.training.pretrain.value_semantics import (
    coerce_int,
    is_disabled_flag,
    is_non_positive_int,
    normalized_str,
)


_AUTO_UNSET = object()
_MACHINE_ADAPTIVE_RUNTIME_PAYLOAD_KEY = "_sophia_machine_runtime"
_RESUME_RECIPE_FIELDS: tuple[str, ...] = (
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "min_lr_ratio",
    "wsd_stable_ratio",
    "muon_target_rms",
    "embedding_lr_scale",
    "total_tokens",
    "target_tokens_per_update",
    "warmup_steps",
    "max_steps",
    "muon_ns_steps",
    "seq_len",
    "auto_batch_size_max",
    "log_interval",
    "eval_interval",
    "eval_steps",
    "save_interval",
    "save_total_limit",
    "save_best",
    "async_checkpoint",
    "async_metrics",
    "ema_update_interval",
    "ema_eval",
    "ema_in_ckpt",
    "ema_use_for_export",
    "lr_schedule",
    "wsd_decay_style",
    "grad_clip_mode",
    "ckpt_staging_dir",
    "layerwise_lr_decay",
    "max_grad_norm",
    "agc_clip",
    "agc_eps",
    "agc_exclude_bias_and_norm",
    "ema_decay",
)
_CURRENT_RECIPE_OWNED_FIELDS: frozenset[str] = frozenset(
    {
        "grad_clip_mode",
        "max_grad_norm",
        "agc_clip",
        "agc_eps",
        "agc_exclude_bias_and_norm",
    }
)
_RESUME_MACHINE_RECIPE_RULES: tuple[tuple[str, Callable[[object], bool]], ...] = (
    ("batch_size", is_non_positive_int),
    ("accumulation_steps", is_non_positive_int),
    ("gradient_checkpointing", is_disabled_flag),
    ("gradient_checkpointing_exclude_first", is_disabled_flag),
    ("gradient_checkpointing_exclude_last", is_disabled_flag),
    ("loss_chunk_size", is_disabled_flag),
)
_RESUME_AUTO_RUNTIME_FIELDS: tuple[str, ...] = (
    "dataloader_num_workers",
    "dataloader_prefetch_factor",
    "dataloader_persistent_workers",
    "shard_preload",
    "shard_preload_bytes",
)
def _apply_resume_field(
    *,
    args: PretrainRunConfig,
    state: PretrainRuntimeState,
    resume_args: dict[str, object],
    field_name: str,
    default_if_missing: object = _AUTO_UNSET,
) -> PretrainRunConfig:
    raw = resume_args.get(field_name, _AUTO_UNSET)
    if raw is _AUTO_UNSET:
        if default_if_missing is _AUTO_UNSET:
            return args
        raw = default_if_missing
    try:
        if PretrainRuntimeState.owns_field(str(field_name)):
            setattr(state, str(field_name), raw)
            return state.project_run_config(args)
        return replace(args, **{str(field_name): raw})
    except (AttributeError, TypeError, ValueError):
        print(
            f"[WARN] resume args {field_name} invalid: {raw!r}; ignoring.",
            flush=True,
        )
    return args


def apply_resume_recipe_settings(
    *,
    args: PretrainRunConfig,
    resume_args: dict[str, object] | None,
    runtime_state: PretrainRuntimeState | None = None,
    current_recipe_owned_fields: frozenset[str] = _CURRENT_RECIPE_OWNED_FIELDS,
) -> PretrainRunConfig:
    state = (
        runtime_state
        if runtime_state is not None
        else PretrainRuntimeState.from_config(args)
    )
    if not isinstance(resume_args, dict):
        return args
    resolved_args = args
    for field_name in _RESUME_RECIPE_FIELDS:
        if str(field_name) in current_recipe_owned_fields:
            continue
        resolved_args = _apply_resume_field(
            args=resolved_args,
            state=state,
            resume_args=resume_args,
            field_name=field_name,
        )
    return resolved_args


def apply_resume_machine_recipe_settings(
    *,
    args: PretrainRunConfig,
    resume_args: dict[str, object] | None,
    runtime_state: PretrainRuntimeState | None = None,
) -> PretrainRunConfig:
    state = (
        runtime_state
        if runtime_state is not None
        else PretrainRuntimeState.from_config(args)
    )
    if not isinstance(resume_args, dict):
        return args
    resolved_args = args
    for field_name, should_restore in _RESUME_MACHINE_RECIPE_RULES:
        if not should_restore(getattr(state, field_name)):
            continue
        resolved_args = _apply_resume_field(
            args=resolved_args,
            state=state,
            resume_args=resume_args,
            field_name=field_name,
            default_if_missing=0,
        )
    return resolved_args


def apply_resume_runtime_settings(
    *,
    args: PretrainRunConfig,
    resume_args: dict[str, object] | None,
    runtime_state: PretrainRuntimeState | None = None,
) -> PretrainRunConfig:
    state = (
        runtime_state
        if runtime_state is not None
        else PretrainRuntimeState.from_config(args)
    )
    if not isinstance(resume_args, dict):
        return args
    resolved_args = args
    for field_name in _RESUME_AUTO_RUNTIME_FIELDS:
        current = coerce_int(getattr(state, field_name), default=-1)
        if current >= 0:
            continue
        resolved_args = _apply_resume_field(
            args=resolved_args,
            state=state,
            resume_args=resume_args,
            field_name=field_name,
        )
    state.apply_machine_runtime(
        PretrainMachineRuntime.from_source(
            resume_args.get(_MACHINE_ADAPTIVE_RUNTIME_PAYLOAD_KEY)
        )
    )
    return state.project_run_config(resolved_args)


def machine_runtime_payload(
    *,
    machine_runtime: PretrainMachineRuntime | None,
) -> dict[str, object]:
    if machine_runtime is None:
        return {}
    payload = machine_runtime.to_payload()
    return payload if payload else {}


def validate_resume_data_path(
    *,
    args: PretrainRunConfig,
    resume_args: dict[str, object] | None,
) -> None:
    if not isinstance(resume_args, dict):
        return
    prev_data_path = normalized_str(resume_args.get("data_path"))
    cur_data_path = normalized_str(args.data_path)
    if not prev_data_path or not cur_data_path:
        return
    prev_abs = os.path.normpath(os.path.abspath(prev_data_path))
    cur_abs = os.path.normpath(os.path.abspath(cur_data_path))
    if os.path.normcase(prev_abs) == os.path.normcase(cur_abs):
        return
    raise SophiaUsageError(
        "[ERR] resume checkpoint data_path mismatch.\n"
        f"Checkpoint: {prev_abs}\n"
        f"Current:    {cur_abs}\n"
        "Delete the output_dir and restart to run on a different dataset."
    )


__all__ = [
    "apply_resume_machine_recipe_settings",
    "apply_resume_recipe_settings",
    "apply_resume_runtime_settings",
    "machine_runtime_payload",
    "validate_resume_data_path",
]
