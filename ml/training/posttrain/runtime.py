from __future__ import annotations

import contextlib
from collections.abc import Callable
import os

import torch

from ml.integrations.export.artifacts import export_model_artifacts
from ml.integrations.export.model_dir import (
    load_local_export_tokenizer,
    load_trainable_decoder_from_export_dir as load_trainable_decoder,
)
from ml.core.engine.artifacts import prepare_output_dir_and_resume
from ml.core.engine.machine_signature import (
    build_machine_adaptive_signature,
    machine_signature_payload,
)
from ml.core.engine.checkpointing import EngineCheckpoint, save_checkpoint, load_checkpoint
from ml.core.engine.context import RunContext
from ml.core.engine.recipe_loader import (
    apply_machine_recipe_from_path,
    load_json_object_from_path,
)
from ml.core.engine.bootstrap import build_run_artifacts, write_run_artifacts
from ml.runtime.stack import ensure_standard_stack
from ml.runtime.torch_env import require_cuda, setup_seed, setup_torch_backends
from ml.training.ema import ModelEMA
from ml.training.posttrain import runtime_config as runtime_config_mod
from ml.training.posttrain.contracts import SupportsPosttrainTokenizer
from ml.training.posttrain.release_gate import (
    run_posttrain_release_preflight,
    validate_resume_checkpoint_contract,
)
from ml.training.runtime_tools import (
    apply_gradient_checkpointing,
    create_optimizer,
    move_optimizer_state_to_device,
)
from ml.core.common.rng import capture_rng_state, restore_rng_state
from ml.training.scheduler import build_lr_scheduler
from ml.training.posttrain.types import (
    PosttrainCheckpointSpec,
    PosttrainCheckpointPolicy,
    PosttrainEmaConfig,
    PosttrainExportSpec,
    PosttrainLaunchConfig,
    PosttrainOptimizerConfig,
    PosttrainReportSpec,
    PosttrainRunContext,
    PosttrainRunSpec,
    PosttrainStageArgs,
    ResumeState,
)
from ml.training.posttrain.runtime_config import (
    MACHINE_SIGNATURE_ARG_KEY,
    POSTTRAIN_MACHINE_SIGNATURE_SCHEMA,
    POSTTRAIN_MACHINE_RECIPE_KIND,
)

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

POSTTRAIN_EXPORT_DIRNAME = "export"


def setup_stage_runtime(
    *,
    output_dir: str,
    resume_path: str | None,
    launch: PosttrainLaunchConfig,
) -> RunContext:
    stack_versions = ensure_standard_stack(mode="train", require_tensorboard=True)
    base_dtype = torch.bfloat16
    if base_dtype != torch.bfloat16:
        raise RuntimeError("post-training runtime is pinned to BF16 base dtype")
    device_str = require_cuda(str(launch.requested_device))
    device = torch.device(device_str)
    setup_torch_backends()
    setup_seed(int(launch.seed))
    return RunContext(
        output_dir=str(output_dir),
        device=device,
        base_dtype=base_dtype,
        runtime_metadata={
            "stack_versions": dict(stack_versions),
            "precision_stack": "pinned",
        },
        machine_signature=build_machine_adaptive_signature(
            schema=POSTTRAIN_MACHINE_SIGNATURE_SCHEMA,
            device=device,
        ),
        resume_path=resume_path,
        resolved_device=str(device_str),
    )
def maybe_resume_model(
    *,
    model: torch.nn.Module,
    resume_path: str | None,
) -> ResumeState:
    if resume_path is None:
        return ResumeState(checkpoint=None, start_step=0, train_state={})
    ckpt = load_checkpoint(str(resume_path))
    model.load_state_dict(ckpt.model, strict=True)
    train_state = dict(ckpt.train_state or {})
    print(f"[INFO] resumed checkpoint: {resume_path}", flush=True)
    return ResumeState(
        checkpoint=ckpt,
        start_step=int(ckpt.step),
        train_state=train_state,
    )


def maybe_restore_rng_from_checkpoint(resume_checkpoint: EngineCheckpoint | None) -> None:
    if resume_checkpoint is None or resume_checkpoint.rng is None:
        return
    restore_rng_state(resume_checkpoint.rng)


def build_optimizer_scheduler(
    *,
    model: torch.nn.Module,
    config: PosttrainOptimizerConfig,
    max_steps: int,
    resume_checkpoint: EngineCheckpoint | None,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = create_optimizer(
        model,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        betas=(float(config.beta1), float(config.beta2)),
        eps=float(config.adam_eps),
        layerwise_lr_decay=float(config.layerwise_lr_decay),
        muon_ns_steps=config.muon_ns_steps,
        muon_target_rms=config.muon_target_rms,
    )
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint.optimizer)
        move_optimizer_state_to_device(optimizer, str(device))
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        max_steps=int(max_steps),
        warmup_steps=int(config.warmup_steps),
        warmup_ratio=float(config.warmup_ratio),
        min_lr_ratio=float(config.min_lr_ratio),
        schedule=str(config.lr_schedule),
        wsd_stable_ratio=float(config.wsd_stable_ratio),
        wsd_decay_style=str(config.wsd_decay_style),
        resume_state=(
            None if resume_checkpoint is None else resume_checkpoint.scheduler
        ),
    )
    return optimizer, scheduler


def maybe_build_ema(
    *,
    model: torch.nn.Module,
    config: PosttrainEmaConfig,
    resume_checkpoint: EngineCheckpoint | None,
) -> ModelEMA | None:
    decay = float(config.decay)
    if not (0.0 < decay < 1.0):
        return None
    ema = ModelEMA(
        model,
        decay=float(decay),
        update_interval=int(config.update_interval),
    )
    if resume_checkpoint is not None and resume_checkpoint.ema is not None:
        ema.load_state_dict(resume_checkpoint.ema)
    return ema
def save_stage_checkpoint(
    *,
    checkpoint: PosttrainCheckpointSpec,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ema: ModelEMA | None,
) -> None:
    save_checkpoint(
        output_dir=str(checkpoint.output_dir),
        step=int(checkpoint.step),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args=dict(checkpoint.args_payload),
        rng=capture_rng_state(),
        ema=(None if ema is None else ema.state_dict()),
        train_state=dict(checkpoint.train_state),
        save_total_limit=int(checkpoint.save_total_limit),
        staging_dir=checkpoint.staging_dir,
    )


def finalize_posttrain_run(
    *,
    export: PosttrainExportSpec,
    model: torch.nn.Module,
    tokenizer: SupportsPosttrainTokenizer,
    ema: ModelEMA | None,
    report_callback: Callable[[], None] | None = None,
) -> None:
    ctx = ema.apply_to_model() if ema is not None else contextlib.nullcontext()
    with ctx:
        if callable(report_callback):
            report_callback()
        export_model_artifacts(
            model=model,
            tokenizer=tokenizer,
            output_dir=os.path.join(str(export.output_dir), POSTTRAIN_EXPORT_DIRNAME),
            safe_serialization=bool(export.safe_serialization),
            ema=None,
        )


def finalize_posttrain_stage_outputs(
    *,
    runtime: PosttrainRunContext,
    report: PosttrainReportSpec,
    report_fn: Callable[[dict[str, str]], None] | None = None,
) -> None:
    def _report() -> None:
        if not callable(report_fn):
            return
        if not str(report.report_name or "").strip():
            return
        if not any(str(path or "").strip() for path in report.split_paths.values()):
            return
        report_fn(dict(report.split_paths))

    finalize_posttrain_run(
        export=runtime.export,
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        ema=runtime.ema,
        report_callback=_report,
    )


def initialize_posttrain_run(
    *,
    run_spec: PosttrainRunSpec,
    runtime_args: PosttrainStageArgs,
) -> PosttrainRunContext:
    output_dir, resume_path = prepare_output_dir_and_resume(
        repo_root=str(REPO_ROOT),
        output_dir=str(run_spec.launch.output_dir),
        resume_from_checkpoint=str(run_spec.launch.resume_from_checkpoint),
        overwrite_output_dir=bool(run_spec.launch.overwrite_output_dir),
        required_output_dir_message="[ERR] --output_dir is required for post-training stages.",
    )

    runtime_ctx = setup_stage_runtime(
        output_dir=str(output_dir),
        resume_path=resume_path,
        launch=run_spec.launch,
    )
    tokenizer = load_local_export_tokenizer(
        str(run_spec.launch.export_dir),
        padding_side="right",
        truncation_side="left",
        model_max_length=int(run_spec.resolved_max_seq_len),
    )
    model = load_trainable_decoder(
        export_dir=str(run_spec.launch.export_dir),
        device=runtime_ctx.device,
        base_dtype=runtime_ctx.base_dtype,
        gradient_checkpointing=bool(run_spec.launch.gradient_checkpointing),
        apply_gradient_checkpointing_fn=apply_gradient_checkpointing,
    )
    resume = maybe_resume_model(model=model, resume_path=resume_path)
    validate_resume_checkpoint_contract(
        args=runtime_args,
        stage=str(run_spec.run.run_kind),
        resolved_max_seq_len=int(run_spec.resolved_max_seq_len),
        resume_checkpoint=resume.checkpoint,
    )
    runtime_config_mod.maybe_apply_resume_machine_settings(
        args=runtime_args,
        resume_checkpoint=resume.checkpoint,
        current_machine_signature=runtime_ctx.machine_signature,
        model=model,
        apply_gradient_checkpointing_fn=apply_gradient_checkpointing,
    )
    maybe_restore_rng_from_checkpoint(resume.checkpoint)

    optimizer, scheduler = build_optimizer_scheduler(
        model=model,
        config=run_spec.optimizer,
        max_steps=int(run_spec.launch.max_steps),
        resume_checkpoint=resume.checkpoint,
        device=runtime_ctx.device,
    )
    ema = maybe_build_ema(
        model=model,
        config=run_spec.ema,
        resume_checkpoint=resume.checkpoint,
    )

    start_step = int(resume.start_step)
    resolved_output_dir = os.path.abspath(str(output_dir))
    base_dtype = str(runtime_ctx.base_dtype).replace("torch.", "")
    run_args_payload = dict(vars(runtime_args))
    run_args_payload["output_dir"] = resolved_output_dir
    run_args_payload["_sophia_run_kind"] = str(run_spec.run.run_kind)
    run_args_payload["_sophia_resolved_max_seq_len"] = int(run_spec.resolved_max_seq_len)
    run_args_payload["_sophia_stack_versions"] = dict(
        runtime_ctx.runtime_metadata.get("stack_versions", {})
    )
    run_args_payload["_sophia_base_dtype"] = str(base_dtype)
    run_args_payload["_sophia_precision_stack"] = str(
        runtime_ctx.runtime_metadata.get("precision_stack", "pinned")
    )
    run_args_payload["_sophia_resolved_device"] = str(runtime_ctx.resolved_device)
    run_args_payload["_sophia_resolved_resume_checkpoint"] = (
        None if resume_path is None else str(resume_path)
    )
    run_args_payload[MACHINE_SIGNATURE_ARG_KEY] = machine_signature_payload(
        runtime_ctx.machine_signature
    )
    write_run_artifacts(
        build_run_artifacts(
            output_dir=str(output_dir),
            args_payload=run_args_payload,
            run_kind=str(run_spec.run.run_kind),
            start_step=int(start_step),
            max_steps=int(run_spec.launch.max_steps),
            repo_root=REPO_ROOT,
            device=runtime_ctx.device,
            stack_versions=dict(runtime_ctx.runtime_metadata.get("stack_versions", {})),
            extra_meta={
                "resume_checkpoint": None if resume_path is None else str(resume_path),
                "base_dtype": str(runtime_ctx.base_dtype).replace("torch.", ""),
                "precision_stack": str(runtime_ctx.runtime_metadata.get("precision_stack", "pinned")),
            },
        )
    )
    return PosttrainRunContext(
        run=runtime_ctx,
        args=runtime_args,
        tokenizer=tokenizer,
        model=model,
        resume=resume,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        start_step=int(start_step),
        run_args_payload=run_args_payload,
        checkpoint_policy=run_spec.checkpoint_policy,
        export=PosttrainExportSpec(
            output_dir=str(output_dir),
            safe_serialization=bool(run_spec.safe_serialization),
        ),
    )


def prepare_posttrain_stage_runtime(
    *,
    args: PosttrainStageArgs,
    stage: str,
    resolve_max_seq_len: Callable[[PosttrainStageArgs], int],
) -> tuple[int, PosttrainRunContext]:
    normalized_stage = str(stage or "").strip().lower()
    if not normalized_stage:
        raise ValueError("stage must be non-empty")
    resolved_max_seq_len = int(resolve_max_seq_len(args))
    loaded_recipe = load_json_object_from_path(
        path_label="machine_recipe",
        recipe_path=str(args.machine_recipe_json or ""),
    )
    run_posttrain_release_preflight(
        args=args,
        stage=normalized_stage,
        resolved_max_seq_len=int(resolved_max_seq_len),
        recipe_payload=(None if loaded_recipe is None else loaded_recipe.payload),
    )
    apply_machine_recipe_from_path(
        args=args,
        stage=normalized_stage,
        recipe_path=str(args.machine_recipe_json or ""),
        kind=POSTTRAIN_MACHINE_RECIPE_KIND,
        recipe_key="machine_recipe",
        int_fields=("batch_size", "accumulation_steps", "gradient_checkpointing"),
        log_label="post-train",
    )
    run_spec = args.build_run_spec(
        stage=normalized_stage,
        resolved_max_seq_len=int(resolved_max_seq_len),
    )
    runtime = initialize_posttrain_run(
        run_spec=run_spec,
        runtime_args=args,
    )
    return int(resolved_max_seq_len), runtime


__all__ = [
    "PosttrainCheckpointPolicy",
    "PosttrainCheckpointSpec",
    "PosttrainEmaConfig",
    "PosttrainExportSpec",
    "PosttrainLaunchConfig",
    "PosttrainOptimizerConfig",
    "PosttrainReportSpec",
    "PosttrainRunContext",
    "PosttrainRunSpec",
    "PosttrainStageArgs",
    "ResumeState",
    "finalize_posttrain_run",
    "finalize_posttrain_stage_outputs",
    "initialize_posttrain_run",
    "maybe_build_ema",
    "maybe_restore_rng_from_checkpoint",
    "maybe_resume_model",
    "prepare_posttrain_stage_runtime",
    "save_stage_checkpoint",
    "setup_stage_runtime",
]
