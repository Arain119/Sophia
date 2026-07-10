from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

import ml.tasks.pretrain.planner as stage_plan
from ml.core.engine.checkpointing import load_checkpoint
from ml.core.engine.recipe_loader import load_json_object_from_path
from ml.errors import SophiaUsageError
from ml.tasks.pretrain.deps import PretrainDeps
from ml.tasks.pretrain.run_spec_builder import build_pretrain_run_spec
from ml.tasks.pretrain.session import (
    PretrainSession,
    attach_pretrain_session,
    ensure_pretrain_session,
    normalize_pretrain_pipeline_args,
)
from ml.tasks.pretrain.stage_flow import (
    PretrainStageCursor,
    run_planned_stages,
    run_resume_boundaries,
)
from ml.tasks.pretrain.training_setup import build_pretrain_loop_setup
from ml.training.pretrain import runtime_bootstrap
from ml.training.pretrain.model_setup import (
    build_decoder_config,
    load_or_init_model,
    sync_runtime_batch_capacity,
)
from ml.training.pretrain.release_gate import run_pretrain_release_preflight
from ml.training.pretrain.resume_loader import maybe_resume_from_checkpoint
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.training.pretrain.batch_size_autofit import create_optimizer

if TYPE_CHECKING:
    from ml.tasks.pretrain.pipeline import PretrainPipeline


def setup_runtime_impl(
    pipeline: PretrainPipeline,
    *,
    deps: PretrainDeps,
    machine_signature_arg_key: str,
    build_pretrain_run_spec_fn,
) -> None:
    args = normalize_pretrain_pipeline_args(pipeline)
    pipeline.run_spec = build_pretrain_run_spec_fn(args)
    _validate_runtime_spec(pipeline.run_spec, args)
    runtime = deps.setup_train_runtime_and_device(
        args,
        run_kind=str(pipeline.run_spec.run_kind),
    )
    pipeline.device = runtime.device
    pipeline.base_dtype = runtime.base_dtype
    pipeline.runtime_metadata = dict(runtime.runtime_metadata)
    deps.ensure_machine_adaptive_signature(
        args=args,
        device=runtime.device,
        arg_key=str(machine_signature_arg_key),
    )


def _validate_runtime_spec(run_spec, args) -> None:
    profile = run_spec.profile
    seq_len = int(run_spec.seq_len)
    max_seq_len = int(run_spec.max_seq_len)
    if seq_len <= 0:
        raise SophiaUsageError(
            f"[ERR] seq_len must be > 0 (got {args.seq_len!r})."
        )
    if max_seq_len != int(profile.curriculum.max_seq_len):
        raise SophiaUsageError(
            "[ERR] max_seq_len is pinned to the model context limit "
            f"{int(profile.curriculum.max_seq_len)} "
            f"(got {args.max_seq_len!r})."
        )
    if seq_len > max_seq_len:
        raise SophiaUsageError(
            f"[ERR] seq_len must be <= max_seq_len ({max_seq_len}) (got {seq_len})."
        )


def run_runtime_preflight_impl(
    pipeline: PretrainPipeline,
    *,
    deps: PretrainDeps | None = None,
    hooks=None,
) -> None:
    if pipeline.device is None:
        raise RuntimeError("pipeline device is not initialized")
    runtime = deps if deps is not None else hooks.runtime
    args = normalize_pretrain_pipeline_args(pipeline)
    torch.manual_seed(int(args.seed))
    cfg = runtime.runtime_preflight_config(batch_size=2, seq_len=128)
    dtype = torch.bfloat16 if pipeline.device.type == "cuda" else torch.float32
    model = None
    optimizer = None
    input_ids = None
    labels = None
    outputs = None
    loss = None
    try:
        model = runtime.decoder_model_cls(cfg).to(device=pipeline.device, dtype=dtype)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        last_loss = 0.0
        for _ in range(2):
            input_ids = torch.randint(
                0,
                int(cfg.vocab_size),
                (2, 128),
                device=pipeline.device,
                dtype=torch.long,
            )
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            from ml.modeling.decoder_output import require_output_loss

            loss = require_output_loss(outputs, context="pretrain runtime preflight")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().float().item())
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"[PREFLIGHT] device={pipeline.device} dtype={dtype} params={total_params:,} "
            f"seq_len=128 loss={last_loss:.4f}",
            flush=True,
        )
    finally:
        del loss
        del outputs
        del labels
        del input_ids
        del optimizer
        del model
        runtime.clear_cuda()


def prepare_model_session(
    *,
    session: PretrainSession,
    args: PretrainRunConfig,
    build_decoder_config_fn=build_decoder_config,
    load_or_init_model_fn=load_or_init_model,
    sync_runtime_batch_capacity_fn=sync_runtime_batch_capacity,
) -> None:
    decoder_config, vocab_size = build_decoder_config_fn(
        args=args,
        tokenizer=session.tokenizer,
    )
    session.vocab_size = int(vocab_size)
    session.model = load_or_init_model_fn(
        args=args,
        decoder_config=decoder_config,
        device=session.device,
        base_dtype=session.base_dtype,
    )
    sync_runtime_batch_capacity_fn(model=session.model, args=args)


def resume_model_session(
    *,
    session: PretrainSession,
    args: PretrainRunConfig,
    machine_signature_arg_key: str = "",
    maybe_resume_from_checkpoint_fn=maybe_resume_from_checkpoint,
    load_checkpoint_fn=load_checkpoint,
    create_optimizer_fn=create_optimizer,
    sync_runtime_batch_capacity_fn=sync_runtime_batch_capacity,
    hooks=None,
) -> PretrainRunConfig:
    if session.model is None:
        raise RuntimeError("session model is not initialized")
    if hooks is not None:
        maybe_resume_from_checkpoint_fn = getattr(
            hooks,
            "maybe_resume_from_checkpoint",
            maybe_resume_from_checkpoint_fn,
        )
        load_checkpoint_fn = getattr(hooks, "load_checkpoint", load_checkpoint_fn)
        create_optimizer_fn = getattr(hooks, "create_optimizer", create_optimizer_fn)
        sync_runtime_batch_capacity_fn = getattr(
            hooks,
            "sync_runtime_batch_capacity",
            sync_runtime_batch_capacity_fn,
        )
    session.resume = maybe_resume_from_checkpoint_fn(
        args=args,
        model=session.model,
        resume_path=session.resume_path,
        device=session.device,
        runtime_state=session.runtime_state,
        ckpt=session.bootstrap.resume_checkpoint,
        machine_signature_arg_key=str(machine_signature_arg_key),
        load_checkpoint_fn=load_checkpoint_fn,
        create_optimizer_fn=create_optimizer_fn,
        sync_runtime_batch_capacity_fn=sync_runtime_batch_capacity_fn,
    )
    resolved_args = session.resume.resolved_args
    if resolved_args is not None:
        args = resolved_args
    session.bind_runtime_state(
        PretrainRuntimeState.from_config_with_machine_runtime(
            args,
            source=session.runtime_state,
        )
    )
    total_tokens = int(args.total_tokens)
    if total_tokens <= 0:
        total_tokens = int(session.bootstrap.total_tokens)
    session.bootstrap = replace(session.bootstrap, total_tokens=int(total_tokens))
    return args


def apply_resume_seq_len(
    *,
    session: PretrainSession,
    args: PretrainRunConfig,
) -> None:
    del args
    if session.resume is None or session.resume.ckpt is None:
        return
    resolved_seq_len = int(session.runtime_state.seq_len)
    if resolved_seq_len <= 0:
        resolved_seq_len = int(session.seq_len)
    session.seq_len = int(resolved_seq_len)
    session.tokenizer.model_max_length = int(session.seq_len)


def prepare_data_and_model_impl(
    pipeline: PretrainPipeline,
    *,
    deps: PretrainDeps,
    refresh_curriculum_from_active_args_fn,
) -> None:
    args = normalize_pretrain_pipeline_args(pipeline)
    if pipeline.device is None:
        raise RuntimeError("pipeline device is not initialized")
    bootstrap = runtime_bootstrap.prepare_pretrain_runtime_bootstrap(
        args=args,
        prepare_output_dir_and_resume=deps.prepare_output_dir_and_resume,
        load_checkpoint=deps.load_checkpoint,
        prepare_pretrain_eval_data=deps.prepare_pretrain_eval_data,
        load_manifest_or_die=deps.load_manifest_or_die,
        load_tokenizer_and_validate_manifest=deps.load_tokenizer_and_validate_manifest,
    )
    args = bootstrap.project_run_config(args)
    loaded_recipe = load_json_object_from_path(
        path_label="machine_recipe",
        recipe_path=str(args.machine_recipe_json or ""),
    )
    args = run_pretrain_release_preflight(
        args=args,
        bootstrap=bootstrap,
        recipe_payload=(None if loaded_recipe is None else loaded_recipe.payload),
    )
    session = PretrainSession.from_bootstrap(
        spec=build_pretrain_run_spec(args),
        bootstrap=bootstrap,
        device=pipeline.device,
        base_dtype=pipeline.base_dtype,
        machine_signature=args._sophia_machine_signature,
        runtime_metadata=pipeline.runtime_metadata,
        runtime_state=PretrainRuntimeState.from_config(args),
    )
    args = attach_pretrain_session(pipeline, session, args=args)

    refresh_curriculum_from_active_args_fn(pipeline)
    args = normalize_pretrain_pipeline_args(pipeline)

    prepare_model_session(
        session=session,
        args=args,
        build_decoder_config_fn=deps.build_decoder_config,
        load_or_init_model_fn=deps.load_or_init_model,
        sync_runtime_batch_capacity_fn=deps.sync_runtime_batch_capacity,
    )
    args = resume_model_session(
        session=session,
        args=args,
        machine_signature_arg_key=str(
            getattr(pipeline, "_MACHINE_SIGNATURE_ARG_KEY", "")
        ),
        maybe_resume_from_checkpoint_fn=deps.maybe_resume_from_checkpoint,
        load_checkpoint_fn=deps.load_checkpoint,
        create_optimizer_fn=deps.create_optimizer,
        sync_runtime_batch_capacity_fn=deps.sync_runtime_batch_capacity,
    )
    session.absorb_run_config(args)
    if session.resume is not None and session.resume.ckpt is not None:
        args = attach_pretrain_session(pipeline, session, args=args)
        apply_resume_seq_len(session=session, args=args)
        refresh_curriculum_from_active_args_fn(pipeline)
    attach_pretrain_session(pipeline, session, args=args)


def refresh_curriculum_from_active_args(pipeline: PretrainPipeline) -> None:
    args = normalize_pretrain_pipeline_args(pipeline)
    session = ensure_pretrain_session(pipeline)
    profile = session.spec.profile
    resolved_max_seq_len = int(args.max_seq_len)
    if resolved_max_seq_len <= 0:
        resolved_max_seq_len = int(profile.curriculum.max_seq_len)
    session.runtime_plan.curriculum_stages = stage_plan.build_length_curriculum(
        total_tokens=int(args.total_tokens),
        max_seq_len=int(resolved_max_seq_len),
        train_seq_stages=list(profile.curriculum.train_seq_stages),
        stage_token_weights=list(profile.curriculum.stage_token_weights),
    )
    attach_pretrain_session(pipeline, session)


def run(
    pipeline: PretrainPipeline,
    *,
    deps: PretrainDeps,
    machine_signature_arg_key: str,
) -> None:
    setup_runtime(
        pipeline,
        deps=deps,
        machine_signature_arg_key=machine_signature_arg_key,
    )
    run_runtime_preflight(pipeline, deps=deps)
    prepare_data_and_model(pipeline, deps=deps)
    finalize_budget_and_runner(pipeline, deps=deps)
    args = normalize_pretrain_pipeline_args(pipeline)
    session = ensure_pretrain_session(pipeline)
    deps.write_machine_recipe_summary(
        args=args,
        output_dir=str(session.output_dir),
        train_manifest_path=str(session.manifest_path),
        max_steps=int(session.max_steps),
        seq_len=int(session.seq_len),
        base_dtype=session.base_dtype,
        device=session.device,
    )
    run_train_loop(pipeline, deps=deps)


def setup_runtime(
    pipeline: PretrainPipeline,
    *,
    deps: PretrainDeps,
    machine_signature_arg_key: str,
) -> None:
    setup_runtime_impl(
        pipeline,
        deps=deps,
        machine_signature_arg_key=machine_signature_arg_key,
        build_pretrain_run_spec_fn=build_pretrain_run_spec,
    )


def run_runtime_preflight(
    pipeline: PretrainPipeline,
    *,
    deps: PretrainDeps,
) -> None:
    run_runtime_preflight_impl(
        pipeline,
        deps=deps,
    )


def prepare_data_and_model(
    pipeline: PretrainPipeline,
    *,
    deps: PretrainDeps,
) -> None:
    prepare_data_and_model_impl(
        pipeline,
        deps=deps,
        refresh_curriculum_from_active_args_fn=refresh_curriculum_from_active_args,
    )


def finalize_budget_and_runner(
    pipeline: PretrainPipeline,
    *,
    deps: PretrainDeps,
) -> None:
    args = normalize_pretrain_pipeline_args(pipeline)
    session = ensure_pretrain_session(pipeline)
    if session.resume is None or session.model is None:
        raise RuntimeError("pipeline is missing resume/model state")
    if not session.runtime_plan.curriculum_stages:
        raise RuntimeError("pipeline curriculum is not initialized")
    budget = stage_plan.resolve_stage_token_budget(
        seq_len=int(session.seq_len),
        runtime_state=session.runtime_state,
    )
    stage_plan.build_stage_execution_plans(
        session=session,
        args=args,
        budget=budget,
    )
    args = stage_plan.apply_entry_stage_recipe(
        session=session,
        args=args,
    )
    args = deps.auto_observability_policy(
        args=args,
        output_dir=str(session.output_dir),
        seq_len=int(session.seq_len),
        max_steps=int(session.max_steps),
        runtime_state=session.runtime_state,
    )
    args = deps.auto_ema_policy(
        args=args,
        max_steps=int(session.max_steps),
        output_dir=str(session.output_dir),
        model=session.model,
        device=session.device,
        runtime_state=session.runtime_state,
    )
    session.absorb_run_config(args)
    session.runner = deps.build_runner(
        args=args,
        session=session,
    )
    deps.write_release_pretrain_profile(
        args=args,
        output_dir=str(session.output_dir),
        manifest_path=str(session.manifest_path),
        tokenizer_path=str(session.tokenizer_path),
        stage_execution_plans=list(session.runtime_plan.stage_execution_plans),
    )
    attach_pretrain_session(pipeline, session, args=args)


def run_train_loop(
    pipeline: PretrainPipeline,
    *,
    deps: PretrainDeps,
) -> None:
    session = ensure_pretrain_session(pipeline)
    if session.resume is None or session.model is None or session.tokenizer is None:
        raise RuntimeError("pipeline is missing resume/tokenizer state")
    if session.manifest is None:
        raise RuntimeError("pipeline manifest is not initialized")

    setup = build_pretrain_loop_setup(
        pipeline,
        deps=deps,
    )
    if int(setup.current_step) >= int(session.max_steps):
        return

    cursor = PretrainStageCursor(
        current_step=int(setup.current_step),
        train_state=setup.train_state,
    )
    cursor = run_resume_boundaries(
        pipeline=pipeline,
        deps=deps,
        stage_plans=setup.stage_plans,
        optimizer=setup.optimizer,
        scheduler=setup.scheduler,
        ema=setup.ema,
        cursor=cursor,
    )
    if int(cursor.current_step) >= int(session.max_steps):
        return
    run_planned_stages(
        pipeline=pipeline,
        deps=deps,
        stage_plans=setup.stage_plans,
        optimizer=setup.optimizer,
        scheduler=setup.scheduler,
        ema=setup.ema,
        cursor=cursor,
    )


__all__ = [
    "apply_resume_seq_len",
    "finalize_budget_and_runner",
    "prepare_data_and_model",
    "prepare_data_and_model_impl",
    "prepare_model_session",
    "refresh_curriculum_from_active_args",
    "resume_model_session",
    "run",
    "run_runtime_preflight",
    "run_runtime_preflight_impl",
    "setup_runtime",
    "setup_runtime_impl",
]
