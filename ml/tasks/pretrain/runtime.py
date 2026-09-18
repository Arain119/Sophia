from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

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
from ml.tasks.pretrain.training_setup import build_pretrain_loop_setup
from ml.training.pretrain import runtime_bootstrap
from ml.training.pretrain.model_setup import (
    build_decoder_config,
    initialize_model,
    sync_runtime_batch_capacity,
)
from ml.training.pretrain.release_gate import run_pretrain_release_preflight
from ml.training.pretrain.resume_loader import maybe_resume_from_checkpoint
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.training.runtime_tools import create_optimizer

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
    if max_seq_len != int(profile.training.max_seq_len):
        raise SophiaUsageError(
            "[ERR] max_seq_len is pinned to the model context limit "
            f"{int(profile.training.max_seq_len)} "
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
    probe_batch_size = 1
    probe_seq_len = 2
    cfg = runtime.runtime_preflight_config(
        batch_size=probe_batch_size,
        seq_len=probe_seq_len,
    )
    dtype = torch.bfloat16 if pipeline.device.type == "cuda" else torch.float32
    model = None
    input_ids = None
    labels = None
    outputs = None
    loss = None
    try:
        model = runtime.decoder_model_cls(cfg).to(device=pipeline.device, dtype=dtype)
        model.train()
        input_ids = torch.randint(
            0,
            int(cfg.vocab_size),
            (probe_batch_size, probe_seq_len),
            device=pipeline.device,
            dtype=torch.long,
        )
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
        from ml.modeling.decoder_output import require_output_loss

        loss = require_output_loss(outputs, context="pretrain runtime preflight")
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("pretrain runtime preflight produced a non-finite loss")
        loss.backward()
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"[PREFLIGHT] device={pipeline.device} dtype={dtype} params={total_params:,} "
            f"batch_size={probe_batch_size} seq_len={probe_seq_len} "
            f"loss={float(loss.detach().float().item()):.4f}",
            flush=True,
        )
    finally:
        del loss
        del outputs
        del labels
        del input_ids
        del model
        runtime.clear_cuda()


def prepare_model_session(
    *,
    session: PretrainSession,
    args: PretrainRunConfig,
    build_decoder_config_fn=build_decoder_config,
    initialize_model_fn=initialize_model,
    sync_runtime_batch_capacity_fn=sync_runtime_batch_capacity,
) -> None:
    decoder_config, vocab_size = build_decoder_config_fn(
        args=args,
        tokenizer=session.tokenizer,
    )
    session.vocab_size = int(vocab_size)
    session.model = initialize_model_fn(
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


def validate_fixed_context(
    *,
    session: PretrainSession,
    args: PretrainRunConfig,
) -> None:
    profile = session.spec.profile
    expected_max_seq_len = int(profile.training.max_seq_len)
    expected_seq_len = int(profile.training.train_seq_len)
    if int(args.max_seq_len) != expected_max_seq_len:
        raise SophiaUsageError(
            "[ERR] pretrain max_seq_len does not match the fixed profile: "
            f"observed={int(args.max_seq_len)} expected={expected_max_seq_len}"
        )
    if int(session.runtime_state.seq_len) != expected_seq_len:
        raise SophiaUsageError(
            "[ERR] pretrain seq_len does not match the fixed profile: "
            f"observed={int(session.runtime_state.seq_len)} expected={expected_seq_len}"
        )
    session.seq_len = expected_seq_len
    session.tokenizer.model_max_length = expected_seq_len


def prepare_data_and_model_impl(
    pipeline: PretrainPipeline,
    *,
    deps: PretrainDeps,
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
    validate_fixed_context(session=session, args=args)

    prepare_model_session(
        session=session,
        args=args,
        build_decoder_config_fn=deps.build_decoder_config,
        initialize_model_fn=deps.initialize_model,
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
    validate_fixed_context(session=session, args=args)
    attach_pretrain_session(pipeline, session, args=args)


def resolve_max_steps(*, total_tokens: int, tokens_per_update: int, limit: int) -> int:
    if int(total_tokens) <= 0:
        raise SophiaUsageError("[ERR] total_tokens must be positive.")
    if int(tokens_per_update) <= 0:
        raise SophiaUsageError("[ERR] tokens_per_update must be positive.")
    required_steps = int(math.ceil(int(total_tokens) / int(tokens_per_update)))
    return min(required_steps, int(limit)) if int(limit) > 0 else required_steps


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
    validate_fixed_context(session=session, args=args)
    batch_size = int(session.runtime_state.batch_size)
    accumulation_steps = int(session.runtime_state.accumulation_steps)
    if batch_size <= 0 or accumulation_steps <= 0:
        raise SophiaUsageError(
            "[ERR] batch_size and accumulation_steps must be positive before training."
        )
    tokens_per_update = int(session.seq_len) * batch_size * accumulation_steps
    expected_tokens_per_update = int(session.runtime_state.target_tokens_per_update)
    if tokens_per_update != expected_tokens_per_update:
        raise SophiaUsageError(
            "[ERR] machine recipe changes the fixed tokens/update contract: "
            f"observed={tokens_per_update} expected={expected_tokens_per_update}"
        )
    max_steps = resolve_max_steps(
        total_tokens=int(args.total_tokens),
        tokens_per_update=tokens_per_update,
        limit=int(args.max_steps),
    )
    args = replace(args, max_steps=max_steps)
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
    )
    attach_pretrain_session(pipeline, session, args=args)


def run_train_loop(
    pipeline: PretrainPipeline,
    *,
    deps: PretrainDeps,
) -> None:
    session = ensure_pretrain_session(pipeline)
    if session.resume is None or session.model is None or session.runner is None:
        raise RuntimeError("pipeline is missing resume/model/runner state")
    if session.manifest is None:
        raise RuntimeError("pipeline manifest is not initialized")

    setup = build_pretrain_loop_setup(
        pipeline,
        deps=deps,
    )
    if int(setup.current_step) >= int(session.max_steps):
        return

    deps.run_train_loop(
        args=pipeline.args,
        model=session.model,
        optimizer=setup.optimizer,
        scheduler=setup.scheduler,
        runner=session.runner,
        tokenizer=session.tokenizer,
        output_dir=str(session.output_dir),
        manifest=session.manifest,
        manifest_path=str(session.manifest_path),
        device=session.device,
        base_dtype=session.base_dtype,
        seq_len=int(session.seq_len),
        start_step=int(setup.current_step),
        max_steps=int(session.max_steps),
        resume_train_state=setup.train_state,
        safe_serialization=bool(int(pipeline.args.safe_serialization)),
    )


__all__ = [
    "finalize_budget_and_runner",
    "prepare_data_and_model",
    "prepare_data_and_model_impl",
    "prepare_model_session",
    "resolve_max_steps",
    "resume_model_session",
    "run",
    "run_runtime_preflight",
    "run_runtime_preflight_impl",
    "setup_runtime",
    "setup_runtime_impl",
    "validate_fixed_context",
]
