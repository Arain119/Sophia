from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

import ml.training.pretrain.observability as observability
from ml.integrations.export.artifacts import export_model_artifacts
from ml.errors import SophiaUsageError
from ml.data.token_shards.shard_manifest import validate_tokenizer_fingerprint
from ml.data.token_shards.shard_manifest import load_manifest
from ml.training.pretrain.resources import (
    PretrainDataIter,
    PretrainManifestLike,
    PretrainTokenizerLike,
)
from ml.training.pretrain.loop_controls import PretrainDataControl, PretrainLoopControl
from ml.training.pretrain.machine_runtime import PretrainMachineRuntime
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.training.pretrain.engine.loop_execution import (
    LoopConfig,
    TrainLoopResult,
    train_loop,
)
from ml.training.pretrain.engine.data.pretrain import build_pretrain_data_iter
from ml.training.pretrain.engine.eval import evaluate_loss
from ml.training.pretrain.engine.step_runner_impl import StepRunner
from ml.training.pretrain import manifest_policy
from ml.training.pretrain.external_eval import build_pretrain_external_evals
from ml.training.pretrain.resume_restore import machine_runtime_payload
from ml.training.pretrain.train_state import PretrainTrainState


def build_pretrain_eval_fn(
    *,
    args: PretrainRunConfig,
    seq_len: int,
    model: torch.nn.Module,
    device: torch.device,
    base_dtype: torch.dtype,
    data_ctrl: PretrainDataControl | None = None,
    validate_tokenizer_fingerprint_fn=None,
    build_pretrain_data_iter_fn=None,
    evaluate_loss_fn=None,
) -> tuple[Callable[[], float] | None, int]:
    if validate_tokenizer_fingerprint_fn is None:
        validate_tokenizer_fingerprint_fn = validate_tokenizer_fingerprint
    if build_pretrain_data_iter_fn is None:
        build_pretrain_data_iter_fn = build_pretrain_data_iter
    if evaluate_loss_fn is None:
        evaluate_loss_fn = evaluate_loss
    if not isinstance(data_ctrl, PretrainDataControl):
        data_ctrl = PretrainDataControl.from_bound(
            args,
            PretrainRuntimeState.from_config(args),
        )
    eval_fn = None
    eval_interval = int(data_ctrl.eval_interval)
    eval_steps = int(data_ctrl.eval_steps)
    eval_manifest_path = manifest_policy.resolve_manifest_path(str(data_ctrl.eval_data_path))

    if eval_interval <= 0 or eval_steps <= 0:
        return None, int(eval_interval)
    if not eval_manifest_path:
        raise SophiaUsageError("[ERR] eval is enabled but eval split is missing.")
    if not os.path.exists(eval_manifest_path):
        raise SophiaUsageError(f"eval manifest.json not found: {eval_manifest_path}")

    eval_manifest = load_manifest(eval_manifest_path)
    try:
        validate_tokenizer_fingerprint_fn(
            tokenizer_dir=str(data_ctrl.tokenizer_path),
            expected_sha1=str(eval_manifest.tokenizer_sha1),
        )
    except (OSError, ValueError) as exc:
        raise SophiaUsageError(f"Tokenizer mismatch for eval_data_path: {exc}") from exc

    def _build_eval_iter() -> PretrainDataIter:
        return build_pretrain_data_iter_fn(
            manifest_path=str(eval_manifest_path),
            seq_len=int(seq_len),
            batch_size=int(data_ctrl.batch_size),
            seed=int(data_ctrl.seed),
            device=device,
            num_workers=0,
            dataloader_prefetch_factor=0,
            dataloader_persistent_workers=0,
            shard_preload=0,
            shard_preload_bytes=0,
        )

    def _eval_once() -> float:
        data_iter = _build_eval_iter()
        try:
            return float(
                evaluate_loss_fn(
                    model=model,
                    data_iter=data_iter,
                    base_dtype=base_dtype,
                    steps=int(eval_steps),
                )
            )
        finally:
            data_iter.close()

    eval_fn = _eval_once
    return eval_fn, int(eval_interval)


def build_pretrain_test_eval_fn(
    *,
    args: PretrainRunConfig,
    seq_len: int,
    model: torch.nn.Module,
    device: torch.device,
    base_dtype: torch.dtype,
    steps: int,
    data_ctrl: PretrainDataControl | None = None,
    validate_tokenizer_fingerprint_fn=None,
    build_pretrain_data_iter_fn=None,
    evaluate_loss_fn=None,
) -> Callable[[], float] | None:
    if validate_tokenizer_fingerprint_fn is None:
        validate_tokenizer_fingerprint_fn = validate_tokenizer_fingerprint
    if build_pretrain_data_iter_fn is None:
        build_pretrain_data_iter_fn = build_pretrain_data_iter
    if evaluate_loss_fn is None:
        evaluate_loss_fn = evaluate_loss
    if not isinstance(data_ctrl, PretrainDataControl):
        data_ctrl = PretrainDataControl.from_bound(
            args,
            PretrainRuntimeState.from_config(args),
        )
    if int(steps) <= 0:
        return None
    test_manifest_path = manifest_policy.resolve_manifest_path(str(data_ctrl.test_data_path))
    if not test_manifest_path:
        return None
    if not os.path.exists(test_manifest_path):
        raise SophiaUsageError(f"test manifest.json not found: {test_manifest_path}")

    test_manifest = load_manifest(test_manifest_path)
    try:
        validate_tokenizer_fingerprint_fn(
            tokenizer_dir=str(data_ctrl.tokenizer_path),
            expected_sha1=str(test_manifest.tokenizer_sha1),
        )
    except (OSError, ValueError) as exc:
        raise SophiaUsageError(f"Tokenizer mismatch for test_data_path: {exc}") from exc

    def _build_test_iter() -> PretrainDataIter:
        return build_pretrain_data_iter_fn(
            manifest_path=str(test_manifest_path),
            seq_len=int(seq_len),
            batch_size=int(data_ctrl.batch_size),
            seed=int(data_ctrl.seed),
            device=device,
            num_workers=0,
            dataloader_prefetch_factor=0,
            dataloader_persistent_workers=0,
            shard_preload=0,
            shard_preload_bytes=0,
        )

    def _eval_once() -> float:
        data_iter = _build_test_iter()
        try:
            return float(
                evaluate_loss_fn(
                    model=model,
                    data_iter=data_iter,
                    base_dtype=base_dtype,
                    steps=int(steps),
                )
            )
        finally:
            data_iter.close()

    return _eval_once


def _persistent_workers(cfg: PretrainMachineRuntime) -> int | None:
    return (
        None
        if int(cfg.dataloader_persistent_workers) < 0
        else int(cfg.dataloader_persistent_workers)
    )


def build_run_args_dict(
    *,
    args: PretrainRunConfig,
    tokens_per_update: int,
    runner: StepRunner | None = None,
) -> dict[str, object]:
    run_args = args.to_payload()
    run_args["_sophia_requested_target_tokens_per_update"] = run_args.get(
        "target_tokens_per_update"
    )
    run_args["target_tokens_per_update"] = int(tokens_per_update)
    machine_runtime = PretrainMachineRuntime.from_source(args)
    backend = (
        None
        if runner is None
        else getattr(getattr(runner, "execution_plan", None), "backend", None)
    )
    if backend is not None:
        machine_runtime = replace(machine_runtime, step_execution_backend=backend)
        run_args["step_execution_backend"] = str(backend)
        run_args["_sophia_runner_execution_backend"] = str(backend)
    runtime_payload = machine_runtime_payload(machine_runtime=machine_runtime)
    if runtime_payload:
        run_args["_sophia_machine_runtime"] = dict(runtime_payload)
    run_args["_sophia_effective_config_schema"] = "pretrain_effective_config_v1"
    return run_args


def build_train_data_iter(
    *,
    args: PretrainRunConfig,
    manifest_path: str,
    seq_len: int,
    device: torch.device,
    resume_state: dict[str, object] | None,
    data_ctrl: PretrainDataControl | None = None,
    machine_runtime: PretrainMachineRuntime | None = None,
    build_pretrain_data_iter_fn=None,
) -> PretrainDataIter:
    if build_pretrain_data_iter_fn is None:
        build_pretrain_data_iter_fn = build_pretrain_data_iter
    runtime_state = None
    if not isinstance(data_ctrl, PretrainDataControl):
        runtime_state = PretrainRuntimeState.from_config(args)
        data_ctrl = PretrainDataControl.from_bound(args, runtime_state)
    if not isinstance(machine_runtime, PretrainMachineRuntime):
        if runtime_state is None:
            runtime_state = PretrainRuntimeState.from_config(args)
        machine_runtime = PretrainMachineRuntime.from_source(runtime_state)
    num_workers = int(machine_runtime.dataloader_num_workers)
    return build_pretrain_data_iter_fn(
        manifest_path=str(manifest_path),
        seq_len=int(seq_len),
        batch_size=int(data_ctrl.batch_size),
        seed=int(data_ctrl.seed),
        device=device,
        num_workers=int(num_workers),
        dataloader_prefetch_factor=int(machine_runtime.dataloader_prefetch_factor),
        dataloader_persistent_workers=_persistent_workers(machine_runtime),
        shard_preload=int(machine_runtime.shard_preload),
        shard_preload_bytes=int(machine_runtime.shard_preload_bytes),
        resume_state=resume_state,
    )


@dataclass(frozen=True)
class TrainLoopBindings:
    eval_fn: Callable[[], float] | None
    extra_evals: list[tuple[str, int, Callable[[], float] | None]]
    cfg: LoopConfig
    args_dict: dict[str, object]


def _bind_eval_to_runner(
    eval_fn: Callable[[], float] | None,
    *,
    runner: StepRunner,
) -> Callable[[], float] | None:
    if eval_fn is None:
        return None

    def _eval_after_runner_release() -> float:
        runner.prepare_for_evaluation()
        return float(eval_fn())

    return _eval_after_runner_release


def build_train_loop_bindings(
    *,
    args: PretrainRunConfig,
    model: torch.nn.Module,
    runner: StepRunner,
    output_dir: str,
    manifest: PretrainManifestLike,
    device: torch.device,
    base_dtype: torch.dtype,
    seq_len: int,
    start_step: int,
    max_steps: int,
    data_ctrl: PretrainDataControl,
    loop_ctrl: PretrainLoopControl,
    tokenizer: PretrainTokenizerLike,
    build_pretrain_eval_fn_fn,
    build_pretrain_test_eval_fn_fn,
) -> TrainLoopBindings:
    eval_fn, eval_interval = build_pretrain_eval_fn_fn(
        args=args,
        seq_len=int(seq_len),
        model=model,
        device=device,
        base_dtype=base_dtype,
        data_ctrl=data_ctrl,
    )
    eval_fn = _bind_eval_to_runner(eval_fn, runner=runner)
    test_eval_fn = build_pretrain_test_eval_fn_fn(
        args=args,
        seq_len=int(seq_len),
        model=model,
        device=device,
        base_dtype=base_dtype,
        steps=int(data_ctrl.eval_steps),
        data_ctrl=data_ctrl,
    )
    test_eval_fn = _bind_eval_to_runner(test_eval_fn, runner=runner)
    extra_evals: list[tuple[str, int, Callable[[], float] | None]] = []
    if test_eval_fn is not None:
        extra_evals.append(("test_loss", int(max_steps), test_eval_fn))
    extra_evals.extend(
        (
            name,
            step,
            _bind_eval_to_runner(external_eval_fn, runner=runner),
        )
        for name, step, external_eval_fn in build_pretrain_external_evals(
            args=args,
            model=model,
            tokenizer=tokenizer,
            output_dir=str(output_dir),
            device=device,
            max_steps=int(max_steps),
        )
    )

    tokens_per_update = (
        int(seq_len)
        * int(data_ctrl.batch_size)
        * int(loop_ctrl.accumulation_steps)
    )
    _log_train_start(
        output_dir=str(output_dir),
        args=args,
        manifest=manifest,
        seq_len=int(seq_len),
        start_step=int(start_step),
        max_steps=int(max_steps),
        tokens_per_update=int(tokens_per_update),
        base_dtype=base_dtype,
    )
    return TrainLoopBindings(
        eval_fn=eval_fn,
        extra_evals=extra_evals,
        cfg=LoopConfig(
            output_dir=str(output_dir),
            max_steps=int(max_steps),
            accumulation_steps=int(loop_ctrl.accumulation_steps),
            log_interval=int(loop_ctrl.log_interval),
            save_interval=int(loop_ctrl.save_interval),
            save_total_limit=int(loop_ctrl.save_total_limit),
            tokens_per_update=int(tokens_per_update),
            eval_interval=int(eval_interval),
            async_checkpoint=int(loop_ctrl.async_checkpoint),
            async_metrics=int(loop_ctrl.async_metrics),
            ckpt_staging_dir=str(loop_ctrl.ckpt_staging_dir),
            save_best=int(loop_ctrl.save_best),
            enable_checkpoints=int(loop_ctrl.enable_checkpoints),
            token_weighted_loss=1,
        ),
        args_dict=build_run_args_dict(
            args=args,
            tokens_per_update=int(tokens_per_update),
            runner=runner,
        ),
    )


def _log_train_start(
    *,
    output_dir: str,
    args: PretrainRunConfig,
    manifest: PretrainManifestLike,
    seq_len: int,
    start_step: int,
    max_steps: int,
    tokens_per_update: int,
    base_dtype: torch.dtype,
) -> None:
    print(
        f"[INFO] training start | seq_len={int(seq_len)} | steps={int(start_step)}->{int(max_steps)} | "
        f"tokens/update={int(tokens_per_update)} | base_dtype={base_dtype} | "
        f"manifest_dtype={getattr(manifest, 'dtype', '?')}",
        flush=True,
    )
    observability.maybe_print_train_eta(
        args=args,
        output_dir=str(output_dir),
        max_steps=int(max_steps),
        start_step=int(start_step),
        seq_len=int(seq_len),
    )


def run_train_loop(
    *,
    args: PretrainRunConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    runner: StepRunner,
    tokenizer: PretrainTokenizerLike,
    output_dir: str,
    manifest: PretrainManifestLike,
    manifest_path: str,
    device: torch.device,
    base_dtype: torch.dtype,
    seq_len: int,
    start_step: int,
    max_steps: int,
    resume_train_state: PretrainTrainState | dict[str, object] | None,
    safe_serialization: bool,
    build_train_data_iter_fn=None,
    build_pretrain_eval_fn_fn=None,
    build_pretrain_test_eval_fn_fn=None,
    train_loop_fn=None,
) -> TrainLoopResult:
    if build_train_data_iter_fn is None:
        build_train_data_iter_fn = build_train_data_iter
    if build_pretrain_eval_fn_fn is None:
        build_pretrain_eval_fn_fn = build_pretrain_eval_fn
    if build_pretrain_test_eval_fn_fn is None:
        build_pretrain_test_eval_fn_fn = build_pretrain_test_eval_fn
    if train_loop_fn is None:
        train_loop_fn = train_loop
    runtime_state = PretrainRuntimeState.from_config(args)
    data_ctrl = PretrainDataControl.from_bound(args, runtime_state)
    machine_runtime = PretrainMachineRuntime.from_source(runtime_state)
    loop_ctrl = PretrainLoopControl.from_bound(args, runtime_state)
    data_iter = None
    try:
        resume_state = PretrainTrainState.resolve(resume_train_state)
        data_iter_state = resume_state.data_iter_state
        resume_data_state = (
            None if data_iter_state is None else data_iter_state.to_payload()
        )
        if (
            int(start_step) > 0
            and resume_data_state is None
        ):
            raise RuntimeError(
                "resuming training requires an exact data iterator state in train_state.data_iter_state"
            )
        data_iter = build_train_data_iter_fn(
            args=args,
            manifest_path=str(manifest_path),
            seq_len=int(seq_len),
            device=device,
            resume_state=resume_data_state,
            data_ctrl=data_ctrl,
            machine_runtime=machine_runtime,
        )
        bindings = build_train_loop_bindings(
            args=args,
            model=model,
            runner=runner,
            output_dir=str(output_dir),
            manifest=manifest,
            device=device,
            base_dtype=base_dtype,
            seq_len=int(seq_len),
            start_step=int(start_step),
            max_steps=int(max_steps),
            data_ctrl=data_ctrl,
            loop_ctrl=loop_ctrl,
            tokenizer=tokenizer,
            build_pretrain_eval_fn_fn=build_pretrain_eval_fn_fn,
            build_pretrain_test_eval_fn_fn=build_pretrain_test_eval_fn_fn,
        )
        model.train(True)
        result = train_loop_fn(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_iter=data_iter,
            runner=runner,
            device=device,
            cfg=bindings.cfg,
            start_step=int(start_step),
            args_dict=bindings.args_dict,
            tokenizer=tokenizer,
            safe_serialization=bool(safe_serialization),
            export_model_artifacts_fn=export_model_artifacts,
            eval_fn=bindings.eval_fn,
            extra_evals=bindings.extra_evals,
            resume_train_state=resume_state.to_payload(),
        )
        return TrainLoopResult(
            last_step=int(result.last_step),
            train_state=PretrainTrainState.from_payload(result.train_state).to_payload(),
        )
    finally:
        close_fn = getattr(data_iter, "close", None)
        if callable(close_fn):
            close_fn()


__all__ = [
    "LoopConfig",
    "TrainLoopBindings",
    "TrainLoopResult",
    "build_pretrain_eval_fn",
    "build_pretrain_data_iter",
    "build_pretrain_test_eval_fn",
    "build_run_args_dict",
    "build_train_data_iter",
    "evaluate_loss",
    "run_train_loop",
    "train_loop",
    "validate_tokenizer_fingerprint",
]
