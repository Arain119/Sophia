from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

import ml.tasks.pretrain.observability as observability_mod
import ml.tasks.pretrain.runtime as pretrain_runtime
import ml.tasks.pretrain.stage_runtime as stage_runtime_mod
from ml.core.engine.checkpointing import load_checkpoint
from ml.core.engine.machine_signature import ensure_machine_adaptive_signature
from ml.modeling import SophiaDecoder, SophiaDecoderConfig
from ml.core.spec import build_release_pretrain_schedule_spec
from ml.tasks.pretrain.deps import PretrainDeps
from ml.tasks.pretrain.pipeline_hooks import build_pretrain_runner, clear_cuda
from ml.training.pretrain.profiles import (
    RELEASE_PROFILE,
    VALIDATION_PROFILE,
    PretrainProfile,
    is_release_pretrain_profile,
)
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.tasks.pretrain.session import PretrainPipelineSnapshot, PretrainSession
from ml.training.ema import ModelEMA
from ml.training.pretrain import bootstrap_io, loop_runtime, resume_loader
from ml.training.pretrain.engine.grad_clip import (
    PINNED_AGC_CLIP,
    PINNED_AGC_EPS,
    PINNED_AGC_EXCLUDE_BIAS_AND_NORM,
    PINNED_GRAD_CLIP_MODE,
)
from ml.training.pretrain.engine.step_runner_impl import StepRunner
from ml.training.pretrain.release_config import (
    CANONICAL_PRETRAIN_MACHINE_RECIPE_PAYLOAD,
    CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD,
    RELEASE_PRETRAIN_DEFAULTS,
    canonical_pretrain_machine_recipe_path,
)
from ml.training.pretrain.model_setup import (
    build_decoder_config,
    load_or_init_model,
    sync_runtime_batch_capacity,
)
from ml.training.pretrain.output_policy import prepare_output_dir_and_resume
from ml.training.pretrain.runtime_setup import setup_train_runtime_and_device
from ml.training.pretrain.semantic_defaults import (
    PINNED_SEMANTIC_ADAM_EPS,
    PINNED_SEMANTIC_LEARNING_RATE,
    PINNED_SEMANTIC_TARGET_TOKENS_PER_UPDATE,
    PINNED_SEMANTIC_WEIGHT_DECAY,
)
from ml.training.pretrain.batch_size_autofit import create_optimizer
from ml.training.scheduler import build_lr_scheduler

_MACHINE_SIGNATURE_ARG_KEY = "_sophia_machine_signature"
_PRETRAIN_MACHINE_SIGNATURE_SCHEMA = "pretrain_machine_adaptive_v1"
BASE_SEQ_LEN = int(build_release_pretrain_schedule_spec().train_seq_len)
PINNED_MUON_TARGET_RMS = 0.18
PINNED_MUON_NS_STEPS = 4


def _default_target_tokens_per_update(*, profile: PretrainProfile) -> int:
    if is_release_pretrain_profile(profile):
        return int(RELEASE_PRETRAIN_DEFAULTS.target_tokens_per_update)
    return int(PINNED_SEMANTIC_TARGET_TOKENS_PER_UPDATE)


def _default_semantic_learning_rate(*, profile: PretrainProfile) -> float:
    if is_release_pretrain_profile(profile):
        return float(RELEASE_PRETRAIN_DEFAULTS.learning_rate)
    return float(PINNED_SEMANTIC_LEARNING_RATE)


def _default_adam_eps(*, profile: PretrainProfile) -> float:
    if is_release_pretrain_profile(profile):
        return float(RELEASE_PRETRAIN_DEFAULTS.adam_eps)
    return float(PINNED_SEMANTIC_ADAM_EPS)


def _default_embedding_lr_scale(*, profile: PretrainProfile) -> float:
    if is_release_pretrain_profile(profile):
        return float(RELEASE_PRETRAIN_DEFAULTS.embedding_lr_scale)
    return 1.0


def _default_ema_decay(*, profile: PretrainProfile) -> float:
    if is_release_pretrain_profile(profile):
        return float(RELEASE_PRETRAIN_DEFAULTS.ema_decay)
    return 0.999


def _default_machine_recipe_json(*, profile: PretrainProfile, value: str) -> str:
    normalized = str(value or "").strip()
    if normalized:
        return normalized
    if is_release_pretrain_profile(profile):
        return canonical_pretrain_machine_recipe_path()
    return ""


def _release_machine_int(field_name: str, *, default: int = 0) -> int:
    raw = CANONICAL_PRETRAIN_MACHINE_RECIPE_PAYLOAD.get(
        str(field_name), default
    )
    return int(raw)


def _release_runtime_int(field_name: str, *, default: int = -1) -> int:
    raw = CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD.get(str(field_name), default)
    return int(raw)


def runtime_preflight_config(*, batch_size: int, seq_len: int) -> SophiaDecoderConfig:
    validation_model = VALIDATION_PROFILE.model
    cfg = validation_model.with_overrides(
        max_seq_len=int(seq_len),
        max_batch_size=int(batch_size),
    ).to_config()
    return SophiaDecoderConfig.from_object(cfg)


def build_pretrain_args(
    *,
    data_path: str,
    tokenizer_path: str = "",
    output_dir: str = "",
    resume_from_checkpoint: str = "",
    overwrite_output_dir: int = 0,
    machine_recipe_json: str = "",
    decay_data_path: str = "",
    profile: PretrainProfile | None = None,
) -> PretrainRunConfig:
    active_profile = profile or RELEASE_PROFILE
    release_profile = is_release_pretrain_profile(active_profile)
    release_batch_size = _release_machine_int("batch_size") if release_profile else 0
    release_accumulation_steps = (
        _release_machine_int("accumulation_steps") if release_profile else 0
    )
    release_target_tokens_per_microbatch = (
        int(release_batch_size) * int(active_profile.curriculum.train_seq_len)
        if release_profile
        else 0
    )
    return PretrainRunConfig(
        data_path=str(data_path),
        tokenizer_path=str(tokenizer_path),
        output_dir=str(output_dir),
        resume_from_checkpoint=str(resume_from_checkpoint),
        overwrite_output_dir=int(overwrite_output_dir),
        machine_recipe_json=_default_machine_recipe_json(
            profile=active_profile,
            value=str(machine_recipe_json),
        ),
        _sophia_pretrain_profile=active_profile,
        _sophia_run_kind="pretrain",
        device="cuda:0",
        seed=42,
        safe_serialization=1,
        eval_data_path="",
        test_data_path="",
        decay_data_path=str(decay_data_path),
        eval_interval=1000,
        eval_steps=64,
        external_eval_contamination_samples=8,
        external_eval_contamination_interval=0,
        log_interval=100,
        save_interval=5000,
        save_total_limit=3,
        save_weights_steps=0,
        save_weights_total_limit=0,
        ckpt_staging_dir="",
        async_checkpoint=1,
        async_metrics=1,
        save_best=1,
        enable_checkpoints=1,
        stability_guard_enabled=1,
        stability_guard_loss_window=20,
        stability_guard_loss_max=12.5,
        stability_guard_grad_norm_max=20.0,
        stability_guard_grad_window=100,
        stability_guard_grad_spike_limit=30,
        stability_guard_min_step=100,
        total_tokens=int(active_profile.curriculum.default_total_tokens),
        target_tokens_per_update=int(
            _default_target_tokens_per_update(profile=active_profile)
        ),
        learning_rate=float(_default_semantic_learning_rate(profile=active_profile)),
        weight_decay=float(PINNED_SEMANTIC_WEIGHT_DECAY),
        beta1=0.9,
        beta2=0.95,
        adam_eps=float(_default_adam_eps(profile=active_profile)),
        max_steps=-1,
        min_lr_ratio=0.1,
        warmup_steps=800,
        warmup_ratio=0.0,
        lr_schedule="wsd",
        wsd_stable_ratio=0.9,
        wsd_decay_style="cosine",
        layerwise_lr_decay=1.0,
        embedding_lr_scale=float(_default_embedding_lr_scale(profile=active_profile)),
        muon_ns_steps=int(PINNED_MUON_NS_STEPS),
        muon_target_rms=float(PINNED_MUON_TARGET_RMS),
        max_grad_norm=1.0,
        grad_clip_mode=str(PINNED_GRAD_CLIP_MODE),
        agc_clip=float(PINNED_AGC_CLIP),
        agc_eps=float(PINNED_AGC_EPS),
        agc_exclude_bias_and_norm=int(bool(PINNED_AGC_EXCLUDE_BIAS_AND_NORM)),
        ema_decay=float(_default_ema_decay(profile=active_profile)),
        ema_update_interval=-1,
        ema_eval=1,
        ema_use_for_export=1,
        ema_in_ckpt=1,
        dataloader_num_workers=_release_runtime_int("dataloader_num_workers")
        if release_profile
        else -1,
        dataloader_prefetch_factor=_release_runtime_int("dataloader_prefetch_factor")
        if release_profile
        else -1,
        dataloader_persistent_workers=_release_runtime_int(
            "dataloader_persistent_workers"
        )
        if release_profile
        else -1,
        shard_preload=_release_runtime_int("shard_preload")
        if release_profile
        else -1,
        shard_preload_bytes=_release_runtime_int("shard_preload_bytes")
        if release_profile
        else -1,
        step_execution_backend=(
            str(CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD["step_execution_backend"])
            if release_profile
            else ""
        ),
        max_seq_len=int(active_profile.curriculum.max_seq_len),
        seq_len=int(active_profile.curriculum.train_seq_len),
        base_stage_seq_len=int(active_profile.curriculum.train_seq_len),
        target_tokens_per_microbatch=int(release_target_tokens_per_microbatch),
        batch_size=int(release_batch_size),
        accumulation_steps=int(release_accumulation_steps),
        gradient_checkpointing=(
            _release_machine_int("gradient_checkpointing") if release_profile else 0
        ),
        gradient_checkpointing_exclude_first=(
            _release_machine_int("gradient_checkpointing_exclude_first")
            if release_profile
            else 0
        ),
        gradient_checkpointing_exclude_last=(
            _release_machine_int("gradient_checkpointing_exclude_last")
            if release_profile
            else 0
        ),
        loss_chunk_size=_release_machine_int("loss_chunk_size")
        if release_profile
        else 0,
        auto_batch_size_max=4096,
    )


def _build_runner(
    *,
    args: PretrainRunConfig,
    session: PretrainSession | None = None,
    model: torch.nn.Module | None = None,
    base_dtype: torch.dtype | None = None,
) -> StepRunner:
    if session is None and (model is None or base_dtype is None):
        raise TypeError("_build_runner requires either session or model/base_dtype")
    runtime_state = (
        session.runtime_state
        if session is not None
        else PretrainRuntimeState.from_config(args)
    )
    return build_pretrain_runner(
        args=args,
        runtime_state=runtime_state,
        model=session.model if session is not None else model,
        base_dtype=session.base_dtype if session is not None else base_dtype,
    )


def _ensure_machine_signature(*, args, device, arg_key):
    return ensure_machine_adaptive_signature(
        args=args,
        device=device,
        arg_key=str(arg_key),
        schema=_PRETRAIN_MACHINE_SIGNATURE_SCHEMA,
    )


def _pretrain_deps() -> PretrainDeps:
    return PretrainDeps(
        setup_train_runtime_and_device=setup_train_runtime_and_device,
        ensure_machine_adaptive_signature=_ensure_machine_signature,
        runtime_preflight_config=runtime_preflight_config,
        decoder_model_cls=SophiaDecoder,
        clear_cuda=clear_cuda,
        prepare_output_dir_and_resume=prepare_output_dir_and_resume,
        load_checkpoint=load_checkpoint,
        prepare_pretrain_eval_data=bootstrap_io.prepare_pretrain_eval_data,
        load_manifest_or_die=bootstrap_io.load_manifest_or_die,
        load_tokenizer_and_validate_manifest=bootstrap_io.load_tokenizer_and_validate_manifest,
        build_decoder_config=build_decoder_config,
        load_or_init_model=load_or_init_model,
        sync_runtime_batch_capacity=sync_runtime_batch_capacity,
        maybe_resume_from_checkpoint=resume_loader.maybe_resume_from_checkpoint,
        auto_observability_policy=observability_mod.auto_observability_policy,
        auto_ema_policy=observability_mod.auto_ema_policy,
        write_release_pretrain_profile=observability_mod.write_release_pretrain_profile,
        write_machine_recipe_summary=observability_mod.write_machine_recipe_summary,
        build_runner=_build_runner,
        seed_active_stage_plan_recipe=stage_runtime_mod.seed_active_stage_plan_recipe,
        realize_stage_plan_machine_recipe=stage_runtime_mod.realize_stage_plan_machine_recipe,
        apply_stage_recipe=stage_runtime_mod.apply_stage_recipe,
        run_train_loop=loop_runtime.run_train_loop,
        create_optimizer=create_optimizer,
        build_lr_scheduler=build_lr_scheduler,
        model_ema_cls=ModelEMA,
    )


class PretrainPipeline:
    _MACHINE_SIGNATURE_ARG_KEY = _MACHINE_SIGNATURE_ARG_KEY

    def __init__(self, args: PretrainRunConfig) -> None:
        if not isinstance(args, PretrainRunConfig):
            raise TypeError(
                "PretrainPipeline requires PretrainRunConfig; build args via build_pretrain_args()"
            )
        self.args = args
        self.device: torch.device | None = None
        self.base_dtype: torch.dtype = torch.bfloat16
        self.runtime_metadata: dict[str, object] = {}
        self.snapshot = PretrainPipelineSnapshot(seq_len=int(BASE_SEQ_LEN))
        self.run_spec = None
        self.session = None

    def _invalidate_session(self) -> None:
        self.session = None

    def _update_snapshot(self, field_name: str, value: object) -> None:
        self._invalidate_session()
        self.snapshot = self.snapshot.updated(field_name, value)

    @property
    def output_dir(self) -> object:
        return self.snapshot.output_dir

    @output_dir.setter
    def output_dir(self, value: object) -> None:
        self._update_snapshot("output_dir", value)

    @property
    def resume_path(self) -> object:
        return self.snapshot.resume_path

    @resume_path.setter
    def resume_path(self, value: object) -> None:
        self._update_snapshot("resume_path", value)

    @property
    def manifest_path(self) -> object:
        return self.snapshot.manifest_path

    @manifest_path.setter
    def manifest_path(self, value: object) -> None:
        self._update_snapshot("manifest_path", value)

    @property
    def manifest(self) -> object:
        return self.snapshot.manifest

    @manifest.setter
    def manifest(self, value: object) -> None:
        self._update_snapshot("manifest", value)

    @property
    def tokenizer(self) -> object:
        return self.snapshot.tokenizer

    @tokenizer.setter
    def tokenizer(self, value: object) -> None:
        self._update_snapshot("tokenizer", value)

    @property
    def tokenizer_path(self) -> object:
        return self.snapshot.tokenizer_path

    @tokenizer_path.setter
    def tokenizer_path(self, value: object) -> None:
        self._update_snapshot("tokenizer_path", value)

    @property
    def seq_len(self) -> object:
        return self.snapshot.seq_len

    @seq_len.setter
    def seq_len(self, value: object) -> None:
        self._update_snapshot("seq_len", value)

    @property
    def vocab_size(self) -> object:
        return self.snapshot.vocab_size

    @vocab_size.setter
    def vocab_size(self, value: object) -> None:
        self._update_snapshot("vocab_size", value)

    @property
    def model(self) -> object:
        return self.snapshot.model

    @model.setter
    def model(self, value: object) -> None:
        self._update_snapshot("model", value)

    @property
    def resume(self) -> object:
        return self.snapshot.resume

    @resume.setter
    def resume(self, value: object) -> None:
        self._update_snapshot("resume", value)

    @property
    def max_steps(self) -> object:
        return self.snapshot.max_steps

    @max_steps.setter
    def max_steps(self, value: object) -> None:
        self._update_snapshot("max_steps", value)

    @property
    def runner(self) -> object:
        return self.snapshot.runner

    @runner.setter
    def runner(self, value: object) -> None:
        self._update_snapshot("runner", value)

    @property
    def curriculum_stages(self) -> object:
        return self.snapshot.curriculum_stages

    @curriculum_stages.setter
    def curriculum_stages(self, value: object) -> None:
        self._update_snapshot("curriculum_stages", value)

    @property
    def stage_execution_plans(self) -> object:
        return self.snapshot.stage_execution_plans

    @stage_execution_plans.setter
    def stage_execution_plans(self, value: object) -> None:
        self._update_snapshot("stage_execution_plans", value)

    @property
    def target_tokens_per_update(self) -> object:
        return self.snapshot.target_tokens_per_update

    @target_tokens_per_update.setter
    def target_tokens_per_update(self, value: object) -> None:
        self._update_snapshot("target_tokens_per_update", value)

    def run(self) -> None:
        pretrain_runtime.run(
            self,
            deps=_pretrain_deps(),
            machine_signature_arg_key=self._MACHINE_SIGNATURE_ARG_KEY,
        )

    def _setup_runtime(self) -> None:
        pretrain_runtime.setup_runtime(
            self,
            deps=_pretrain_deps(),
            machine_signature_arg_key=self._MACHINE_SIGNATURE_ARG_KEY,
        )

    def _run_runtime_preflight(self) -> None:
        pretrain_runtime.run_runtime_preflight(self, deps=_pretrain_deps())

    def _prepare_data_and_model(self) -> None:
        pretrain_runtime.prepare_data_and_model(self, deps=_pretrain_deps())

    def _finalize_budget_and_runner(self) -> None:
        pretrain_runtime.finalize_budget_and_runner(self, deps=_pretrain_deps())

    def _run_train_loop(self) -> None:
        pretrain_runtime.run_train_loop(self, deps=_pretrain_deps())


def run(args: PretrainRunConfig) -> None:
    PretrainPipeline(args=args).run()


__all__ = [
    "BASE_SEQ_LEN",
    "PretrainPipeline",
    "build_pretrain_args",
    "build_decoder_config",
    "build_lr_scheduler",
    "create_optimizer",
    "load_checkpoint",
    "load_or_init_model",
    "prepare_output_dir_and_resume",
    "run",
    "sync_runtime_batch_capacity",
    "_build_runner",
    "_pretrain_deps",
]
