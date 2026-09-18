from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from ml.core.engine.context import RunContext
from ml.training.pretrain.resources import (
    PretrainManifestLike,
    PretrainTokenizerLike,
)
from ml.training.pretrain.pipeline_snapshot import PretrainPipelineSnapshot
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.tasks.pretrain.run_spec_builder import (
    PretrainRunSpec,
    build_pretrain_run_spec,
)
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.training.pretrain.value_semantics import (
    normalized_str,
)

if TYPE_CHECKING:
    from ml.tasks.pretrain.pipeline import PretrainPipeline
    from ml.training.pretrain.resume_loader import PretrainResumeState
    from ml.training.pretrain.engine.step_runner_impl import StepRunner
    from ml.training.pretrain.runtime_bootstrap import PretrainRuntimeBootstrap
else:
    PretrainResumeState = object
    StepRunner = object


def _snapshot_from_session(session: PretrainSession) -> PretrainPipelineSnapshot:
    return PretrainPipelineSnapshot(
        output_dir=session.output_dir,
        resume_path=session.resume_path,
        manifest_path=session.manifest_path,
        manifest=session.manifest,
        tokenizer=session.tokenizer,
        tokenizer_path=session.tokenizer_path,
        seq_len=int(session.seq_len),
        vocab_size=int(session.vocab_size),
        model=session.model,
        resume=session.resume,
        max_steps=int(session.max_steps),
        runner=session.runner,
        target_tokens_per_update=int(session.target_tokens_per_update),
    )


@dataclass
class PretrainSession:
    spec: PretrainRunSpec
    run: RunContext
    bootstrap: PretrainRuntimeBootstrap
    resume: PretrainResumeState | None = None
    model: torch.nn.Module | None = None
    runner: StepRunner | None = None
    vocab_size: int = 0
    runtime_state: PretrainRuntimeState = field(default_factory=PretrainRuntimeState)

    def __post_init__(self) -> None:
        self.vocab_size = int(self.vocab_size)
        self.bind_runtime_state(self.runtime_state)
        if int(self.runtime_state.seq_len) <= 0:
            self.runtime_state.seq_len = int(self.bootstrap.seq_len)

    @classmethod
    def from_bootstrap(
        cls,
        *,
        spec: PretrainRunSpec,
        bootstrap: PretrainRuntimeBootstrap,
        device: torch.device,
        base_dtype: torch.dtype,
        machine_signature: dict[str, object] | None,
        runtime_metadata: dict[str, object] | None = None,
        resume: PretrainResumeState | None = None,
        model: torch.nn.Module | None = None,
        runner: StepRunner | None = None,
        vocab_size: int = 0,
        runtime_state: PretrainRuntimeState | None = None,
    ) -> PretrainSession:
        return cls(
            spec=spec,
            run=RunContext(
                output_dir=str(bootstrap.output_dir),
                device=device,
                base_dtype=base_dtype,
                tokenizer=bootstrap.tokenizer,
                runtime_metadata=dict(runtime_metadata or {}),
                machine_signature=dict(machine_signature or {}),
                resume_path=bootstrap.resume_path,
                resolved_device=str(device),
            ),
            bootstrap=bootstrap,
            resume=resume,
            model=model,
            runner=runner,
            vocab_size=int(vocab_size),
            runtime_state=(
                PretrainRuntimeState() if runtime_state is None else runtime_state
            ),
        )

    def bind_runtime_state(self, runtime_state: PretrainRuntimeState) -> None:
        self.runtime_state = runtime_state

    @property
    def output_dir(self) -> str:
        return str(self.run.output_dir)

    @property
    def resume_path(self) -> str | None:
        return self.run.resume_path

    @property
    def device(self) -> torch.device:
        return self.run.device

    @property
    def base_dtype(self) -> torch.dtype:
        return self.run.base_dtype

    @property
    def seq_len(self) -> int:
        resolved = int(self.runtime_state.seq_len)
        if resolved > 0:
            return int(resolved)
        return int(self.bootstrap.seq_len)

    @seq_len.setter
    def seq_len(self, value: int) -> None:
        resolved = int(value)
        if resolved <= 0:
            resolved = int(self.bootstrap.seq_len)
        self.runtime_state.seq_len = int(resolved)

    @property
    def max_steps(self) -> int:
        return int(self.runtime_state.max_steps)

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        self.runtime_state.max_steps = int(value)

    @property
    def target_tokens_per_update(self) -> int:
        return int(self.runtime_state.target_tokens_per_update)

    @target_tokens_per_update.setter
    def target_tokens_per_update(self, value: int) -> None:
        self.runtime_state.target_tokens_per_update = int(value)

    @property
    def tokenizer_path(self) -> str:
        return str(self.bootstrap.tokenizer_path)

    @property
    def tokenizer(self) -> PretrainTokenizerLike:
        return self.bootstrap.tokenizer

    @property
    def manifest(self) -> PretrainManifestLike:
        return self.bootstrap.manifest

    @property
    def manifest_path(self) -> str:
        return str(self.bootstrap.manifest_path)

    def project_run_config(self, cfg: PretrainRunConfig) -> PretrainRunConfig:
        projected = self.bootstrap.project_run_config(cfg)
        return self.runtime_state.project_run_config(projected)

    def absorb_run_config(self, cfg: PretrainRunConfig) -> None:
        next_state = PretrainRuntimeState.from_config_with_machine_runtime(
            cfg,
            source=self.runtime_state,
        )
        self.bind_runtime_state(next_state)
        self.spec = build_pretrain_run_spec(cfg)


@dataclass(frozen=True)
class _PretrainSessionSeed:
    cfg: PretrainRunConfig
    run_spec: PretrainRunSpec
    bootstrap: PretrainRuntimeBootstrap
    device: torch.device
    base_dtype: torch.dtype
    runtime_metadata: dict[str, object]
    snapshot: PretrainPipelineSnapshot
    runtime_state: PretrainRuntimeState


def normalize_pretrain_pipeline_args(
    pipeline: PretrainPipeline,
) -> PretrainRunConfig:
    if not isinstance(pipeline.args, PretrainRunConfig):
        raise TypeError(
            "pretrain pipeline requires PretrainRunConfig; build args via build_pretrain_args()"
        )
    return pipeline.args


def _build_pretrain_session_seed(pipeline: PretrainPipeline) -> _PretrainSessionSeed:
    from ml.training.pretrain.runtime_bootstrap import PretrainRuntimeBootstrap

    cfg = normalize_pretrain_pipeline_args(pipeline)
    if pipeline.device is None:
        raise RuntimeError("pipeline device is not initialized")
    snapshot = pipeline.snapshot
    output_dir = normalized_str(snapshot.output_dir)
    if not output_dir:
        raise RuntimeError("pipeline output_dir is not initialized")
    runtime_state = PretrainRuntimeState.from_config(cfg)
    overrides = snapshot.runtime_state_overrides()
    runtime_state.apply_mapping(overrides, field_names=tuple(overrides.keys()))
    return _PretrainSessionSeed(
        cfg=cfg,
        run_spec=(
            build_pretrain_run_spec(cfg)
            if pipeline.run_spec is None
            else pipeline.run_spec
        ),
        bootstrap=PretrainRuntimeBootstrap.from_snapshot(
            snapshot=snapshot,
            cfg=cfg,
            resume_checkpoint=(None if snapshot.resume is None else snapshot.resume.ckpt),
        ),
        device=pipeline.device,
        base_dtype=pipeline.base_dtype,
        runtime_metadata=dict(pipeline.runtime_metadata),
        snapshot=snapshot,
        runtime_state=runtime_state,
    )


def attach_pretrain_session(
    pipeline: PretrainPipeline,
    session: PretrainSession,
    *,
    args: PretrainRunConfig | None = None,
) -> PretrainRunConfig:
    cfg = normalize_pretrain_pipeline_args(pipeline) if args is None else args
    pipeline.session = session
    pipeline.args = session.project_run_config(cfg)
    pipeline.run_spec = session.spec
    pipeline.device = session.device
    pipeline.base_dtype = session.base_dtype
    pipeline.runtime_metadata = dict(session.run.runtime_metadata)
    pipeline.snapshot = _snapshot_from_session(session)
    return pipeline.args


def ensure_pretrain_session(pipeline: PretrainPipeline) -> PretrainSession:
    session = pipeline.session
    if isinstance(session, PretrainSession):
        return session

    seed = _build_pretrain_session_seed(pipeline)
    session = PretrainSession.from_bootstrap(
        spec=seed.run_spec,
        bootstrap=seed.bootstrap,
        device=seed.device,
        base_dtype=seed.base_dtype,
        machine_signature=seed.cfg._sophia_machine_signature,
        runtime_metadata=seed.runtime_metadata,
        resume=seed.snapshot.resume,
        model=seed.snapshot.model,
        runner=seed.snapshot.runner,
        vocab_size=int(seed.snapshot.vocab_size),
        runtime_state=seed.runtime_state,
    )
    attach_pretrain_session(pipeline, session)
    return session


__all__ = [
    "PretrainRuntimeState",
    "attach_pretrain_session",
    "ensure_pretrain_session",
    "normalize_pretrain_pipeline_args",
    "PretrainSession",
]
