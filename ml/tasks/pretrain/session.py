from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from ml.core.engine.context import RunContext
from ml.training.pretrain.resources import (
    PretrainManifestLike,
    PretrainTokenizerLike,
)
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.tasks.pretrain.run_spec_builder import (
    PretrainRunSpec,
    build_pretrain_run_spec,
)
from ml.tasks.pretrain.runtime_plan import PretrainRuntimePlan
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.training.pretrain.value_semantics import (
    coerce_int,
    coerce_str,
    normalized_str,
)

if TYPE_CHECKING:
    from ml.tasks.pretrain.pipeline import PretrainPipeline
    from ml.tasks.pretrain.planner import CurriculumStage, StagePlan
    from ml.training.pretrain.resume_loader import PretrainResumeState
    from ml.training.pretrain.engine.step_runner_impl import StepRunner
    from ml.training.pretrain.runtime_bootstrap import PretrainRuntimeBootstrap
else:
    PretrainResumeState = object
    StepRunner = object


def _coerce_optional_str(value: object) -> object:
    return None if value is None else str(value)


_SNAPSHOT_FIELD_NAMES = frozenset(
    {
        "output_dir",
        "resume_path",
        "manifest_path",
        "manifest",
        "tokenizer",
        "tokenizer_path",
        "seq_len",
        "vocab_size",
        "model",
        "resume",
        "max_steps",
        "runner",
        "curriculum_stages",
        "stage_execution_plans",
        "target_tokens_per_update",
    }
)
_SNAPSHOT_RUNTIME_STATE_OVERRIDE_FIELDS = (
    "seq_len",
    "max_steps",
    "target_tokens_per_update",
)


@dataclass(frozen=True)
class PretrainPipelineSnapshot:
    output_dir: str = ""
    resume_path: str | None = None
    manifest_path: str = ""
    manifest: PretrainManifestLike | None = None
    tokenizer: PretrainTokenizerLike | None = None
    tokenizer_path: str = ""
    seq_len: int = 0
    vocab_size: int = 0
    model: torch.nn.Module | None = None
    resume: PretrainResumeState | None = None
    max_steps: int = 0
    runner: StepRunner | None = None
    curriculum_stages: list[CurriculumStage] = field(default_factory=list)
    stage_execution_plans: list[StagePlan] = field(default_factory=list)
    target_tokens_per_update: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", coerce_str(self.output_dir))
        object.__setattr__(self, "resume_path", _coerce_optional_str(self.resume_path))
        object.__setattr__(self, "manifest_path", coerce_str(self.manifest_path))
        object.__setattr__(self, "tokenizer_path", coerce_str(self.tokenizer_path))
        object.__setattr__(self, "seq_len", coerce_int(self.seq_len))
        object.__setattr__(self, "vocab_size", coerce_int(self.vocab_size))
        object.__setattr__(self, "max_steps", coerce_int(self.max_steps))
        object.__setattr__(
            self,
            "curriculum_stages",
            [] if self.curriculum_stages is None else list(self.curriculum_stages),
        )
        object.__setattr__(
            self,
            "stage_execution_plans",
            []
            if self.stage_execution_plans is None
            else list(self.stage_execution_plans),
        )
        object.__setattr__(
            self,
            "target_tokens_per_update",
            coerce_int(self.target_tokens_per_update),
        )

    def updated(self, field_name: str, value: object) -> PretrainPipelineSnapshot:
        resolved_field_name = str(field_name)
        if resolved_field_name not in _SNAPSHOT_FIELD_NAMES:
            raise AttributeError(
                f"unknown pretrain pipeline snapshot field: {resolved_field_name!r}"
            )
        return replace(self, **{resolved_field_name: value})

    def runtime_state_overrides(self) -> dict[str, int]:
        overrides: dict[str, int] = {}
        for field_name in _SNAPSHOT_RUNTIME_STATE_OVERRIDE_FIELDS:
            value = int(getattr(self, field_name))
            if value > 0:
                overrides[field_name] = value
        return overrides

    @classmethod
    def from_session(cls, session: PretrainSession) -> PretrainPipelineSnapshot:
        return cls(
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
            curriculum_stages=session.runtime_plan.curriculum_stages,
            stage_execution_plans=session.runtime_plan.stage_execution_plans,
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
    runtime_plan: PretrainRuntimePlan = field(default_factory=PretrainRuntimePlan)

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
        runtime_plan: PretrainRuntimePlan | None = None,
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
            runtime_plan=PretrainRuntimePlan() if runtime_plan is None else runtime_plan,
        )

    def bind_runtime_state(self, runtime_state: PretrainRuntimeState) -> None:
        self.runtime_state = runtime_state
        self.runtime_plan.bind_runtime_state(runtime_state)

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
    runtime_plan: PretrainRuntimePlan


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
        runtime_plan=PretrainRuntimePlan.create(
            curriculum_stages=list(snapshot.curriculum_stages),
            stage_execution_plans=list(snapshot.stage_execution_plans),
        ),
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
    pipeline.snapshot = PretrainPipelineSnapshot.from_session(session)
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
        runtime_plan=seed.runtime_plan,
    )
    attach_pretrain_session(pipeline, session)
    return session


__all__ = [
    "PretrainRuntimeState",
    "PretrainRuntimePlan",
    "attach_pretrain_session",
    "ensure_pretrain_session",
    "normalize_pretrain_pipeline_args",
    "PretrainSession",
    "PretrainPipelineSnapshot",
]
