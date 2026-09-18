from __future__ import annotations

from dataclasses import dataclass

from ml.core.engine.spec import RunSpec
from ml.training.pretrain.profiles import (
    PretrainProfile,
    is_release_pretrain_profile,
    resolve_pretrain_profile,
)
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.value_semantics import (
    coerce_int,
    enabled_flag,
    normalized_str,
    positive_int_or_default,
)
from ml.training.pretrain.engine.step_execution import StepExecutionPolicy


@dataclass(frozen=True)
class PretrainRunSpec:
    run: RunSpec
    profile: PretrainProfile
    data_path: str
    tokenizer_path: str
    eval_data_path: str
    test_data_path: str
    requested_device: str
    overwrite_output_dir: bool
    total_tokens: int
    target_tokens_per_update: int
    max_seq_len: int
    seq_len: int
    learning_rate: float
    weight_decay: float
    optimizer_beta1: float
    optimizer_beta2: float
    optimizer_eps: float
    step_execution: StepExecutionPolicy

    @property
    def run_kind(self) -> str:
        return str(self.run.run_kind)


RELEASE_STEP_EXECUTION_POLICY = StepExecutionPolicy(
    backend="sm120_graph",
)
VALIDATION_STEP_EXECUTION_POLICY = StepExecutionPolicy(
    backend="eager",
)


def _step_execution_policy_for(
    *,
    cfg: PretrainRunConfig,
    profile: PretrainProfile,
) -> StepExecutionPolicy:
    backend = normalized_str(getattr(cfg, "step_execution_backend", ""))
    if backend:
        return StepExecutionPolicy(backend=str(backend))
    return (
        RELEASE_STEP_EXECUTION_POLICY
        if is_release_pretrain_profile(profile)
        else VALIDATION_STEP_EXECUTION_POLICY
    )


def build_pretrain_run_spec(args: PretrainRunConfig) -> PretrainRunSpec:
    cfg = args
    profile = resolve_pretrain_profile(cfg)
    max_steps = int(cfg.max_steps)
    resume_from_checkpoint = normalized_str(cfg.resume_from_checkpoint)
    step_execution = _step_execution_policy_for(cfg=cfg, profile=profile)
    return PretrainRunSpec(
        run=RunSpec(
            run_kind=normalized_str(cfg._sophia_run_kind, default="pretrain"),
            model=profile.model,
            seed=int(cfg.seed),
            max_steps=(None if max_steps <= 0 else int(max_steps)),
            output_dir=str(cfg.output_dir),
            resume_from_checkpoint=(
                None if resume_from_checkpoint == "" else resume_from_checkpoint
            ),
            safe_serialization=enabled_flag(cfg.safe_serialization),
        ),
        profile=profile,
        data_path=str(cfg.data_path),
        tokenizer_path=str(cfg.tokenizer_path),
        eval_data_path=str(cfg.eval_data_path),
        test_data_path=str(cfg.test_data_path),
        requested_device=normalized_str(cfg.device, default="cuda:0"),
        overwrite_output_dir=enabled_flag(cfg.overwrite_output_dir),
        total_tokens=max(coerce_int(cfg.total_tokens), 0),
        target_tokens_per_update=max(coerce_int(cfg.target_tokens_per_update), 0),
        max_seq_len=max(
            positive_int_or_default(
                cfg.max_seq_len,
                default=int(profile.training.max_seq_len),
            ),
            0,
        ),
        seq_len=max(
            positive_int_or_default(
                cfg.seq_len,
                default=int(profile.training.train_seq_len),
            ),
            0,
        ),
        learning_rate=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        optimizer_beta1=float(cfg.beta1),
        optimizer_beta2=float(cfg.beta2),
        optimizer_eps=float(cfg.adam_eps),
        step_execution=step_execution,
    )


__all__ = [
    "PretrainRunSpec",
    "build_pretrain_run_spec",
]
