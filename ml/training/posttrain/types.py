from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ml.core.engine.checkpointing import EngineCheckpoint
from ml.core.engine.context import RunContext
from ml.core.engine.spec import RunSpec
from ml.core.engine.types import MetricsRow, StatePayload
from ml.core.spec import ModelSpec
from ml.training.ema import ModelEMA
from ml.training.posttrain.contracts import SupportsPosttrainTokenizer

if TYPE_CHECKING:
    from collections.abc import Callable
    from ml.tasks.sft.spec import SftStageConfig


@dataclass(frozen=True)
class ResumeState:
    checkpoint: EngineCheckpoint | None
    start_step: int
    train_state: StatePayload


@dataclass(frozen=True)
class PosttrainReportSpec:
    split_paths: dict[str, str]
    report_name: str


@dataclass(frozen=True)
class PosttrainLaunchConfig:
    output_dir: str
    resume_from_checkpoint: str
    overwrite_output_dir: bool
    export_dir: str
    requested_device: str
    seed: int
    gradient_checkpointing: bool
    max_steps: int


@dataclass(frozen=True)
class PosttrainOptimizerConfig:
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    adam_eps: float
    warmup_steps: int
    warmup_ratio: float
    min_lr_ratio: float
    lr_schedule: str
    wsd_stable_ratio: float
    wsd_decay_style: str
    layerwise_lr_decay: float
    muon_ns_steps: int | None
    muon_target_rms: float | None


@dataclass(frozen=True)
class PosttrainEmaConfig:
    decay: float
    update_interval: int


@dataclass(frozen=True)
class PosttrainCheckpointPolicy:
    save_total_limit: int
    staging_dir: str | None


@dataclass(frozen=True)
class PosttrainMachineSelection:
    batch_size: int
    accumulation_steps: int
    gradient_checkpointing: bool

    def to_payload(self) -> dict[str, int]:
        return {
            "batch_size": int(self.batch_size),
            "accumulation_steps": int(self.accumulation_steps),
            "gradient_checkpointing": int(bool(self.gradient_checkpointing)),
        }


@dataclass(frozen=True)
class PosttrainExportSpec:
    output_dir: str
    safe_serialization: bool


@dataclass(frozen=True)
class PosttrainRunSpec:
    run: RunSpec
    resolved_max_seq_len: int
    launch: PosttrainLaunchConfig
    optimizer: PosttrainOptimizerConfig
    ema: PosttrainEmaConfig
    checkpoint_policy: PosttrainCheckpointPolicy
    safe_serialization: bool

    @property
    def run_kind(self) -> str:
        return str(self.run.run_kind)


@dataclass(frozen=True)
class PosttrainCheckpointSpec:
    output_dir: str
    step: int
    args_payload: StatePayload
    train_state: StatePayload
    save_total_limit: int
    staging_dir: str | None


@dataclass(frozen=True)
class PosttrainRunContext:
    run: RunContext
    args: PosttrainStageArgs
    tokenizer: SupportsPosttrainTokenizer | None
    model: torch.nn.Module
    resume: ResumeState
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    ema: ModelEMA | None
    start_step: int
    run_args_payload: StatePayload
    checkpoint_policy: PosttrainCheckpointPolicy
    export: PosttrainExportSpec

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


@dataclass(frozen=True)
class PosttrainStepResult:
    train_time_s: float
    metrics_row: MetricsRow | None = None
    extra_metrics_rows: tuple[MetricsRow, ...] = ()
    print_line: str | None = None
    checkpoint_train_state: StatePayload | None = None
    checkpoint_report_callback: Callable[[], None] | None = None
    reset_interval: bool = False


@dataclass
class PosttrainStageArgs:
    export_dir: str = ""
    train_data: str = ""
    eval_data: str = ""
    test_data: str = ""
    output_dir: str = ""
    resume_from_checkpoint: str = ""
    overwrite_output_dir: int = 0
    machine_recipe_json: str = ""
    allow_missing_machine_recipe: int = 0
    device: str = "cuda:0"
    seed: int = 42
    max_seq_len: int = 0
    curriculum_path: str = ""
    curriculum_stage: str = ""
    batch_size: int = 1
    accumulation_steps: int = 1
    max_steps: int = 0
    learning_rate: float = 5e-6
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.0
    lr_schedule: str = "cosine"
    wsd_stable_ratio: float = 0.0
    wsd_decay_style: str = "cosine"
    layerwise_lr_decay: float = 1.0
    muon_ns_steps: int = 0
    muon_target_rms: float | None = None
    max_grad_norm: float = 0.0
    gradient_checkpointing: int = 1
    pad_to_multiple_of: int = 128
    eval_interval: int = 0
    eval_steps: int = 0
    save_interval: int = 0
    save_total_limit: int = 2
    safe_serialization: int = 1
    report_max_examples_per_split: int = 0
    report_progress_every: int = 0
    ema_decay: float = 0.0
    ema_update_interval: int = 1
    ckpt_staging_dir: str = ""

    @property
    def safe_serialization_enabled(self) -> bool:
        return bool(int(self.safe_serialization) == 1)

    @property
    def gradient_checkpointing_enabled(self) -> bool:
        return bool(int(self.gradient_checkpointing) == 1)

    @property
    def accumulation_steps_value(self) -> int:
        return int(self.accumulation_steps)

    @property
    def effective_accumulation_steps(self) -> int:
        return max(self.accumulation_steps_value, 1)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> PosttrainStageArgs:
        def _string(name: str, default: str = "") -> str:
            raw = values.get(name, default)
            if raw is None:
                return str(default)
            return str(raw)

        def _int(name: str, default: int = 0) -> int:
            raw = values.get(name, default)
            if raw is None:
                return int(default)
            return int(raw)

        def _float(name: str, default: float = 0.0) -> float:
            raw = values.get(name, default)
            if raw is None:
                return float(default)
            return float(raw)

        return cls(
            export_dir=_string("export_dir"),
            train_data=_string("train_data"),
            eval_data=_string("eval_data"),
            test_data=_string("test_data"),
            output_dir=_string("output_dir"),
            resume_from_checkpoint=_string("resume_from_checkpoint"),
            overwrite_output_dir=_int("overwrite_output_dir", 0),
            machine_recipe_json=_string("machine_recipe_json"),
            allow_missing_machine_recipe=_int("allow_missing_machine_recipe", 0),
            device=_string("device", "cuda:0"),
            seed=_int("seed", 42),
            max_seq_len=_int("max_seq_len", 0),
            curriculum_path=_string("curriculum_path"),
            curriculum_stage=_string("curriculum_stage"),
            batch_size=_int("batch_size", 1),
            accumulation_steps=_int("accumulation_steps", 1),
            max_steps=_int("max_steps", 0),
            learning_rate=_float("learning_rate", 5e-6),
            weight_decay=_float("weight_decay", 0.1),
            beta1=_float("beta1", 0.9),
            beta2=_float("beta2", 0.95),
            adam_eps=_float("adam_eps", 1e-8),
            warmup_steps=_int("warmup_steps", 0),
            warmup_ratio=_float("warmup_ratio", 0.0),
            min_lr_ratio=_float("min_lr_ratio", 0.0),
            lr_schedule=_string("lr_schedule", "cosine"),
            wsd_stable_ratio=_float("wsd_stable_ratio", 0.0),
            wsd_decay_style=_string("wsd_decay_style", "cosine"),
            layerwise_lr_decay=_float("layerwise_lr_decay", 1.0),
            muon_ns_steps=_int("muon_ns_steps", 0),
            muon_target_rms=values.get("muon_target_rms"),
            max_grad_norm=_float("max_grad_norm", 0.0),
            gradient_checkpointing=_int("gradient_checkpointing", 1),
            pad_to_multiple_of=_int("pad_to_multiple_of", 128),
            eval_interval=_int("eval_interval", 0),
            eval_steps=_int("eval_steps", 0),
            save_interval=_int("save_interval", 0),
            save_total_limit=_int("save_total_limit", 2),
            safe_serialization=_int("safe_serialization", 1),
            report_max_examples_per_split=_int("report_max_examples_per_split", 0),
            report_progress_every=_int("report_progress_every", 0),
            ema_decay=_float("ema_decay", 0.0),
            ema_update_interval=_int("ema_update_interval", 1),
            ckpt_staging_dir=_string("ckpt_staging_dir"),
        )

    def launch_config(self) -> PosttrainLaunchConfig:
        return PosttrainLaunchConfig(
            output_dir=str(self.output_dir),
            resume_from_checkpoint=str(self.resume_from_checkpoint),
            overwrite_output_dir=bool(int(self.overwrite_output_dir) == 1),
            export_dir=str(self.export_dir),
            requested_device=str(self.device),
            seed=int(self.seed),
            gradient_checkpointing=bool(self.gradient_checkpointing_enabled),
            max_steps=int(self.max_steps),
        )

    def optimizer_config(self) -> PosttrainOptimizerConfig:
        return PosttrainOptimizerConfig(
            learning_rate=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
            beta1=float(self.beta1),
            beta2=float(self.beta2),
            adam_eps=float(self.adam_eps),
            warmup_steps=int(self.warmup_steps),
            warmup_ratio=float(self.warmup_ratio),
            min_lr_ratio=float(self.min_lr_ratio),
            lr_schedule=str(self.lr_schedule),
            wsd_stable_ratio=float(self.wsd_stable_ratio),
            wsd_decay_style=str(self.wsd_decay_style),
            layerwise_lr_decay=float(self.layerwise_lr_decay),
            muon_ns_steps=(
                None if int(self.muon_ns_steps) <= 0 else int(self.muon_ns_steps)
            ),
            muon_target_rms=(
                None if self.muon_target_rms is None else float(self.muon_target_rms)
            ),
        )

    def ema_config(self) -> PosttrainEmaConfig:
        return PosttrainEmaConfig(
            decay=float(self.ema_decay),
            update_interval=int(self.ema_update_interval),
        )

    def checkpoint_policy(self) -> PosttrainCheckpointPolicy:
        return PosttrainCheckpointPolicy(
            save_total_limit=int(self.save_total_limit),
            staging_dir=(str(self.ckpt_staging_dir or "").strip() or None),
        )

    def machine_selection(self) -> PosttrainMachineSelection:
        return PosttrainMachineSelection(
            batch_size=int(self.batch_size),
            accumulation_steps=int(self.effective_accumulation_steps),
            gradient_checkpointing=bool(self.gradient_checkpointing_enabled),
        )

    def machine_selection_payload(self) -> dict[str, int]:
        return self.machine_selection().to_payload()

    @classmethod
    def machine_selection_payload_from_mapping(
        cls,
        values: Mapping[str, object],
    ) -> dict[str, int]:
        return cls.from_mapping(values).machine_selection_payload()

    def resolve_stage_max_seq_len(self, *, stage_kind: str) -> int:
        from ml.training.posttrain.curriculum import resolve_posttrain_max_seq_len

        return resolve_posttrain_max_seq_len(
            stage_kind=str(stage_kind),
            requested_max_seq_len=int(self.max_seq_len),
            curriculum_path=str(self.curriculum_path),
            curriculum_stage=str(self.curriculum_stage),
        )

    def sft_stage_config(self, *, max_seq_len: int) -> SftStageConfig:
        from ml.tasks.sft.spec import SftStageConfig
        from ml.training.posttrain.curriculum import resolve_posttrain_stage_spec

        stage_spec = resolve_posttrain_stage_spec(
            stage_kind="sft",
            curriculum_path=str(self.curriculum_path),
            curriculum_stage=str(self.curriculum_stage),
        )

        return SftStageConfig(
            train_data_path=str(self.train_data),
            eval_data_path=str(self.eval_data),
            test_data_path=str(self.test_data),
            curriculum_path=str(self.curriculum_path),
            curriculum_stage=str(self.curriculum_stage),
            max_seq_len=int(max_seq_len),
            selection=(
                "all_examples" if stage_spec is None else str(stage_spec.selection)
            ),
            crop_policy=(
                "tail_tokens" if stage_spec is None else str(stage_spec.crop_policy)
            ),
            batch_size=int(self.batch_size),
            pad_to_multiple_of=int(self.pad_to_multiple_of),
            seed=int(self.seed),
            accum_steps=int(self.effective_accumulation_steps),
            max_steps=int(self.max_steps),
            eval_interval=int(self.eval_interval),
            eval_steps=int(self.eval_steps),
            save_interval=int(self.save_interval),
            max_grad_norm=float(self.max_grad_norm),
            report_max_examples_per_split=int(self.report_max_examples_per_split),
            report_progress_every=int(self.report_progress_every),
        )

    def build_run_spec(
        self,
        *,
        stage: str,
        resolved_max_seq_len: int,
    ) -> PosttrainRunSpec:
        return PosttrainRunSpec(
            run=RunSpec(
                run_kind=f"posttrain_{str(stage)}",
                model=ModelSpec.default(),
                seed=int(self.seed),
                max_steps=int(self.max_steps),
                output_dir=str(self.output_dir),
                resume_from_checkpoint=(str(self.resume_from_checkpoint).strip() or None),
                safe_serialization=bool(self.safe_serialization_enabled),
            ),
            resolved_max_seq_len=int(resolved_max_seq_len),
            launch=self.launch_config(),
            optimizer=self.optimizer_config(),
            ema=self.ema_config(),
            checkpoint_policy=self.checkpoint_policy(),
            safe_serialization=bool(self.safe_serialization_enabled),
        )

__all__ = [
    "PosttrainCheckpointPolicy",
    "PosttrainCheckpointSpec",
    "PosttrainEmaConfig",
    "PosttrainExportSpec",
    "PosttrainLaunchConfig",
    "PosttrainMachineSelection",
    "PosttrainOptimizerConfig",
    "PosttrainReportSpec",
    "PosttrainRunContext",
    "PosttrainRunSpec",
    "PosttrainStageArgs",
    "PosttrainStepResult",
    "ResumeState",
]
