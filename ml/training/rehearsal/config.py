from __future__ import annotations

from dataclasses import dataclass, replace

from ml.errors import SophiaUsageError
from ml.core.common.mapping import object_mapping
from ml.training.rehearsal.defaults import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POSTTRAIN_CURRICULUM,
    DEFAULT_PRETRAIN_DATA,
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_SFT_EVAL,
    DEFAULT_SFT_TEST,
    DEFAULT_SFT_TRAIN,
)


@dataclass(frozen=True)
class RehearsalConfig:
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    device: str = "cuda:0"
    seed: int = 1337
    overwrite_output_dir: bool = True
    pretrain_data: str = str(DEFAULT_PRETRAIN_DATA)
    posttrain_curriculum_path: str = str(DEFAULT_POSTTRAIN_CURRICULUM)
    pretrain_tokens: int = 0
    sft_train_data: str = str(DEFAULT_SFT_TRAIN)
    sft_eval_data: str = str(DEFAULT_SFT_EVAL)
    sft_test_data: str = str(DEFAULT_SFT_TEST)
    sft_max_steps: int = 1
    posttrain_batch_size: int = 1
    posttrain_eval_steps: int = 1
    posttrain_report_max_examples_per_split: int = 0
    posttrain_report_progress_every: int = 0
    progress_interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS

    @classmethod
    def from_args(cls, args: object) -> RehearsalConfig:
        payload = object_mapping(args)
        return cls(
            output_root=str(payload.get("output_root", DEFAULT_OUTPUT_ROOT) or DEFAULT_OUTPUT_ROOT),
            device=str(payload.get("device", "cuda:0") or "cuda:0"),
            seed=int(payload.get("seed", 1337)),
            overwrite_output_dir=bool(int(payload.get("overwrite_output_dir", 1))),
            pretrain_data=str(
                payload.get("pretrain_data", DEFAULT_PRETRAIN_DATA)
                or DEFAULT_PRETRAIN_DATA
            ),
            posttrain_curriculum_path=str(
                payload.get("posttrain_curriculum_path", DEFAULT_POSTTRAIN_CURRICULUM)
                or DEFAULT_POSTTRAIN_CURRICULUM
            ),
            pretrain_tokens=int(payload.get("pretrain_tokens", 0) or 0),
            sft_train_data=str(payload.get("sft_train_data", DEFAULT_SFT_TRAIN) or DEFAULT_SFT_TRAIN),
            sft_eval_data=str(payload.get("sft_eval_data", DEFAULT_SFT_EVAL) or DEFAULT_SFT_EVAL),
            sft_test_data=str(payload.get("sft_test_data", DEFAULT_SFT_TEST) or DEFAULT_SFT_TEST),
            sft_max_steps=int(payload.get("sft_max_steps", 1) or 0),
            posttrain_batch_size=int(payload.get("posttrain_batch_size", 1) or 0),
            posttrain_eval_steps=int(payload.get("posttrain_eval_steps", 1) or 0),
            posttrain_report_max_examples_per_split=int(
                payload.get("posttrain_report_max_examples_per_split", 0) or 0
            ),
            posttrain_report_progress_every=int(
                payload.get("posttrain_report_progress_every", 0) or 0
            ),
            progress_interval_seconds=float(
                payload.get(
                    "progress_interval_seconds",
                    DEFAULT_PROGRESS_INTERVAL_SECONDS,
                )
                or 0.0
            ),
        )

def resolve_rehearsal_config(config: RehearsalConfig) -> RehearsalConfig:
    output_root = str(config.output_root or "").strip()
    if not output_root:
        raise SophiaUsageError("[ERR] --output_root must be non-empty.")
    device = str(config.device or "").strip()
    if not device:
        raise SophiaUsageError("[ERR] --device must be non-empty.")
    return replace(
        config,
        output_root=output_root,
        device=device,
        progress_interval_seconds=max(float(config.progress_interval_seconds), 0.0),
    )


__all__ = [
    "RehearsalConfig",
    "resolve_rehearsal_config",
]
