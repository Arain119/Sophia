from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SftStageConfig:
    train_data_path: str
    eval_data_path: str
    test_data_path: str
    curriculum_path: str
    curriculum_stage: str
    max_seq_len: int
    selection: str
    crop_policy: str
    batch_size: int
    pad_to_multiple_of: int
    seed: int
    accum_steps: int
    max_steps: int
    eval_interval: int
    eval_steps: int
    save_interval: int
    max_grad_norm: float
    report_max_examples_per_split: int
    report_progress_every: int


@dataclass
class SftDataState:
    train_iter: object
    eval_iter: object | None
    test_iter: object | None
    running_loss_sum: float
    running_tokens: int
    samples_seen: int


__all__ = ["SftDataState", "SftStageConfig"]
