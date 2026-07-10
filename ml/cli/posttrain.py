from __future__ import annotations

from typing import Any


def add_common_posttrain_args(
    parser: Any,
    *,
    curriculum_path_default: str,
    curriculum_stage_default: str,
    accumulation_steps_default: int,
    max_steps_default: int,
    learning_rate_default: float,
    weight_decay_default: float,
    eval_interval_default: int,
    eval_steps_default: int,
    save_interval_default: int,
    report_max_examples_per_split_default: int,
    report_progress_every_default: int,
) -> None:
    parser.add_argument("--export_dir", required=True, type=str)
    parser.add_argument("--train_data", required=True, type=str)
    parser.add_argument("--eval_data", default="", type=str)
    parser.add_argument("--test_data", default="", type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--resume_from_checkpoint", default="", type=str)
    parser.add_argument("--overwrite_output_dir", default=0, type=int, choices=[0, 1])
    parser.add_argument("--machine_recipe_json", default="", type=str)
    parser.add_argument(
        "--allow_missing_machine_recipe",
        default=0,
        type=int,
        choices=[0, 1],
        help="0=require a signed machine recipe for release post-training; 1=allow local selection/rehearsal runs without one.",
    )
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_seq_len", default=0, type=int)
    parser.add_argument("--curriculum_path", default=curriculum_path_default, type=str)
    parser.add_argument("--curriculum_stage", default=curriculum_stage_default, type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument(
        "--accumulation_steps",
        default=accumulation_steps_default,
        type=int,
    )
    parser.add_argument("--max_steps", default=max_steps_default, type=int)
    parser.add_argument("--learning_rate", default=learning_rate_default, type=float)
    parser.add_argument("--weight_decay", default=weight_decay_default, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--warmup_ratio", default=0.0, type=float)
    parser.add_argument("--min_lr_ratio", default=0.0, type=float)
    parser.add_argument("--lr_schedule", default="cosine", type=str)
    parser.add_argument("--wsd_stable_ratio", default=0.0, type=float)
    parser.add_argument("--wsd_decay_style", default="cosine", type=str)
    parser.add_argument("--layerwise_lr_decay", default=1.0, type=float)
    parser.add_argument("--muon_ns_steps", default=0, type=int)
    parser.add_argument("--muon_target_rms", default=None, type=float)
    parser.add_argument("--max_grad_norm", default=0.0, type=float)
    parser.add_argument(
        "--gradient_checkpointing",
        default=1,
        type=int,
        choices=[0, 1],
    )
    parser.add_argument("--pad_to_multiple_of", default=128, type=int)
    parser.add_argument("--eval_interval", default=eval_interval_default, type=int)
    parser.add_argument("--eval_steps", default=eval_steps_default, type=int)
    parser.add_argument("--save_interval", default=save_interval_default, type=int)
    parser.add_argument("--save_total_limit", default=2, type=int)
    parser.add_argument(
        "--safe_serialization",
        default=1,
        type=int,
        choices=[0, 1],
    )
    parser.add_argument(
        "--report_max_examples_per_split",
        default=report_max_examples_per_split_default,
        type=int,
    )
    parser.add_argument(
        "--report_progress_every",
        default=report_progress_every_default,
        type=int,
    )
    parser.add_argument("--ema_decay", default=0.0, type=float)
    parser.add_argument("--ema_update_interval", default=1, type=int)
    parser.add_argument("--ckpt_staging_dir", default="", type=str)
