from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from ml.errors import SophiaUsageError
from ml.training.posttrain.curriculum import (
    default_posttrain_curriculum_stage,
    resolve_posttrain_max_seq_len,
)
from ml.training.rehearsal.config import RehearsalConfig
from ml.training.rehearsal.defaults import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POSTTRAIN_CURRICULUM,
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_PRETRAIN_DATA,
    DEFAULT_SFT_EVAL,
    DEFAULT_SFT_TEST,
    DEFAULT_SFT_TRAIN,
)


@dataclass(frozen=True)
class RehearsalStage:
    name: str
    argv: tuple[str, ...]
    output_dir: str
    stage_type: str = "run"
    reason: str = ""


@dataclass(frozen=True)
class RehearsalPlan:
    output_root: str
    stages: tuple[RehearsalStage, ...]
    progress_interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS


def require_existing_path(path: str, *, label: str) -> str:
    resolved = os.path.abspath(str(path or "").strip())
    if not resolved:
        raise SophiaUsageError(f"[ERR] missing required path for {label}.")
    if not os.path.exists(resolved):
        raise SophiaUsageError(f"[ERR] {label} not found: {resolved}")
    return resolved

def argv_value(argv: tuple[str, ...], flag: str) -> str:
    try:
        idx = argv.index(str(flag))
    except ValueError:
        return ""
    next_idx = int(idx) + 1
    if next_idx >= len(argv):
        return ""
    return str(argv[next_idx])


def export_config_max_seq_len(export_dir: str) -> int:
    resolved_dir = Path(os.path.abspath(str(export_dir or "").strip()))
    config_path = resolved_dir / "config.json"
    if not config_path.is_file():
        raise SophiaUsageError(f"[ERR] missing exported config for rehearsal stage input: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SophiaUsageError(f"[ERR] invalid exported config JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise SophiaUsageError(f"[ERR] exported config must be a JSON object: {config_path}")
    max_seq_len = int(payload.get("max_seq_len", 0) or 0)
    if max_seq_len <= 0:
        raise SophiaUsageError(
            "[ERR] exported config is missing a valid max_seq_len for rehearsal stage input: "
            f"{config_path}"
        )
    return int(max_seq_len)


def validate_posttrain_stage_inputs(stage: RehearsalStage) -> None:
    stage_kind = str(stage.name or "").strip()
    if stage.stage_type != "run" or stage_kind != "sft":
        return
    export_dir = argv_value(stage.argv, "--export_dir")
    curriculum_path = argv_value(stage.argv, "--curriculum_path")
    curriculum_stage = argv_value(stage.argv, "--curriculum_stage")
    if not export_dir or not curriculum_path or not curriculum_stage:
        raise SophiaUsageError(
            "[ERR] rehearsal stage is missing required post-training inputs: "
            f"{stage.name}"
        )
    exported_max_seq_len = export_config_max_seq_len(export_dir)
    required_max_seq_len = resolve_posttrain_max_seq_len(
        stage_kind=stage_kind,
        requested_max_seq_len=0,
        curriculum_path=str(curriculum_path),
        curriculum_stage=str(curriculum_stage),
    )
    if int(exported_max_seq_len) >= int(required_max_seq_len):
        return
    raise SophiaUsageError(
        "[ERR] rehearsal stage input incompatible: "
        f"{stage.name} requires max_seq_len>={int(required_max_seq_len)} "
        f"for curriculum_stage={str(curriculum_stage)!r}, "
        f"but {os.path.abspath(str(export_dir))} exports max_seq_len={int(exported_max_seq_len)}."
    )


def build_rehearsal_plan(config: RehearsalConfig) -> RehearsalPlan:
    output_root = os.path.abspath(str(config.output_root))
    pretrain_data = require_existing_path(str(config.pretrain_data), label="pretrain_data")
    posttrain_curriculum_path = require_existing_path(
        str(config.posttrain_curriculum_path),
        label="posttrain_curriculum_path",
    )
    sft_train_data = require_existing_path(str(config.sft_train_data), label="sft_train_data")
    sft_eval_data = require_existing_path(str(config.sft_eval_data), label="sft_eval_data")
    sft_test_data = require_existing_path(str(config.sft_test_data), label="sft_test_data")

    overwrite = "1" if bool(config.overwrite_output_dir) else "0"
    posttrain_batch_size = str(max(int(config.posttrain_batch_size), 1))
    eval_steps = str(max(int(config.posttrain_eval_steps), 1))
    report_max_examples = str(max(int(config.posttrain_report_max_examples_per_split), 0))
    report_progress_every = str(max(int(config.posttrain_report_progress_every), 0))
    pretrain_tokens = int(config.pretrain_tokens or 0)
    pretrain_check_dir = os.path.join(output_root, "pretrain_check")
    sft_dir = os.path.join(output_root, "sft")

    stages: list[RehearsalStage] = [
        RehearsalStage(
            name="pretrain_check",
            output_dir=pretrain_check_dir,
            argv=(
                sys.executable,
                "-m",
                "ml.cli.pretrain_check",
                "--data_path",
                pretrain_data,
                "--output_dir",
                pretrain_check_dir,
                "--overwrite_output_dir",
                overwrite,
                "--device",
                str(config.device),
                "--seed",
                str(int(config.seed)),
                *(
                    ()
                    if pretrain_tokens <= 0
                    else ("--total_tokens", str(int(pretrain_tokens)))
                ),
            ),
        ),
        RehearsalStage(
            name="sft",
            output_dir=sft_dir,
            argv=(
                sys.executable,
                "-m",
                "ml.cli.sft",
                "--export_dir",
                pretrain_check_dir,
                "--train_data",
                sft_train_data,
                "--eval_data",
                sft_eval_data,
                "--test_data",
                sft_test_data,
                "--output_dir",
                sft_dir,
                "--overwrite_output_dir",
                overwrite,
                "--allow_missing_machine_recipe",
                "1",
                "--device",
                str(config.device),
                "--seed",
                str(int(config.seed)),
                "--curriculum_path",
                posttrain_curriculum_path,
                "--curriculum_stage",
                default_posttrain_curriculum_stage(stage_kind="sft"),
                "--batch_size",
                posttrain_batch_size,
                "--accumulation_steps",
                "1",
                "--max_steps",
                str(max(int(config.sft_max_steps), 1)),
                "--eval_interval",
                "1",
                "--eval_steps",
                eval_steps,
                "--save_interval",
                "1",
                "--save_total_limit",
                "1",
                "--report_max_examples_per_split",
                report_max_examples,
                "--report_progress_every",
                report_progress_every,
            ),
        ),
    ]

    return RehearsalPlan(
        output_root=output_root,
        stages=tuple(stages),
        progress_interval_seconds=max(float(config.progress_interval_seconds), 0.0),
    )

__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_POSTTRAIN_CURRICULUM",
    "DEFAULT_PRETRAIN_DATA",
    "DEFAULT_SFT_EVAL",
    "DEFAULT_SFT_TEST",
    "DEFAULT_SFT_TRAIN",
    "RehearsalPlan",
    "RehearsalStage",
    "argv_value",
    "build_rehearsal_plan",
    "export_config_max_seq_len",
    "require_existing_path",
    "validate_posttrain_stage_inputs",
]
