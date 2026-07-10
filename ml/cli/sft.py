from __future__ import annotations

import argparse

from ml.cli.support import run_cli
from ml.cli.posttrain import add_common_posttrain_args
from ml.tasks.sft.train import run
from ml.training.posttrain.curriculum import default_posttrain_curriculum_stage
from ml.training.posttrain.defaults import (
    DEFAULT_POSTTRAIN_CURRICULUM_PATH,
    RELEASE_SFT_EVAL_INTERVAL,
    RELEASE_SFT_EVAL_STEPS,
    RELEASE_SFT_LEARNING_RATE,
    RELEASE_SFT_MAX_STEPS,
    RELEASE_SFT_REPORT_MAX_EXAMPLES_PER_SPLIT,
    RELEASE_SFT_REPORT_PROGRESS_EVERY,
    RELEASE_SFT_SAVE_INTERVAL,
    RELEASE_SFT_TARGET_EXAMPLES_PER_UPDATE,
    RELEASE_SFT_WEIGHT_DECAY,
)
from ml.training.posttrain.types import PosttrainStageArgs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sophia supervised fine-tuning")
    add_common_posttrain_args(
        parser,
        curriculum_path_default=str(DEFAULT_POSTTRAIN_CURRICULUM_PATH),
        curriculum_stage_default=default_posttrain_curriculum_stage(stage_kind="sft"),
        accumulation_steps_default=RELEASE_SFT_TARGET_EXAMPLES_PER_UPDATE,
        max_steps_default=RELEASE_SFT_MAX_STEPS,
        learning_rate_default=RELEASE_SFT_LEARNING_RATE,
        weight_decay_default=RELEASE_SFT_WEIGHT_DECAY,
        eval_interval_default=RELEASE_SFT_EVAL_INTERVAL,
        eval_steps_default=RELEASE_SFT_EVAL_STEPS,
        save_interval_default=RELEASE_SFT_SAVE_INTERVAL,
        report_max_examples_per_split_default=RELEASE_SFT_REPORT_MAX_EXAMPLES_PER_SPLIT,
        report_progress_every_default=RELEASE_SFT_REPORT_PROGRESS_EVERY,
    )
    return parser


def main() -> None:
    run(PosttrainStageArgs.from_mapping(vars(build_parser().parse_args())))


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
