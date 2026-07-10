from __future__ import annotations

import argparse
from collections.abc import Sequence

from ml.cli.support import run_cli


def build_parser() -> argparse.ArgumentParser:
    from ml.training import rehearsal as rehearsal_mod
    from ml.training.posttrain.defaults import (
        REHEARSAL_POSTTRAIN_BATCH_SIZE,
        REHEARSAL_POSTTRAIN_EVAL_STEPS,
        REHEARSAL_POSTTRAIN_REPORT_MAX_EXAMPLES_PER_SPLIT,
        REHEARSAL_POSTTRAIN_REPORT_PROGRESS_EVERY,
        REHEARSAL_SFT_MAX_STEPS,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run the local same-chain rehearsal for Sophia training. "
            "The chain is pinned to pretrain.validate -> SFT."
        )
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(rehearsal_mod.DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite_output_dir", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--pretrain_data",
        type=str,
        default=str(rehearsal_mod.DEFAULT_PRETRAIN_DATA),
    )
    parser.add_argument(
        "--posttrain_curriculum_path",
        type=str,
        default=str(rehearsal_mod.DEFAULT_POSTTRAIN_CURRICULUM),
    )
    parser.add_argument("--pretrain_tokens", type=int, default=0)
    parser.add_argument(
        "--sft_train_data",
        type=str,
        default=str(rehearsal_mod.DEFAULT_SFT_TRAIN),
    )
    parser.add_argument(
        "--sft_eval_data",
        type=str,
        default=str(rehearsal_mod.DEFAULT_SFT_EVAL),
    )
    parser.add_argument(
        "--sft_test_data",
        type=str,
        default=str(rehearsal_mod.DEFAULT_SFT_TEST),
    )
    parser.add_argument(
        "--sft_max_steps",
        type=int,
        default=REHEARSAL_SFT_MAX_STEPS,
    )
    parser.add_argument(
        "--posttrain_batch_size",
        type=int,
        default=REHEARSAL_POSTTRAIN_BATCH_SIZE,
    )
    parser.add_argument(
        "--posttrain_eval_steps",
        type=int,
        default=REHEARSAL_POSTTRAIN_EVAL_STEPS,
    )
    parser.add_argument(
        "--posttrain_report_max_examples_per_split",
        type=int,
        default=REHEARSAL_POSTTRAIN_REPORT_MAX_EXAMPLES_PER_SPLIT,
    )
    parser.add_argument(
        "--posttrain_report_progress_every",
        type=int,
        default=REHEARSAL_POSTTRAIN_REPORT_PROGRESS_EVERY,
    )
    parser.add_argument(
        "--progress_interval_seconds",
        type=float,
        default=rehearsal_mod.DEFAULT_PROGRESS_INTERVAL_SECONDS,
        help="Emit a stage heartbeat every N seconds while a child stage is still running.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    from ml.training.rehearsal import (
        RehearsalConfig,
        resolve_rehearsal_config,
        run,
    )

    config = RehearsalConfig.from_args(build_parser().parse_args(argv))
    run(resolve_rehearsal_config(config))


if __name__ == "__main__":
    raise SystemExit(run_cli(lambda: main()))
