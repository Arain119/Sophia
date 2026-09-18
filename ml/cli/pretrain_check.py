from __future__ import annotations

import argparse
import os

from ml.cli.support import run_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Same-chain local validation for the release Sophia pretrain pipeline."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join("dataset", "pretrain"),
        help="Dataset root or train split path containing train/val/test token shards.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Optional explicit tokenizer directory. Empty = resolve from manifest fingerprint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Validation output directory. Empty = `<repo>/out/pretrain_check`.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        type=int,
        default=0,
        choices=[0, 1],
        help="Delete a previously marked validation run directory before starting fresh.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Execution device. Use CUDA for realistic same-chain local validation.",
    )
    parser.add_argument(
        "--total_tokens",
        type=int,
        default=0,
        help="Optional validation token budget override. 0 = use the validation default.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main() -> None:
    parsed = build_parser().parse_args()

    from ml.tasks.pretrain.pipeline import PretrainPipeline
    from ml.tasks.pretrain.validate import (
        build_validation_args,
        pretrain_validation_config_from_args,
        resolve_pretrain_validation_config,
    )

    config = pretrain_validation_config_from_args(parsed)
    PretrainPipeline(
        args=build_validation_args(resolve_pretrain_validation_config(config))
    ).run()


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
