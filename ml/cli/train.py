from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ml.cli.support import run_cli
from ml.tasks.pretrain.pipeline import build_pretrain_args, run
from ml.training.pretrain.profiles import RELEASE_PROFILE
from ml.training.pretrain.release_config import (
    canonical_pretrain_machine_recipe_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sophia Pretraining (single stack)")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Dataset root or train split path (must contain sibling val/manifest.json and test/manifest.json).",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help=(
            "Tokenizer directory (expects tokenizer.json). "
            "Default: ml/modeling/text (bundled tokenizer bundle)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Run output directory. Empty = auto-derived Sophia default.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="Checkpoint path, checkpoint directory, or one of: auto/latest.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        type=int,
        default=0,
        choices=[0, 1],
        help="Delete a recognized Sophia run directory before starting a fresh run.",
    )
    parser.add_argument(
        "--machine_recipe_json",
        type=str,
        default=canonical_pretrain_machine_recipe_path(),
        help=(
            "Signed RTX 5090 BF16 machine recipe. Override only with a newly "
            "measured and signed target-GPU recipe."
        ),
    )
    parser.add_argument(
        "--decay_data_path",
        type=str,
        default="",
        help=(
            "Optional high-quality dataset (root or train split path) used for the "
            "WSD decay phase. Empty = train on --data_path for the whole run."
        ),
    )
    return parser


def main() -> None:
    parsed = build_parser().parse_args()
    run(
        build_pretrain_args(
            data_path=str(parsed.data_path),
            tokenizer_path=str(parsed.tokenizer_path),
            output_dir=str(parsed.output_dir),
            resume_from_checkpoint=str(parsed.resume_from_checkpoint),
            overwrite_output_dir=int(parsed.overwrite_output_dir),
            machine_recipe_json=str(parsed.machine_recipe_json),
            decay_data_path=str(parsed.decay_data_path),
            profile=RELEASE_PROFILE,
        )
    )


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
