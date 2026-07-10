from __future__ import annotations

import os
from pathlib import Path

from ml.training.posttrain.defaults import DEFAULT_POSTTRAIN_CURRICULUM_PATH


def discover_repo_root() -> Path:
    cwd = Path(os.getcwd()).resolve()
    for root in (cwd, *cwd.parents):
        if (root / "pyproject.toml").is_file() and (root / "ml").is_dir():
            return root
    return Path(__file__).resolve().parents[3]


REPO_ROOT = discover_repo_root()
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "out" / "local_rehearsal"
DEFAULT_POSTTRAIN_CURRICULUM = REPO_ROOT / DEFAULT_POSTTRAIN_CURRICULUM_PATH
DEFAULT_PRETRAIN_DATA = REPO_ROOT / "dataset" / "pretrain_tokens"
DEFAULT_SFT_TRAIN = REPO_ROOT / "dataset" / "sft" / "train.jsonl"
DEFAULT_SFT_EVAL = REPO_ROOT / "dataset" / "sft" / "val.jsonl"
DEFAULT_SFT_TEST = REPO_ROOT / "dataset" / "sft" / "test.jsonl"
DEFAULT_PROGRESS_INTERVAL_SECONDS = 15.0


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_POSTTRAIN_CURRICULUM",
    "DEFAULT_PRETRAIN_DATA",
    "DEFAULT_PROGRESS_INTERVAL_SECONDS",
    "DEFAULT_SFT_EVAL",
    "DEFAULT_SFT_TEST",
    "DEFAULT_SFT_TRAIN",
    "REPO_ROOT",
    "discover_repo_root",
]
