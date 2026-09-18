from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml.tasks.pretrain.pipeline import build_pretrain_args


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pretrain_base_checkpoint_policy_is_pinned_and_self_consistent() -> None:
    path = REPO_ROOT / "configs/eval/pretrain_base_checkpoint_policy.json"
    policy = json.loads(path.read_text(encoding="utf-8"))

    assert policy["schema"] == "sophia_pretrain_base_checkpoint_policy_v1"
    assert policy["status"] == "fixed_before_target_training"
    assert policy["checkpoint_persistence"]["overwrite_output_dir"] is False
    assert policy["cadence"]["full_checkpoint_every_updates"] == 100

    for name in (
        "full_generation_suite",
        "generation_decontamination_signatures",
        "chinese_multiple_choice",
        "full_benchmark_decontamination",
    ):
        asset = policy["evaluation_assets"][name]
        asset_path = Path(asset["path"])
        if not asset_path.is_absolute():
            asset_path = REPO_ROOT / asset_path
        assert asset_path.is_file(), name
        assert _sha256(asset_path) == asset["sha256"], name

    assert "hard_thresholds" not in policy["release_evidence"]


def test_release_run_defaults_implement_checkpoint_policy() -> None:
    args = build_pretrain_args(data_path="dataset/fresh/train")

    assert args.log_interval == 10
    assert args.save_interval == 100
    assert args.save_total_limit == 6
    assert args.async_checkpoint == 1
    assert args.save_best == 0

    policy = json.loads(
        (REPO_ROOT / "configs/eval/pretrain_base_checkpoint_policy.json").read_text(
            encoding="utf-8"
        )
    )["checkpoint_persistence"]
    assert args.save_total_limit == policy["keep_latest_full"]
