from __future__ import annotations

import os
from dataclasses import dataclass, replace

from ml.core.common.mapping import object_mapping
from ml.errors import SophiaUsageError
from ml.tasks.pretrain.pipeline import build_pretrain_args
from ml.training.pretrain.profiles import VALIDATION_PROFILE
from ml.training.pretrain.run_config import PretrainRunConfig

VALIDATE_DIRNAME = "pretrain_check"


@dataclass(frozen=True)
class PretrainValidationConfig:
    data_path: str = "dataset/pretrain"
    tokenizer_path: str = ""
    output_dir: str = ""
    overwrite_output_dir: bool = False
    device: str = "cuda:0"
    total_tokens: int = 0
    seed: int = 42


def pretrain_validation_config_from_args(args: object) -> PretrainValidationConfig:
    payload = object_mapping(args)
    return PretrainValidationConfig(
        data_path=str(payload.get("data_path", "dataset/pretrain")),
        tokenizer_path=str(payload.get("tokenizer_path", "") or ""),
        output_dir=str(payload.get("output_dir", "") or ""),
        overwrite_output_dir=bool(int(payload.get("overwrite_output_dir", 0))),
        device=str(payload.get("device", "cuda:0") or "cuda:0"),
        total_tokens=int(payload.get("total_tokens", 0) or 0),
        seed=int(payload.get("seed", 42)),
    )


def resolve_pretrain_validation_config(
    config: PretrainValidationConfig,
) -> PretrainValidationConfig:
    device = str(config.device or "").strip()
    if not device:
        raise SophiaUsageError("[ERR] --device must be non-empty.")
    data_path = str(config.data_path or "").strip()
    if not data_path:
        raise SophiaUsageError("[ERR] --data_path must be non-empty.")
    if int(config.total_tokens) < 0:
        raise SophiaUsageError("[ERR] --total_tokens must be >= 0")
    return replace(
        config,
        data_path=data_path,
        tokenizer_path=str(config.tokenizer_path or "").strip(),
        output_dir=str(config.output_dir or "").strip(),
        device=device,
    )


def _resolve_output_dir(raw: str) -> str:
    if str(raw or "").strip():
        return os.path.abspath(str(raw))
    return os.path.abspath(os.path.join("out", VALIDATE_DIRNAME))


def build_validation_args(config: PretrainValidationConfig) -> PretrainRunConfig:
    args = build_pretrain_args(
        data_path=str(config.data_path),
        tokenizer_path=str(config.tokenizer_path),
        output_dir=_resolve_output_dir(str(config.output_dir)),
        overwrite_output_dir=(1 if bool(config.overwrite_output_dir) else 0),
        profile=VALIDATION_PROFILE,
    )
    total_tokens_override = int(config.total_tokens or 0)
    return replace(
        args,
        device=str(config.device),
        seed=int(config.seed),
        async_checkpoint=0,
        async_metrics=0,
        target_tokens_per_update=int(args.seq_len),
        batch_size=1,
        accumulation_steps=1,
        warmup_steps=0,
        warmup_ratio=0.0,
        total_tokens=(
            int(total_tokens_override)
            if total_tokens_override > 0
            else int(args.total_tokens)
        ),
        eval_steps=16,
        save_total_limit=1,
        save_best=0,
        enable_checkpoints=0,
        _sophia_run_kind="pretrain_check",
    )


__all__ = [
    "PretrainValidationConfig",
    "build_validation_args",
    "pretrain_validation_config_from_args",
    "resolve_pretrain_validation_config",
]
