from __future__ import annotations

from dataclasses import dataclass

from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.training.pretrain.value_semantics import coerce_int


@dataclass(frozen=True)
class PretrainDataControl:
    tokenizer_path: str
    eval_data_path: str
    test_data_path: str
    batch_size: int
    seed: int
    eval_interval: int
    eval_steps: int

    @classmethod
    def from_bound(
        cls,
        cfg: PretrainRunConfig,
        state: PretrainRuntimeState,
    ) -> PretrainDataControl:
        return cls(
            tokenizer_path=str(cfg.tokenizer_path),
            eval_data_path=str(cfg.eval_data_path),
            test_data_path=str(cfg.test_data_path),
            batch_size=int(state.batch_size),
            seed=coerce_int(cfg.seed),
            eval_interval=max(int(state.eval_interval), 0),
            eval_steps=max(int(state.eval_steps), 0),
        )


@dataclass(frozen=True)
class PretrainLoopControl:
    total_tokens: int
    accumulation_steps: int
    log_interval: int
    save_interval: int
    save_total_limit: int
    async_checkpoint: int
    async_metrics: int
    ckpt_staging_dir: str
    save_best: int
    enable_checkpoints: int
    warmup_steps: int

    @classmethod
    def from_bound(
        cls,
        cfg: PretrainRunConfig,
        state: PretrainRuntimeState,
    ) -> PretrainLoopControl:
        return cls(
            total_tokens=int(cfg.total_tokens),
            accumulation_steps=max(int(state.accumulation_steps), 1),
            log_interval=int(state.log_interval),
            save_interval=int(state.save_interval),
            save_total_limit=int(cfg.save_total_limit),
            async_checkpoint=int(cfg.async_checkpoint),
            async_metrics=int(cfg.async_metrics),
            ckpt_staging_dir=str(cfg.ckpt_staging_dir),
            save_best=int(cfg.save_best),
            enable_checkpoints=int(cfg.enable_checkpoints),
            warmup_steps=int(cfg.warmup_steps),
        )


__all__ = [
    "PretrainDataControl",
    "PretrainLoopControl",
]
