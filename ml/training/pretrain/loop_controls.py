from __future__ import annotations

from dataclasses import dataclass

from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.training.pretrain.value_semantics import coerce_int, normalized_str


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
    max_grad_norm: float
    grad_clip_mode: str
    agc_clip: float
    agc_eps: float
    agc_exclude_bias_and_norm: int
    log_interval: int
    save_interval: int
    save_total_limit: int
    save_weights_steps: int
    save_weights_total_limit: int
    async_checkpoint: int
    async_metrics: int
    ckpt_staging_dir: str
    save_best: int
    enable_checkpoints: int
    ema_eval: int
    ema_use_for_export: int
    ema_in_ckpt: int
    warmup_steps: int
    wsd_stable_ratio: float
    lr_schedule: str
    stability_guard_enabled: int
    stability_guard_loss_window: int
    stability_guard_loss_max: float
    stability_guard_grad_norm_max: float
    stability_guard_grad_window: int
    stability_guard_grad_spike_limit: int
    stability_guard_min_step: int

    @classmethod
    def from_bound(
        cls,
        cfg: PretrainRunConfig,
        state: PretrainRuntimeState,
    ) -> PretrainLoopControl:
        return cls(
            total_tokens=int(cfg.total_tokens),
            accumulation_steps=max(int(state.accumulation_steps), 1),
            max_grad_norm=float(cfg.max_grad_norm),
            grad_clip_mode=normalized_str(cfg.grad_clip_mode, default="norm"),
            agc_clip=float(cfg.agc_clip),
            agc_eps=float(cfg.agc_eps),
            agc_exclude_bias_and_norm=int(cfg.agc_exclude_bias_and_norm),
            log_interval=int(state.log_interval),
            save_interval=int(state.save_interval),
            save_total_limit=int(cfg.save_total_limit),
            save_weights_steps=int(cfg.save_weights_steps),
            save_weights_total_limit=int(cfg.save_weights_total_limit),
            async_checkpoint=int(cfg.async_checkpoint),
            async_metrics=int(cfg.async_metrics),
            ckpt_staging_dir=str(cfg.ckpt_staging_dir),
            save_best=int(cfg.save_best),
            enable_checkpoints=int(cfg.enable_checkpoints),
            ema_eval=int(cfg.ema_eval),
            ema_use_for_export=int(cfg.ema_use_for_export),
            ema_in_ckpt=int(cfg.ema_in_ckpt),
            warmup_steps=int(cfg.warmup_steps),
            wsd_stable_ratio=float(cfg.wsd_stable_ratio),
            lr_schedule=normalized_str(cfg.lr_schedule, default="cosine"),
            stability_guard_enabled=int(cfg.stability_guard_enabled),
            stability_guard_loss_window=int(cfg.stability_guard_loss_window),
            stability_guard_loss_max=float(cfg.stability_guard_loss_max),
            stability_guard_grad_norm_max=float(cfg.stability_guard_grad_norm_max),
            stability_guard_grad_window=int(cfg.stability_guard_grad_window),
            stability_guard_grad_spike_limit=int(cfg.stability_guard_grad_spike_limit),
            stability_guard_min_step=int(cfg.stability_guard_min_step),
        )


__all__ = [
    "PretrainDataControl",
    "PretrainLoopControl",
]
