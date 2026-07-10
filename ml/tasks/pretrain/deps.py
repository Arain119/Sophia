from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable


@dataclass(frozen=True)
class PretrainDeps:
    setup_train_runtime_and_device: Callable[..., object]
    ensure_machine_adaptive_signature: Callable[..., None]
    runtime_preflight_config: Callable[..., object]
    decoder_model_cls: Callable[..., object]
    clear_cuda: Callable[[], None]
    prepare_output_dir_and_resume: Callable[..., object]
    load_checkpoint: Callable[..., object]
    prepare_pretrain_eval_data: Callable[..., object]
    load_manifest_or_die: Callable[..., object]
    load_tokenizer_and_validate_manifest: Callable[..., object]
    build_decoder_config: Callable[..., object]
    load_or_init_model: Callable[..., object]
    sync_runtime_batch_capacity: Callable[..., None]
    maybe_resume_from_checkpoint: Callable[..., object]
    auto_observability_policy: Callable[..., object]
    auto_ema_policy: Callable[..., object]
    write_release_pretrain_profile: Callable[..., None]
    write_machine_recipe_summary: Callable[..., None]
    build_runner: Callable[..., object]
    seed_active_stage_plan_recipe: Callable[..., object]
    realize_stage_plan_machine_recipe: Callable[..., object]
    apply_stage_recipe: Callable[..., object]
    run_train_loop: Callable[..., object]
    create_optimizer: Callable[..., object]
    build_lr_scheduler: Callable[..., object]
    model_ema_cls: type


__all__ = ["PretrainDeps"]
