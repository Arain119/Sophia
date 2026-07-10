from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from functools import lru_cache

from ml.training.pretrain.profiles import PretrainProfile, pretrain_profile_to_payload
from ml.training.pretrain.value_semantics import coerce_int, coerce_str
from ml.training.pretrain.semantic_defaults import PINNED_SEMANTIC_ADAM_EPS
from ml.training.pretrain.grad_clip_semantics import PINNED_AGC_EPS


@dataclass
class PretrainRunConfig:
    data_path: str
    tokenizer_path: str = ""
    output_dir: str = ""
    resume_from_checkpoint: str = ""
    overwrite_output_dir: int = 0
    machine_recipe_json: str = ""

    _sophia_pretrain_profile: PretrainProfile | None = None
    _sophia_run_kind: str = "pretrain"
    _sophia_machine_signature: dict[str, object] | None = None
    _sophia_resolved_resume_checkpoint: str = ""
    _sophia_train_manifest_sha1: str = ""
    _sophia_val_manifest_sha1: str = ""
    _sophia_test_manifest_sha1: str = ""
    _sophia_decay_manifest_sha1: str = ""
    _sophia_requested_runtime_config: dict[str, object] | None = None
    _sophia_effective_runtime_config: dict[str, object] | None = None
    _sophia_recipe_override_diff: dict[str, object] | None = None

    device: str = "cuda:0"
    seed: int = 42
    safe_serialization: int = 1

    eval_data_path: str = ""
    test_data_path: str = ""
    decay_data_path: str = ""
    eval_interval: int = 1000
    eval_steps: int = 64
    external_eval_contamination_samples: int = 8
    external_eval_contamination_interval: int = 0

    log_interval: int = 100
    save_interval: int = 2000
    save_total_limit: int = 5
    save_weights_steps: int = 0
    save_weights_total_limit: int = 0
    ckpt_staging_dir: str = ""
    async_checkpoint: int = 1
    async_metrics: int = 1
    save_best: int = 1
    enable_checkpoints: int = 1

    total_tokens: int = 0
    target_tokens_per_update: int = 0
    learning_rate: float = 0.0
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = PINNED_SEMANTIC_ADAM_EPS
    max_steps: int = -1
    min_lr_ratio: float = 0.1
    warmup_steps: int = 800
    warmup_ratio: float = 0.0
    lr_schedule: str = "wsd"
    wsd_stable_ratio: float = 0.9
    wsd_decay_style: str = "cosine"
    layerwise_lr_decay: float = 1.0
    embedding_lr_scale: float = 1.0
    muon_ns_steps: int = 0
    muon_target_rms: float | None = None

    max_grad_norm: float = 1.0
    grad_clip_mode: str = "norm"
    agc_clip: float = 0.0
    agc_eps: float = PINNED_AGC_EPS
    agc_exclude_bias_and_norm: int = 1
    ema_decay: float = 0.999
    ema_update_interval: int = -1
    ema_eval: int = 1
    ema_use_for_export: int = 1
    ema_in_ckpt: int = 1

    dataloader_num_workers: int = -1
    dataloader_prefetch_factor: int = -1
    dataloader_persistent_workers: int = -1
    shard_preload: int = -1
    shard_preload_bytes: int = -1
    step_execution_backend: str = ""

    max_seq_len: int = 0
    seq_len: int = 0
    base_stage_seq_len: int = 0
    target_tokens_per_microbatch: int = 0
    batch_size: int = 0
    accumulation_steps: int = 0
    gradient_checkpointing: int = 0
    gradient_checkpointing_exclude_first: int = 0
    gradient_checkpointing_exclude_last: int = 0
    loss_chunk_size: int = 0
    auto_batch_size_max: int = 4096

    stability_guard_enabled: int = 0
    stability_guard_loss_window: int = 20
    stability_guard_loss_max: float = 0.0
    stability_guard_grad_norm_max: float = 0.0
    stability_guard_grad_window: int = 100
    stability_guard_grad_spike_limit: int = 0
    stability_guard_min_step: int = 0

    def __post_init__(self) -> None:
        self.data_path = coerce_str(self.data_path)
        self.tokenizer_path = coerce_str(self.tokenizer_path)
        self.output_dir = coerce_str(self.output_dir)
        self.resume_from_checkpoint = coerce_str(self.resume_from_checkpoint)
        self.machine_recipe_json = coerce_str(self.machine_recipe_json)
        self.device = coerce_str(self.device)
        self._sophia_run_kind = coerce_str(
            self._sophia_run_kind,
            default="pretrain",
        )
        self._sophia_resolved_resume_checkpoint = coerce_str(
            self._sophia_resolved_resume_checkpoint
        )
        self._sophia_machine_signature = (
            None
            if self._sophia_machine_signature is None
            else dict(self._sophia_machine_signature)
        )
        self._sophia_requested_runtime_config = (
            None
            if self._sophia_requested_runtime_config is None
            else dict(self._sophia_requested_runtime_config)
        )
        self._sophia_effective_runtime_config = (
            None
            if self._sophia_effective_runtime_config is None
            else dict(self._sophia_effective_runtime_config)
        )
        self._sophia_recipe_override_diff = (
            None
            if self._sophia_recipe_override_diff is None
            else dict(self._sophia_recipe_override_diff)
        )
        self.decay_data_path = coerce_str(self.decay_data_path)
        self.base_stage_seq_len = coerce_int(self.base_stage_seq_len)
        self.target_tokens_per_microbatch = coerce_int(
            self.target_tokens_per_microbatch
        )
        self.step_execution_backend = coerce_str(self.step_execution_backend)

    def to_payload(self) -> dict[str, object]:
        payload = dict(vars(self))
        profile = payload.get("_sophia_pretrain_profile")
        if isinstance(profile, PretrainProfile):
            payload["_sophia_pretrain_profile"] = pretrain_profile_to_payload(profile)
        return payload
def _field_default(field_info) -> object:
    if field_info.default is not MISSING:
        return field_info.default
    if field_info.default_factory is not MISSING:
        return field_info.default_factory()
    return ""


@lru_cache(maxsize=1)
def _run_config_field_specs() -> tuple[tuple[str, object], ...]:
    specs: list[tuple[str, object]] = []
    for field_info in fields(PretrainRunConfig):
        specs.append((str(field_info.name), _field_default(field_info)))
    return tuple(specs)


__all__ = [
    "PretrainRunConfig",
]
