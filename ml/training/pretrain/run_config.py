from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from functools import lru_cache

from ml.training.pretrain.profiles import (
    PretrainProfile,
    pretrain_profile_to_payload,
)
from ml.training.pretrain.value_semantics import coerce_str
from ml.training.pretrain.release_config import (
    RELEASE_PRETRAIN_DEFAULTS,
    RELEASE_PRETRAIN_TOTAL_TOKENS,
)


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
    _sophia_machine_recipe_path: str = ""
    _sophia_machine_recipe_sha256: str = ""
    _sophia_machine_recipe_migration_from_sha256: str = ""
    _sophia_resolved_resume_checkpoint: str = ""
    _sophia_train_manifest_sha1: str = ""
    _sophia_val_manifest_sha1: str = ""
    _sophia_test_manifest_sha1: str = ""
    _sophia_protocol_path: str = ""
    _sophia_protocol_sha256: str = ""
    _sophia_allow_protocol_path_relocation: int = 0
    _sophia_model_spec_sha256: str = ""
    _sophia_model_parameter_count: int = 0
    _sophia_tokenizer_json_sha1: str = ""
    _sophia_dataset_marker_sha256: str = ""
    _sophia_data_admission_sha256: str = ""
    _sophia_lr_schedule_horizon_steps: int = 0

    device: str = "cuda:0"
    seed: int = 42
    safe_serialization: int = 1

    eval_data_path: str = ""
    test_data_path: str = ""
    eval_interval: int = 1000
    eval_steps: int = 64
    external_eval_contamination_samples: int = 8
    external_eval_contamination_interval: int = 0

    log_interval: int = 10
    save_interval: int = 100
    save_total_limit: int = 6
    ckpt_staging_dir: str = ""
    async_checkpoint: int = 1
    async_metrics: int = 1
    save_best: int = 0
    enable_checkpoints: int = 1

    total_tokens: int = RELEASE_PRETRAIN_TOTAL_TOKENS
    target_tokens_per_update: int = RELEASE_PRETRAIN_DEFAULTS.target_tokens_per_update
    learning_rate: float = RELEASE_PRETRAIN_DEFAULTS.learning_rate
    weight_decay: float = RELEASE_PRETRAIN_DEFAULTS.weight_decay
    beta1: float = RELEASE_PRETRAIN_DEFAULTS.beta1
    beta2: float = RELEASE_PRETRAIN_DEFAULTS.beta2
    adam_eps: float = RELEASE_PRETRAIN_DEFAULTS.adam_eps
    max_steps: int = -1
    min_lr_ratio: float = RELEASE_PRETRAIN_DEFAULTS.min_lr_ratio
    warmup_steps: int = RELEASE_PRETRAIN_DEFAULTS.warmup_steps
    warmup_ratio: float = 0.0
    lr_schedule: str = RELEASE_PRETRAIN_DEFAULTS.lr_schedule
    optimizer_kind: str = RELEASE_PRETRAIN_DEFAULTS.optimizer_kind
    muon_ns_steps: int = RELEASE_PRETRAIN_DEFAULTS.muon_ns_steps

    dataloader_num_workers: int = -1
    dataloader_prefetch_factor: int = -1
    dataloader_persistent_workers: int = -1
    shard_preload: int = -1
    shard_preload_bytes: int = -1
    step_execution_backend: str = ""

    max_seq_len: int = 0
    seq_len: int = 0
    batch_size: int = 0
    accumulation_steps: int = 0
    gradient_checkpointing: int = 0
    gradient_checkpointing_exclude_first: int = 0
    gradient_checkpointing_exclude_last: int = 0
    loss_chunk_size: int = 0

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
        self._sophia_machine_recipe_path = coerce_str(
            self._sophia_machine_recipe_path
        )
        self._sophia_machine_recipe_sha256 = coerce_str(
            self._sophia_machine_recipe_sha256
        )
        self._sophia_resolved_resume_checkpoint = coerce_str(
            self._sophia_resolved_resume_checkpoint
        )
        self._sophia_machine_signature = (
            None
            if self._sophia_machine_signature is None
            else dict(self._sophia_machine_signature)
        )
        self.step_execution_backend = coerce_str(self.step_execution_backend)
        self.optimizer_kind = coerce_str(self.optimizer_kind)

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
