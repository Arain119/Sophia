from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

RELEASE_QK_CLIP_THRESHOLD = 100.0


DEFAULT_PRETRAIN_MACHINE_RECIPE = {
    "batch_size": 1,
    "accumulation_steps": 320,
    "gradient_checkpointing": 0,
    "gradient_checkpointing_exclude_first": 0,
    "gradient_checkpointing_exclude_last": 0,
    "loss_chunk_size": 0,
}
DEFAULT_PRETRAIN_MACHINE_RUNTIME = {
    "step_execution_backend": "sm120_graph",
    "dataloader_num_workers": 0,
    "dataloader_prefetch_factor": 0,
    "dataloader_persistent_workers": 0,
    "shard_preload": 0,
    "shard_preload_bytes": 0,
}


@dataclass(frozen=True)
class ReleasePretrainDefaults:
    target_tokens_per_update: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    adam_eps: float
    optimizer_kind: str
    muon_ns_steps: int
    warmup_steps: int
    min_lr_ratio: float
    lr_schedule: str


@dataclass(frozen=True)
class PretrainReleaseSemantics:
    profile: dict[str, object] = field(default_factory=dict)
    target_tokens_per_update: int = 0
    learning_rate: float = 0.0
    weight_decay: float = 0.0
    beta1: float = 0.0
    beta2: float = 0.0
    adam_eps: float = 0.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.0
    optimizer_kind: str = "torch_muon_hybrid"
    muon_ns_steps: int = 0
    lr_schedule: str = ""
    qk_clip_enabled: bool = True
    qk_clip_threshold: float = RELEASE_QK_CLIP_THRESHOLD

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_run_config(cls, *, args: object) -> PretrainReleaseSemantics:
        from ml.training.pretrain.profiles import (
            pretrain_profile_to_payload,
            resolve_pretrain_profile,
        )

        profile = resolve_pretrain_profile(args)
        return cls(
            profile=pretrain_profile_to_payload(profile),
            target_tokens_per_update=int(
                getattr(args, "target_tokens_per_update", 0) or 0
            ),
            learning_rate=float(getattr(args, "learning_rate", 0.0) or 0.0),
            weight_decay=float(getattr(args, "weight_decay", 0.0) or 0.0),
            beta1=float(getattr(args, "beta1", 0.0) or 0.0),
            beta2=float(getattr(args, "beta2", 0.0) or 0.0),
            adam_eps=float(getattr(args, "adam_eps", 0.0) or 0.0),
            warmup_steps=int(getattr(args, "warmup_steps", 0) or 0),
            warmup_ratio=float(getattr(args, "warmup_ratio", 0.0) or 0.0),
            min_lr_ratio=float(getattr(args, "min_lr_ratio", 0.0) or 0.0),
            optimizer_kind=str(
                getattr(args, "optimizer_kind", "torch_muon_hybrid")
                or "torch_muon_hybrid"
            ),
            muon_ns_steps=int(getattr(args, "muon_ns_steps", 0) or 0),
            lr_schedule=str(getattr(args, "lr_schedule", "") or ""),
            qk_clip_enabled=True,
            qk_clip_threshold=RELEASE_QK_CLIP_THRESHOLD,
        )


RELEASE_PRETRAIN_DEFAULTS = ReleasePretrainDefaults(
    # Moonlight Table 2, 822M non-embedding parameters / 20.76B tokens:
    # 160 examples at 8K equals 320 Sophia examples at 4K.
    target_tokens_per_update=1_310_720,
    learning_rate=8.825e-4,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    adam_eps=1e-8,
    optimizer_kind="torch_muon_hybrid",
    muon_ns_steps=5,
    # K3 uses a 1% linear warmup: ceil(15,259 updates * 0.01) = 153.
    warmup_steps=153,
    min_lr_ratio=0.1,
    lr_schedule="cosine",
)
RELEASE_PRETRAIN_TOTAL_TOKENS = 20_000_000_000
RELEASE_PRETRAIN_STEPS = (
    RELEASE_PRETRAIN_TOTAL_TOKENS
    + RELEASE_PRETRAIN_DEFAULTS.target_tokens_per_update
    - 1
) // RELEASE_PRETRAIN_DEFAULTS.target_tokens_per_update
RELEASE_PRETRAIN_STABILITY_STEPS = RELEASE_PRETRAIN_DEFAULTS.warmup_steps + 1

RELEASE_PRETRAIN_DTYPE = torch.bfloat16
RELEASE_PRETRAIN_PRECISION = "bf16"
RELEASE_PRETRAIN_PRECISION_STACK = "bf16_fla_kda_sdpa_mla_liger"


def release_pretrain_runtime_metadata() -> dict[str, str]:
    return {
        "precision": RELEASE_PRETRAIN_PRECISION,
        "precision_stack": RELEASE_PRETRAIN_PRECISION_STACK,
        "mla_qk_clip": "per_head_weight_clip",
        "mla_qk_clip_enabled": "true",
        "mla_qk_clip_threshold": str(RELEASE_QK_CLIP_THRESHOLD),
    }


__all__ = [
    "DEFAULT_PRETRAIN_MACHINE_RECIPE",
    "DEFAULT_PRETRAIN_MACHINE_RUNTIME",
    "RELEASE_PRETRAIN_DEFAULTS",
    "RELEASE_PRETRAIN_DTYPE",
    "RELEASE_PRETRAIN_PRECISION",
    "RELEASE_PRETRAIN_PRECISION_STACK",
    "RELEASE_PRETRAIN_STABILITY_STEPS",
    "RELEASE_PRETRAIN_STEPS",
    "RELEASE_PRETRAIN_TOTAL_TOKENS",
    "RELEASE_QK_CLIP_THRESHOLD",
    "PretrainReleaseSemantics",
    "release_pretrain_runtime_metadata",
]
