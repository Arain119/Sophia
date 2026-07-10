from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

CANONICAL_PRETRAIN_MACHINE_RECIPE = (
    "configs/pretrain/machine_recipes/rtx5090_bf16_seq4096_bs3_acc22.json"
)
CANONICAL_PRETRAIN_MACHINE_RECIPE_NAME = (
    "release_pretrain_rtx5090_bf16_sdpa_flash_liger_inductor"
)
CANONICAL_PRETRAIN_MACHINE_RECIPE_PAYLOAD = {
    "batch_size": 3,
    "accumulation_steps": 22,
    # RTX 5090 32GB measured best: bs3/seq4096/acc22, no recomputation,
    # ~270k tokens/update, ~27.3k tok/s, ~28.0GB reserved.
    "gradient_checkpointing": 0,
    "gradient_checkpointing_exclude_first": 0,
    "gradient_checkpointing_exclude_last": 0,
    "loss_chunk_size": 0,
}
CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD = {
    "step_execution_backend": "inductor",
    "dataloader_num_workers": 0,
    "dataloader_prefetch_factor": 0,
    "dataloader_persistent_workers": 0,
    "shard_preload": 1,
    "shard_preload_bytes": 4_194_304,
}


def canonical_pretrain_machine_recipe_path() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return str(repo_root / CANONICAL_PRETRAIN_MACHINE_RECIPE)


@dataclass(frozen=True)
class ReleasePretrainDefaults:
    target_tokens_per_update: int
    learning_rate: float
    adam_eps: float
    embedding_lr_scale: float
    ema_decay: float


@dataclass(frozen=True)
class ReleasePretrainPrecisionPolicy:
    precision: str
    precision_stack: str
    runtime_validation_status: str
    runtime_validation_requirements: tuple[str, ...]

    def gate_payload(self) -> dict[str, object]:
        return {
            "release_pretrain_precision": str(self.precision),
            "release_pretrain_precision_stack": str(self.precision_stack),
        }

    def recommendation_payload(self) -> dict[str, object]:
        return self.gate_payload()

    def runtime_metadata(self) -> dict[str, str]:
        return {
            "precision_stack": str(self.precision_stack),
            **{
                str(key): str(value)
                for key, value in self.gate_payload().items()
            },
            "runtime_validation_status": str(self.runtime_validation_status),
        }

    def resolve_dtype(self) -> torch.dtype:
        if str(self.precision) != "bf16":
            raise RuntimeError(
                f"unsupported release pretrain precision={self.precision!r}"
            )
        return torch.bfloat16


@dataclass(frozen=True)
class MachineRuntimeProfile:
    dataloader_num_workers: int
    dataloader_prefetch_factor: int
    dataloader_persistent_workers: int
    shard_preload: int
    shard_preload_bytes: int
    allow_machine_backend_selection: bool


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
    wsd_stable_ratio: float = 0.0
    muon_ns_steps: int = 0
    muon_target_rms: float | None = None
    embedding_lr_scale: float = 1.0
    ema_decay: float = 0.0
    max_grad_norm: float = 0.0
    grad_clip_mode: str = ""
    agc_clip: float = 0.0
    agc_eps: float = 0.0
    agc_exclude_bias_and_norm: int = 0
    lr_schedule: str = ""
    backend: str = ""

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_run_config(cls, *, args: object) -> PretrainReleaseSemantics:
        from ml.tasks.pretrain.run_spec_builder import (
            RELEASE_STEP_EXECUTION_POLICY,
        )
        from ml.training.pretrain.profiles import (
            pretrain_profile_to_payload,
            resolve_pretrain_profile,
        )

        profile = resolve_pretrain_profile(args)
        raw_muon_target_rms = getattr(args, "muon_target_rms", None)
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
            wsd_stable_ratio=float(getattr(args, "wsd_stable_ratio", 0.0) or 0.0),
            muon_ns_steps=int(getattr(args, "muon_ns_steps", 0) or 0),
            muon_target_rms=(
                None
                if raw_muon_target_rms is None
                else float(raw_muon_target_rms)
            ),
            embedding_lr_scale=float(
                getattr(args, "embedding_lr_scale", 1.0) or 1.0
            ),
            ema_decay=float(getattr(args, "ema_decay", 0.0) or 0.0),
            max_grad_norm=float(getattr(args, "max_grad_norm", 0.0) or 0.0),
            grad_clip_mode=str(getattr(args, "grad_clip_mode", "") or ""),
            agc_clip=float(getattr(args, "agc_clip", 0.0) or 0.0),
            agc_eps=float(getattr(args, "agc_eps", 0.0) or 0.0),
            agc_exclude_bias_and_norm=int(
                getattr(args, "agc_exclude_bias_and_norm", 0) or 0
            ),
            lr_schedule=str(getattr(args, "lr_schedule", "") or ""),
            backend=str(
                getattr(args, "step_execution_backend", "")
                or RELEASE_STEP_EXECUTION_POLICY.backend
            ),
        )


RELEASE_PRETRAIN_DEFAULTS = ReleasePretrainDefaults(
    target_tokens_per_update=270_336,
    # BF16 release baseline. Any change must re-pass the >=800-step stability
    # probe gate on the target GPU before a long run.
    learning_rate=1e-4,
    adam_eps=1e-6,
    embedding_lr_scale=0.25,
    ema_decay=0.0,
)

RELEASE_PRETRAIN_PRECISION_POLICY = ReleasePretrainPrecisionPolicy(
    precision="bf16",
    precision_stack="bf16_sdpa_flash_liger",
    runtime_validation_status="bf16_canary_pending_full_run",
    runtime_validation_requirements=(
        "runtime_context",
        "release_warmup",
    ),
)

CUDA_RELEASE_RUNTIME_PROFILE = MachineRuntimeProfile(
    dataloader_num_workers=0,
    dataloader_prefetch_factor=0,
    dataloader_persistent_workers=0,
    shard_preload=1,
    shard_preload_bytes=4_194_304,
    allow_machine_backend_selection=False,
)


def release_precision_status_payload() -> dict[str, object]:
    gate = RELEASE_PRETRAIN_PRECISION_POLICY.gate_payload()
    return {
        **gate,
        "selected_runtime_validation_status": str(
            RELEASE_PRETRAIN_PRECISION_POLICY.runtime_validation_status
        ),
        "selected_runtime_validation_requirements": list(
            RELEASE_PRETRAIN_PRECISION_POLICY.runtime_validation_requirements
        ),
        "project_precision_stack_validated": True,
        "release_precision_safe": False,
        "reason": (
            "The configured release stack is locked to BF16, PyTorch SDPA Flash "
            "attention, fused projections, and Liger fused linear CE on the signed "
            "RTX 5090 machine recipe. Release-safe status remains blocked by "
            "pending target-GPU runtime evidence and the lack of a completed full "
            "warmup run at the pinned release LR."
        ),
    }


def release_pretrain_runtime_metadata() -> dict[str, str]:
    return RELEASE_PRETRAIN_PRECISION_POLICY.runtime_metadata()


def resolve_pinned_precision_dtype() -> torch.dtype:
    return RELEASE_PRETRAIN_PRECISION_POLICY.resolve_dtype()


__all__ = [
    "CANONICAL_PRETRAIN_MACHINE_RECIPE",
    "CANONICAL_PRETRAIN_MACHINE_RECIPE_NAME",
    "CANONICAL_PRETRAIN_MACHINE_RECIPE_PAYLOAD",
    "CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD",
    "RELEASE_PRETRAIN_DEFAULTS",
    "RELEASE_PRETRAIN_PRECISION_POLICY",
    "CUDA_RELEASE_RUNTIME_PROFILE",
    "MachineRuntimeProfile",
    "PretrainReleaseSemantics",
    "release_pretrain_runtime_metadata",
    "canonical_pretrain_machine_recipe_path",
    "release_precision_status_payload",
    "resolve_pinned_precision_dtype",
]
