from __future__ import annotations

from dataclasses import dataclass

from ml.core.common.mapping import object_mapping
from ml.core.spec import (
    ModelSpec,
    build_release_pretrain_schedule_spec,
)
from ml.training.pretrain.release_config import RELEASE_PRETRAIN_TOTAL_TOKENS

PROFILE_ATTR = "_sophia_pretrain_profile"


@dataclass(frozen=True)
class PretrainTrainingSpec:
    max_seq_len: int
    train_seq_len: int
    default_total_tokens: int


@dataclass(frozen=True)
class PretrainProfile:
    name: str
    kind: str
    model: ModelSpec
    training: PretrainTrainingSpec


RELEASE_MODEL = ModelSpec.default()
RELEASE_SCHEDULE = build_release_pretrain_schedule_spec(model=RELEASE_MODEL)


RELEASE_PROFILE = PretrainProfile(
    name="release",
    kind="release",
    model=RELEASE_MODEL,
    training=PretrainTrainingSpec(
        max_seq_len=int(RELEASE_SCHEDULE.max_seq_len),
        train_seq_len=int(RELEASE_SCHEDULE.train_seq_len),
        default_total_tokens=RELEASE_PRETRAIN_TOTAL_TOKENS,
    ),
)

VALIDATION_PROFILE = PretrainProfile(
    name="validation",
    kind="validation",
    model=RELEASE_MODEL.with_overrides(
        dim=384,
        n_layers=4,
        num_heads=6,
        head_dim=64,
        kda_decay_rank=64,
        kda_output_gate_rank=64,
        mla_q_rank=96,
        mla_kv_rank=64,
        norm_eps=1e-5,
        ffn_hidden=768,
    ),
    training=PretrainTrainingSpec(
        max_seq_len=int(RELEASE_MODEL.max_seq_len),
        train_seq_len=int(RELEASE_SCHEDULE.train_seq_len),
        default_total_tokens=65_536,
    ),
)


def is_release_pretrain_profile(profile: PretrainProfile) -> bool:
    return str(profile.kind).strip().lower() == "release"


def resolve_pretrain_profile(args: object | None = None) -> PretrainProfile:
    if args is None:
        return RELEASE_PROFILE
    profile = object_mapping(args).get(PROFILE_ATTR)
    if isinstance(profile, PretrainProfile):
        return profile
    return RELEASE_PROFILE


def pretrain_profile_to_payload(profile: PretrainProfile) -> dict[str, object]:
    return {
        "name": str(profile.name),
        "kind": str(profile.kind),
        "model": dict(profile.model.to_dict()),
        "training": {
            "max_seq_len": int(profile.training.max_seq_len),
            "train_seq_len": int(profile.training.train_seq_len),
            "default_total_tokens": int(profile.training.default_total_tokens),
        },
    }


__all__ = [
    "PROFILE_ATTR",
    "RELEASE_PROFILE",
    "PretrainTrainingSpec",
    "PretrainProfile",
    "VALIDATION_PROFILE",
    "is_release_pretrain_profile",
    "pretrain_profile_to_payload",
    "resolve_pretrain_profile",
]
