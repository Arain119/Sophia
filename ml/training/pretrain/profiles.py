from __future__ import annotations

from dataclasses import dataclass

from ml.core.common.mapping import object_mapping
from ml.core.spec import (
    ModelSpec,
    build_release_pretrain_schedule_spec,
)

PROFILE_ATTR = "_sophia_pretrain_profile"


@dataclass(frozen=True)
class PretrainCurriculumSpec:
    max_seq_len: int
    train_seq_len: int
    train_seq_stages: tuple[int, ...]
    stage_token_weights: tuple[int, ...]
    default_total_tokens: int


@dataclass(frozen=True)
class PretrainProfile:
    name: str
    kind: str
    model: ModelSpec
    curriculum: PretrainCurriculumSpec


RELEASE_MODEL = ModelSpec.default()
RELEASE_SCHEDULE = build_release_pretrain_schedule_spec(model=RELEASE_MODEL)
def _resample_layerwise_int_schedule(
    values: tuple[int, ...] | list[int],
    *,
    target_len: int,
) -> tuple[int, ...]:
    target = int(target_len)
    if target <= 0:
        raise ValueError(f"target_len must be > 0 (got {target_len!r})")
    source = tuple(int(value) for value in values)
    if not source:
        return tuple(0 for _ in range(target))
    if len(source) == target:
        return source
    if target == 1:
        return (int(source[-1]),)
    source_last = len(source) - 1
    target_last = target - 1
    out: list[int] = []
    for idx in range(target):
        source_idx = int(round((float(idx) * float(source_last)) / float(target_last)))
        out.append(int(source[source_idx]))
    return tuple(out)


RELEASE_PROFILE = PretrainProfile(
    name="release",
    kind="release",
    model=RELEASE_MODEL,
    curriculum=PretrainCurriculumSpec(
        max_seq_len=int(RELEASE_SCHEDULE.max_seq_len),
        train_seq_len=int(RELEASE_SCHEDULE.train_seq_len),
        train_seq_stages=tuple(int(x) for x in RELEASE_SCHEDULE.train_seq_stages),
        stage_token_weights=tuple(int(x) for x in RELEASE_SCHEDULE.stage_token_weights),
        default_total_tokens=0,
    ),
)

VALIDATION_PROFILE = PretrainProfile(
    name="validation",
    kind="validation",
    model=RELEASE_MODEL.with_overrides(
        dim=384,
        n_layers=6,
        n_heads=6,
        head_dim=64,
        num_key_value_heads=2,
        rope_head_dim=32,
        rope_theta=500_000.0,
        original_seq_len=0,
        rope_factor=16.0,
        beta_fast=32,
        beta_slow=1,
        norm_eps=1e-6,
        ffn_hidden=768,
    ),
    curriculum=PretrainCurriculumSpec(
        max_seq_len=int(RELEASE_MODEL.max_seq_len),
        train_seq_len=int(RELEASE_SCHEDULE.train_seq_len),
        train_seq_stages=tuple(int(x) for x in RELEASE_SCHEDULE.train_seq_stages),
        stage_token_weights=tuple(int(x) for x in RELEASE_SCHEDULE.stage_token_weights),
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
        "curriculum": {
            "max_seq_len": int(profile.curriculum.max_seq_len),
            "train_seq_len": int(profile.curriculum.train_seq_len),
            "train_seq_stages": [int(x) for x in profile.curriculum.train_seq_stages],
            "stage_token_weights": [int(x) for x in profile.curriculum.stage_token_weights],
            "default_total_tokens": int(profile.curriculum.default_total_tokens),
        },
    }


__all__ = [
    "PROFILE_ATTR",
    "RELEASE_PROFILE",
    "PretrainCurriculumSpec",
    "PretrainProfile",
    "VALIDATION_PROFILE",
    "_resample_layerwise_int_schedule",
    "is_release_pretrain_profile",
    "pretrain_profile_to_payload",
    "resolve_pretrain_profile",
]
