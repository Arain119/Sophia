from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.tooling.scripts.data import resolve_pretrain_mix_plan as mod


REPO_ROOT = Path(__file__).resolve().parents[2]


def _policy() -> dict[str, object]:
    """The materialised policy the corpus was actually built from.

    Superseded drafts used to serve as fixtures here. They were deleted with
    the rest of the configs/data archive, and reaching for one again would only
    re-create the dependency. This file is a better fixture anyway: the dataset
    carries its own hash-bound copy, so the resolver is exercised against the
    policy that really produced the 20B corpus rather than a draft nobody ran.
    """

    return json.loads(
        (REPO_ROOT / "dataset" / "pretrain" / "lineage" / "mix_policy.json").read_text(
            encoding="utf-8"
        )
    )


def _profile(policy: dict[str, object], *, capacity: int) -> dict[str, object]:
    fractions = policy["source_token_fractions"]
    source_split: dict[str, int] = {}
    mix_bucket_split: dict[str, int] = {}
    for source, fraction in fractions.items():
        # A source at fraction zero still has to have supply: the resolver
        # refuses a profile that reports none, and the live policy carries
        # several sources it holds at zero.
        train_supply = max(1, int(float(capacity) * float(fraction)))
        source_split[f"{source}|train"] = train_supply
        source_split[f"{source}|val"] = 10
        source_split[f"{source}|test"] = 10
        mix_bucket_split[f"{source}|chars_lt_2k|train"] = train_supply
    return {
        "schema": "sophia_pretrain_corpus_token_profile_v1",
        "status": "complete",
        "tokenizer_bundle_sha1": "a" * 40,
        "tokens_by": {
            "source_split": source_split,
            "mix_bucket_split": mix_bucket_split,
        },
        "long_context_supply": {},
    }


def test_resolve_mix_plan_keeps_fixed_fractions_with_unique_supply() -> None:
    policy = _policy()
    policy["status"] = "admitted"
    policy["constraints"]["require_independent_modern_chinese_backbone"] = False
    policy["requested_unique_train_tokens"] = 1_000
    report = mod.resolve_mix_plan(
        profile=_profile(policy, capacity=2_000),
        policy=policy,
        profile_path="/profile.json",
        profile_sha256="b" * 64,
        policy_path="/mix_policy.json",
        policy_sha256="c" * 64,
    )

    assert report["status"] == "ready"
    assert report["resolved_unique_train_tokens"] == 1_000
    assert sum(report["source_train_quotas"].values()) == 1_000
    assert sum(report["composite_source_character_length_quotas"].values()) == 1_000
    assert not any(
        report["source_train_quotas"][source]
        > report["source_train_supply_tokens"][source]
        for source in report["source_train_quotas"]
    )
    assert report["profile_sha256"] == "b" * 64
    assert report["mix_policy_sha256"] == "c" * 64


def test_integer_quotas_use_largest_remainders() -> None:
    quotas = mod._integer_quotas(
        total=10,
        fractions={"largest_share": 0.51, "largest_remainder": 0.29, "other": 0.20},
    )

    assert quotas == {"largest_share": 5, "largest_remainder": 3, "other": 2}


def test_integer_quotas_recover_exact_measured_supply() -> None:
    supply = {"short": 3, "medium": 4, "long": 10}
    total = sum(supply.values())

    quotas = mod._integer_quotas(
        total=total,
        fractions={name: value / total for name, value in supply.items()},
    )

    assert quotas == supply


def test_maximum_unique_total_accepts_exact_float_ratio_boundary() -> None:
    supply = {"large": 12_144_107_034, "small": 7_146_741_314}
    total = sum(supply.values())
    fractions = {name: value / total for name, value in supply.items()}

    assert mod._maximum_unique_total(supply=supply, fractions=fractions) == total


def test_resolve_mix_plan_reports_limiting_source_without_upsampling() -> None:
    policy = _policy()
    policy["status"] = "admitted"
    policy["constraints"]["require_independent_modern_chinese_backbone"] = False
    policy["requested_unique_train_tokens"] = 2_000
    profile = _profile(policy, capacity=2_000)
    # Starve whichever source the current policy leans on hardest, so the
    # scarcity is unambiguous rather than a name copied from an old fixture.
    limiting = max(
        policy["source_token_fractions"],
        key=lambda name: float(policy["source_token_fractions"][name]),
    )
    profile["tokens_by"]["source_split"][f"{limiting}|train"] = 50
    profile["tokens_by"]["mix_bucket_split"][
        f"{limiting}|chars_lt_2k|train"
    ] = 50

    report = mod.resolve_mix_plan(profile=profile, policy=policy)

    assert report["status"] == "insufficient_unique_supply"
    assert limiting in report["limiting_sources"]
    # The exact ceiling follows from the limiting source's fraction in whatever
    # policy is current, so the assertion is the property, not the arithmetic:
    # the plan shrinks to what the scarcest source can supply without repeating
    # a document, and never hands that source more than it has.
    assert report["source_train_quotas"][limiting] == 50
    assert 0 < report["resolved_unique_train_tokens"] < 2_000


def test_v2_source_groups_validate_explicit_semantic_buckets() -> None:
    policy = {
        "schema": "sophia_pretrain_mix_policy_v1",
        "status": "admitted",
        "source_token_fractions": {
            "curated_zh": 0.3,
            "open_stem": 0.3,
            "open_code": 0.2,
            "long_books": 0.2,
        },
        "constraints": {
            "maximum_document_repetitions": 1,
            "chinese_token_fraction": 0.3,
            "math_min_fraction": 0.3,
            "code_min_fraction": 0.2,
            "long_form_source_min_fraction": 0.2,
            "source_groups": {
                "chinese": ["curated_zh"],
                "math_stem": ["open_stem"],
                "code": ["open_code"],
                "long_form": ["long_books"],
            },
        },
    }

    assert mod._validate_policy(policy) == policy["source_token_fractions"]
    policy["constraints"]["source_groups"]["chinese"] = ["open_code"]
    with pytest.raises(ValueError, match="source group drifted"):
        mod._validate_policy(policy)


def test_modern_chinese_backbone_group_must_meet_declared_minimum() -> None:
    policy = {
        "schema": "sophia_pretrain_mix_policy_v1",
        "status": "admitted",
        "source_token_fractions": {"modern_zh": 0.2, "open_en": 0.8},
        "constraints": {
            "maximum_document_repetitions": 1,
            "require_independent_modern_chinese_backbone": True,
            "modern_chinese_min_fraction": 0.25,
            "chinese_token_fraction": 0.2,
            "math_min_fraction": 0.0,
            "code_min_fraction": 0.0,
            "long_form_source_min_fraction": 0.0,
            "source_groups": {
                "modern_chinese": ["modern_zh"],
                "chinese": ["modern_zh"],
                "math_stem": ["open_en"],
                "code": ["open_en"],
                "long_form": ["open_en"],
            },
        },
    }

    with pytest.raises(ValueError, match="below minimum"):
        mod._validate_policy(policy)
    policy["constraints"]["modern_chinese_min_fraction"] = 0.2
    assert mod._validate_policy(policy) == policy["source_token_fractions"]


def test_policy_rejects_source_fraction_above_explicit_maximum() -> None:
    policy = _policy()
    policy["status"] = "admitted"
    policy["constraints"]["require_independent_modern_chinese_backbone"] = False
    source = next(iter(policy["source_token_fractions"]))
    actual = float(policy["source_token_fractions"][source])
    policy["constraints"]["source_max_fractions"] = {source: actual / 2.0}

    with pytest.raises(ValueError, match="exceeds policy maximum"):
        mod._validate_policy(policy)
