from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

from ml.core.common.io import write_json_atomic
from ml.tooling.scripts.data.profile_pretrain_corpus import _load_json


REPORT_SCHEMA = "sophia_resolved_pretrain_mix_plan_v1"
GROUP_CONSTRAINTS = {
    "chinese": ("chinese_token_fraction", "exact"),
    "math_stem": ("math_min_fraction", "minimum"),
    "code": ("code_min_fraction", "minimum"),
    "long_form": ("long_form_source_min_fraction", "minimum"),
}


def _integer_quotas(*, total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw_quotas = {
        name: float(total) * float(fraction) for name, fraction in fractions.items()
    }
    quotas = {name: int(math.floor(value)) for name, value in raw_quotas.items()}
    drift = int(total) - sum(quotas.values())
    if drift < 0 or drift > len(fractions):
        raise ValueError(f"invalid integer quota rounding drift: {drift}")
    order = sorted(
        fractions,
        key=lambda name: (
            -(raw_quotas[name] - quotas[name]),
            -float(fractions[name]),
            str(name),
        ),
    )
    for index in range(drift):
        quotas[order[index]] += 1
    return quotas


def _maximum_unique_total(
    *, supply: dict[str, int], fractions: dict[str, float]
) -> int:
    continuous_floor = min(
        int(math.floor(float(supply[source]) / float(fraction)))
        for source, fraction in fractions.items()
    )

    def is_feasible(total: int) -> bool:
        quotas = _integer_quotas(total=total, fractions=fractions)
        return all(quotas[source] <= supply[source] for source in fractions)

    maximum = continuous_floor
    while maximum > 0 and not is_feasible(maximum):
        maximum -= 1
    aggregate_supply = sum(supply.values())
    while maximum < aggregate_supply and is_feasible(maximum + 1):
        maximum += 1
    return maximum


def _validate_policy(policy: dict[str, Any]) -> dict[str, float]:
    if str(policy.get("schema")) != "sophia_pretrain_mix_policy_v1":
        raise ValueError("unsupported pretrain mix policy")
    if str(policy.get("status") or "") not in {
        "admitted",
        "admitted_pending_supply_resolution",
    }:
        raise ValueError("pretrain mix policy is not admitted")
    raw = policy.get("source_token_fractions")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("mix policy has no source fractions")
    fractions = {str(name): float(value) for name, value in raw.items()}
    if any(value <= 0.0 for value in fractions.values()):
        raise ValueError("all source fractions must be positive")
    total = sum(fractions.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"source fractions must sum to 1.0, got {total}")
    constraints = policy.get("constraints")
    if not isinstance(constraints, dict):
        raise ValueError("mix policy has no constraints")
    if int(constraints.get("maximum_document_repetitions") or 0) != 1:
        raise ValueError("random-initialization corpus policy must remain unique-only")
    source_max_fractions = constraints.get("source_max_fractions") or {}
    if not isinstance(source_max_fractions, dict):
        raise ValueError("source_max_fractions must be an object")
    for source, raw_maximum in source_max_fractions.items():
        normalized_source = str(source)
        if normalized_source not in fractions:
            raise ValueError(
                f"source_max_fractions has unknown source: {normalized_source}"
            )
        maximum = float(raw_maximum)
        if not 0.0 < maximum <= 1.0:
            raise ValueError(
                f"source maximum fraction must be in (0, 1]: {normalized_source}"
            )
        if fractions[normalized_source] > maximum:
            raise ValueError(
                "source fraction exceeds policy maximum: "
                f"source={normalized_source} actual={fractions[normalized_source]} "
                f"maximum={maximum}"
            )
    source_groups = constraints.get("source_groups")
    if source_groups is None:
        chinese = sum(
            fraction
            for name, fraction in fractions.items()
            if name.startswith("opencsg_") or "wikipedia_zh" in name
        )
        if not math.isclose(
            chinese,
            float(constraints.get("chinese_token_fraction") or 0.0),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Chinese source fractions drifted from the policy constraint"
            )
        return fractions
    if not isinstance(source_groups, dict):
        raise ValueError("mix policy source_groups must be an object")
    if bool(constraints.get("require_independent_modern_chinese_backbone")):
        modern_members = source_groups.get("modern_chinese")
        if not isinstance(modern_members, list) or not modern_members:
            raise ValueError(
                "pretrain mix policy requires an independent modern_chinese source group"
            )
        normalized_modern = [str(member) for member in modern_members]
        if len(normalized_modern) != len(set(normalized_modern)):
            raise ValueError("modern_chinese source group has duplicates")
        unknown_modern = sorted(set(normalized_modern) - set(fractions))
        if unknown_modern:
            raise ValueError(
                "modern_chinese source group has unknown sources: "
                f"{unknown_modern}"
            )
        modern_minimum = float(
            constraints.get("modern_chinese_min_fraction") or 0.0
        )
        if modern_minimum <= 0.0:
            raise ValueError("modern_chinese_min_fraction must be positive")
        modern_fraction = sum(fractions[source] for source in normalized_modern)
        if modern_fraction < modern_minimum:
            raise ValueError(
                "modern_chinese source group is below minimum: "
                f"actual={modern_fraction} minimum={modern_minimum}"
            )
    for group, (constraint_name, mode) in GROUP_CONSTRAINTS.items():
        members = source_groups.get(group)
        if not isinstance(members, list) or not members:
            raise ValueError(f"mix policy source group is missing: {group}")
        normalized = [str(member) for member in members]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"mix policy source group has duplicates: {group}")
        unknown = sorted(set(normalized) - set(fractions))
        if unknown:
            raise ValueError(
                f"mix policy source group has unknown sources: {group}={unknown}"
            )
        actual = sum(fractions[source] for source in normalized)
        expected = float(constraints.get(constraint_name) or 0.0)
        if mode == "exact" and not math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"mix policy source group drifted: {group}={actual} "
                f"expected={expected}"
            )
        if mode == "minimum" and actual < expected:
            raise ValueError(
                f"mix policy source group is below minimum: {group}={actual} "
                f"minimum={expected}"
            )
    return fractions


def resolve_mix_plan(
    *,
    profile: dict[str, Any],
    policy: dict[str, Any],
    profile_path: str = "",
    profile_sha256: str = "",
    policy_path: str = "",
    policy_sha256: str = "",
) -> dict[str, Any]:
    if str(profile.get("schema")) != "sophia_pretrain_corpus_token_profile_v1":
        raise ValueError("unsupported pretrain corpus token profile")
    if str(profile.get("status")) != "complete":
        raise ValueError("mix resolution requires a complete token profile")
    fractions = _validate_policy(policy)
    source_split = profile.get("tokens_by", {}).get("source_split", {})
    if not isinstance(source_split, dict):
        raise ValueError("token profile is missing source_split totals")
    supply = {
        source: int(source_split.get(f"{source}|train") or 0)
        for source in fractions
    }
    missing = sorted(source for source, value in supply.items() if value <= 0)
    if missing:
        raise ValueError(f"token profile has no train supply for sources: {missing}")
    limiting_capacity = {
        source: int(math.floor(float(supply[source]) / float(fraction)))
        for source, fraction in fractions.items()
    }
    maximum_unique_tokens = _maximum_unique_total(
        supply=supply,
        fractions=fractions,
    )
    limiting_sources = sorted(
        source
        for source, capacity in limiting_capacity.items()
        if capacity <= maximum_unique_tokens
    )
    requested = int(policy.get("requested_unique_train_tokens") or 0)
    if requested <= 0:
        raise ValueError("requested_unique_train_tokens must be positive")
    resolved_total = min(requested, maximum_unique_tokens)
    source_quotas = _integer_quotas(total=resolved_total, fractions=fractions)
    overdrawn = {
        source: source_quotas[source] - supply[source]
        for source in fractions
        if source_quotas[source] > supply[source]
    }
    if overdrawn:
        raise RuntimeError(f"resolved source quotas exceed unique supply: {overdrawn}")

    composite_supply = profile.get("tokens_by", {}).get("mix_bucket_split", {})
    if not isinstance(composite_supply, dict):
        raise ValueError("token profile is missing mix_bucket_split totals")
    composite_quotas: dict[str, int] = {}
    for source, source_quota in source_quotas.items():
        buckets = {
            str(key).removesuffix("|train"): int(value)
            for key, value in composite_supply.items()
            if str(key).startswith(f"{source}|") and str(key).endswith("|train")
        }
        bucket_total = sum(buckets.values())
        if bucket_total != supply[source]:
            raise ValueError(
                f"composite length supply does not sum to source supply: {source}"
            )
        bucket_fractions = {
            name: float(value) / float(bucket_total) for name, value in buckets.items()
        }
        resolved = _integer_quotas(total=source_quota, fractions=bucket_fractions)
        for name, quota in resolved.items():
            if quota > buckets[name]:
                raise RuntimeError(f"composite quota exceeds unique supply: {name}")
            composite_quotas[name] = quota

    evaluation_supply = {
        split: {
            source: int(source_split.get(f"{source}|{split}") or 0)
            for source in fractions
        }
        for split in ("val", "test")
    }
    status = "ready" if resolved_total == requested else "insufficient_unique_supply"
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "requested_unique_train_tokens": requested,
        "resolved_unique_train_tokens": resolved_total,
        "maximum_unique_train_tokens_under_fixed_mix": maximum_unique_tokens,
        "limiting_sources": limiting_sources,
        "source_fractions": fractions,
        "source_train_supply_tokens": supply,
        "source_train_quotas": source_quotas,
        "source_unused_unique_tokens": {
            source: supply[source] - source_quotas[source] for source in fractions
        },
        "composite_source_character_length_quotas": dict(
            sorted(composite_quotas.items())
        ),
        "evaluation_unique_supply_tokens": evaluation_supply,
        "long_context_supply": profile.get("long_context_supply", {}),
        "tokenizer_bundle_sha1": str(profile.get("tokenizer_bundle_sha1") or ""),
        "profile_path": str(profile_path),
        "profile_sha256": str(profile_sha256),
        "mix_policy_path": str(policy_path),
        "mix_policy_sha256": str(policy_sha256),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve a unique-only pretrain mix from measured source supply."
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--policy",
        default="configs/data/pretrain_mix_policy.json",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile_path = Path(args.profile).expanduser().resolve()
    policy_path = Path(args.policy).expanduser().resolve()
    report = resolve_mix_plan(
        profile=_load_json(profile_path),
        policy=_load_json(policy_path),
        profile_path=str(profile_path),
        profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        policy_path=str(policy_path),
        policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    )
    write_json_atomic(
        Path(args.output).expanduser().resolve(),
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    print(
        f"[DONE] status={report['status']} "
        f"tokens={report['resolved_unique_train_tokens']:,} "
        f"output={Path(args.output).resolve()}",
        flush=True,
    )
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
