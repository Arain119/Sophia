from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from ml.core.common.io import write_json_atomic
from ml.tooling.scripts.data.profile_pretrain_corpus import _load_json
from ml.tooling.scripts.data.resolve_pretrain_mix_plan import (
    _integer_quotas,
    _validate_policy,
)


_DRAFT_GENERATIONS = {
    "draft_pending_new_source_cleaning_and_token_profile": "v6",
    "draft_pending_v7_token_profile": "v7",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_groups(draft: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_groups = draft.get("target_source_groups")
    targets = draft.get("target_fractions")
    if not isinstance(raw_groups, dict) or not isinstance(targets, dict):
        raise ValueError("draft policy requires target fractions and source groups")
    if set(raw_groups) != set(targets):
        raise ValueError("target source groups must match target fractions")
    groups: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for name, raw in raw_groups.items():
        if not isinstance(raw, dict):
            raise ValueError(f"target source group must be an object: {name}")
        members = [str(value) for value in raw.get("members") or []]
        allocation = str(raw.get("allocation") or "")
        if (
            not members
            or len(members) != len(set(members))
            or allocation not in {"proportional_to_supply", "priority_fill"}
        ):
            raise ValueError(f"invalid target source group: {name}")
        overlap = seen.intersection(members)
        if overlap:
            raise ValueError(
                f"sources belong to multiple target groups: {sorted(overlap)}"
            )
        seen.update(members)
        groups[str(name)] = {
            "members": members,
            "allocation": allocation,
            "shortfall_reallocation_target": str(
                raw.get("shortfall_reallocation_target") or ""
            ),
        }
    return groups


def _set_group_total(
    fractions: dict[str, float], members: list[str], target: float
) -> None:
    drift = float(target) - sum(float(fractions[source]) for source in members)
    fractions[members[-1]] = float(fractions[members[-1]]) + drift


def materialize_policy(
    *,
    draft: dict[str, Any],
    profile: dict[str, Any],
    draft_path: str = "",
    draft_sha256: str = "",
    profile_path: str = "",
    profile_sha256: str = "",
) -> dict[str, Any]:
    if str(draft.get("schema")) != "sophia_pretrain_mix_policy_v1":
        raise ValueError("unsupported draft pretrain mix policy")
    draft_status = str(draft.get("status") or "")
    generation = _DRAFT_GENERATIONS.get(draft_status)
    if generation is None:
        raise ValueError("pretrain mix policy is not a supported draft")
    if (
        str(profile.get("schema")) != "sophia_pretrain_corpus_token_profile_v1"
        or str(profile.get("status")) != "complete"
    ):
        raise ValueError("materialization requires a complete token profile")
    total = int(draft.get("requested_unique_train_tokens") or 0)
    targets = {
        str(name): float(value)
        for name, value in (draft.get("target_fractions") or {}).items()
    }
    if total <= 0 or not targets or abs(sum(targets.values()) - 1.0) > 1e-12:
        raise ValueError("draft target fractions must sum to one")
    groups = _normalized_groups(draft)
    source_split = profile.get("tokens_by", {}).get("source_split", {})
    if not isinstance(source_split, dict):
        raise ValueError("token profile is missing source_split totals")
    supply = {
        source: int(source_split.get(f"{source}|train") or 0)
        for group in groups.values()
        for source in group["members"]
    }
    missing = sorted(source for source, tokens in supply.items() if tokens <= 0)
    if missing:
        raise ValueError(f"token profile has no train supply for sources: {missing}")

    group_quotas = _integer_quotas(total=total, fractions=targets)
    adjusted_group_quotas = dict(group_quotas)
    adjustments: list[dict[str, Any]] = []
    for _pass in range(len(groups) + 1):
        changed = False
        for name, group in groups.items():
            members = group["members"]
            available = sum(supply[source] for source in members)
            requested = int(adjusted_group_quotas[name])
            if available >= requested:
                continue
            fallback = str(group["shortfall_reallocation_target"])
            if not fallback or fallback not in groups or fallback == name:
                raise ValueError(
                    "target group has insufficient unique supply without "
                    f"fallback: {name}"
                )
            shortfall = requested - available
            adjusted_group_quotas[name] = available
            adjusted_group_quotas[fallback] += shortfall
            adjustments.append(
                {
                    "source_group": name,
                    "requested_tokens": requested,
                    "available_tokens": available,
                    "reallocated_tokens": shortfall,
                    "reallocation_target": fallback,
                }
            )
            changed = True
        if not changed:
            break
    else:
        raise ValueError("source group shortfall reallocation did not converge")
    adjusted_targets = {
        name: float(tokens) / float(total)
        for name, tokens in adjusted_group_quotas.items()
    }

    source_fractions: dict[str, float] = {}
    for name, group in groups.items():
        members = group["members"]
        target = float(adjusted_targets[name])
        available = sum(supply[source] for source in members)
        if group["allocation"] == "proportional_to_supply":
            for source in members:
                source_fractions[source] = (
                    target * float(supply[source]) / float(available)
                )
        else:
            remaining = target
            for source in members[:-1]:
                allocated = min(float(supply[source]) / float(total), remaining)
                source_fractions[source] = allocated
                remaining -= allocated
            source_fractions[members[-1]] = remaining
        _set_group_total(source_fractions, members, target)

    overall_drift = 1.0 - sum(source_fractions.values())
    fallback_group = "fineweb_edu_english_dedup"
    fallback_source = groups[fallback_group]["members"][-1]
    source_fractions[fallback_source] += overall_drift
    constraints = draft.get("formal_constraints")
    if not isinstance(constraints, dict):
        raise ValueError("draft policy is missing formal constraints")
    policy = {
        "schema": "sophia_pretrain_mix_policy_v1",
        "policy_version": f"{generation}_materialized_from_complete_profile",
        "status": "admitted_pending_supply_resolution",
        "purpose": (
            f"Resolver-compatible unique-only {generation} source mix materialized from "
            "measured tokenizer supply."
        ),
        "requested_unique_train_tokens": total,
        "source_token_fractions": dict(sorted(source_fractions.items())),
        "constraints": constraints,
        "materialization": {
            "draft_path": str(draft_path),
            "draft_sha256": str(draft_sha256),
            "profile_path": str(profile_path),
            "profile_sha256": str(profile_sha256),
            "tokenizer_bundle_sha1": str(profile.get("tokenizer_bundle_sha1") or ""),
            "target_fractions": targets,
            "adjusted_target_fractions": adjusted_targets,
            "source_train_supply_tokens": supply,
            "shortfall_reallocations": adjustments,
        },
    }
    fractions = _validate_policy(policy)
    quotas = _integer_quotas(total=total, fractions=fractions)
    overdrawn = {
        source: quota - supply[source]
        for source, quota in quotas.items()
        if quota > supply[source]
    }
    if overdrawn:
        raise ValueError(
            f"materialized source quotas exceed unique supply: {overdrawn}"
        )
    policy["materialization"]["source_train_quotas"] = quotas
    return policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a resolver-compatible pretrain mix policy."
    )
    parser.add_argument("--draft", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    draft_path = Path(args.draft).expanduser().resolve()
    profile_path = Path(args.profile).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    policy = materialize_policy(
        draft=_load_json(draft_path),
        profile=_load_json(profile_path),
        draft_path=str(draft_path),
        draft_sha256=_sha256(draft_path),
        profile_path=str(profile_path),
        profile_sha256=_sha256(profile_path),
    )
    write_json_atomic(
        output_path,
        policy,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    print(
        f"[DONE] sources={len(policy['source_token_fractions'])} output={output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
