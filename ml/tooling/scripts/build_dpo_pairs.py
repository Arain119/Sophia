"""Select DPO pairs from scored rollout groups.

The scorer is intentionally external to this command.  A relay judge, a
reward model, or a deterministic verifier may attach ``score`` to each
completion; this tool only applies the selection rule and writes the stable
pair schema consumed by the policy trainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.training.rl.dpo import PreferencePair, write_preference_pairs


def _load_groups(path: Path) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"rollout group {line_number} is not an object")
            groups.append(payload)
    if not groups:
        raise ValueError("rollout group file is empty")
    return groups


def build_pairs(
    *,
    groups_path: Path,
    output_path: Path,
    min_margin: float = 0.25,
    max_pairs: int = 0,
) -> dict[str, int | float | str]:
    """Choose the highest/lowest scored member of each useful group."""

    pairs: list[PreferencePair] = []
    skipped_missing = 0
    skipped_margin = 0
    for group in _load_groups(groups_path):
        completions = group.get("completions")
        messages = group.get("messages")
        if not isinstance(completions, list) or not isinstance(messages, list) or not messages:
            raise ValueError("rollout group must contain messages and completions")
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in completions:
            if not isinstance(row, dict) or "score" not in row:
                continue
            text = str(row.get("raw") or "")
            if not text:
                continue
            try:
                score = float(row["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError("completion score must be numeric") from exc
            scored.append((score, row))
        if len(scored) < 2:
            skipped_missing += 1
            continue
        rejected_score, rejected = min(scored, key=lambda item: (item[0], str(item[1].get("index", ""))))
        chosen_score, chosen = max(scored, key=lambda item: (item[0], str(item[1].get("index", ""))))
        if chosen_score - rejected_score < float(min_margin) or chosen["raw"] == rejected["raw"]:
            skipped_margin += 1
            continue
        pairs.append(
            PreferencePair(
                prompt_id=str(group.get("prompt_id") or f"group_{len(pairs):06d}"),
                messages=[dict(message) for message in messages],
                chosen=str(chosen["raw"]),
                rejected=str(rejected["raw"]),
                chosen_score=chosen_score,
                rejected_score=rejected_score,
                source=str(group.get("source") or "rollout_judge"),
            )
        )
        if int(max_pairs) > 0 and len(pairs) >= int(max_pairs):
            break
    if not pairs:
        raise RuntimeError("no rollout groups met the DPO margin")
    write_preference_pairs(output_path, pairs)
    return {
        "groups": len(_load_groups(groups_path)),
        "pairs": len(pairs),
        "skipped_missing_scores": skipped_missing,
        "skipped_margin": skipped_margin,
        "min_margin": float(min_margin),
        "output": str(output_path.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-margin", type=float, default=0.25)
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_pairs(
        groups_path=args.groups,
        output_path=args.output,
        min_margin=args.min_margin,
        max_pairs=args.max_pairs,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
