"""Attach deterministic programmatic rewards to rollout groups.

This is the post-budget alignment path. It never imports a provider client or
reads credentials. The reward is intentionally narrow: repetition, stopping,
and artefacts explicitly required by the prompt. Semantic quality remains a
human/evidence decision, not a hidden heuristic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.training.rl.reward import score_completion


def score_rollouts(*, groups_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError(f"scored rollout output already exists: {output_path}")
    rows = [
        json.loads(line)
        for line in groups_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("rollout group file is empty")
    scored_groups = 0
    scored_completions = 0
    signal_groups = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for group in rows:
            if not isinstance(group, dict) or not isinstance(group.get("completions"), list):
                raise ValueError("rollout group is malformed")
            expectations = dict(group.get("expectations") or {})
            scores: list[float] = []
            completions: list[dict[str, Any]] = []
            for row in group["completions"]:
                if not isinstance(row, dict):
                    continue
                item = dict(row)
                reward = score_completion(
                    str(item.get("raw") or ""),
                    hit_eos=bool(item.get("hit_eos", False)),
                    expectations=expectations,
                )
                item["score"] = float(reward.total)
                item["reward_breakdown"] = {
                    "repetition": reward.repetition,
                    "coherence": reward.coherence,
                    "responsiveness": reward.responsiveness,
                    "penalties": reward.penalties,
                    "missed": list(reward.missed),
                }
                scores.append(float(reward.total))
                completions.append(item)
            if len(scores) < 2:
                continue
            mean = sum(scores) / len(scores)
            variance = sum((value - mean) ** 2 for value in scores) / len(scores)
            signal_groups += int(variance**0.5 >= 0.05)
            updated = dict(group)
            updated["completions"] = completions
            updated["reward_source"] = "programmatic_local"
            handle.write(json.dumps(updated, ensure_ascii=False) + "\n")
            scored_groups += 1
            scored_completions += len(completions)
    if scored_groups == 0:
        raise RuntimeError("no rollout groups had two valid completions")
    report = {
        "schema": "sophia_local_rollout_reward_report",
        "status": "complete",
        "input": str(groups_path.resolve()),
        "output": str(output_path.resolve()),
        "groups": scored_groups,
        "completions": scored_completions,
        "groups_with_signal": signal_groups,
        "group_signal_rate": signal_groups / scored_groups,
        "reward_source": "programmatic_local",
    }
    output_path.with_name(output_path.name + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = score_rollouts(groups_path=args.groups, output_path=args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
