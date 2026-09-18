"""The prompt pool for rejection sampling, cut from the held-out SFT-v3 corpus.

Every row of `release/corpus/val.jsonl` is a teacher-simulated conversation the
policy never trained on, and every user turn carries the move that produced it
(constraint, recall, transform, correct, false_correct, ...). Cutting a
conversation at such a turn gives a prompt whose history provably contains the
thing the answer has to honour, which is exactly what the probe tests and what
the previous pools did not contain.

Prompts play one of two roles. *Fuel* prompts are the dependency moves and
logic: the policy answers them right occasionally and wrong usually, which is
the shape rejection sampling can sharpen; they are sampled wide. *Anchor*
prompts are every other category the release has to keep -- companion chat,
QA, writing, English, identity, code, maths, structured output, safety -- and
exist so the round does not drift off them; they are sampled narrow and kept
only when the best clearly beats the group.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

DEPENDENCY = {"constraint", "recall", "recall_far", "transform", "correct", "false_correct", "summarize"}

# share of the anchor half, and the cap the corpus can actually supply
ANCHOR_SHARES = {
    "companion_chat": 0.24,
    "chinese_qa": 0.19,
    "writing": 0.14,
    "english": 0.12,
    "identity": 0.10,
    "code": 0.07,
    "math": 0.07,
    "structured": 0.05,
    "safety": 0.02,
}
RENAME = {
    "chinese_writing": "writing",
    "english_general": "english",
    "structured_output": "structured",
    "dialogue": "companion_chat",
    "daily": "companion_chat",
}


def category_of(row: dict) -> str:
    name = str(row.get("category") or "").removeprefix("replay_")
    return RENAME.get(name, name)


def cuts(row: dict) -> list[dict]:
    """One candidate prompt per user turn, tagged with the move that wrote it."""
    moves = list(row.get("moves") or [])
    out = []
    user_index = 0
    for index, message in enumerate(row["messages"]):
        if message["role"] != "user":
            continue
        move = moves[user_index] if user_index < len(moves) else "follow"
        user_index += 1
        out.append(
            {
                "prompt_id": f"{row['id']}@{index}",
                "messages": [dict(m) for m in row["messages"][: index + 1]],
                "category": category_of(row),
                "source": str(row.get("category") or ""),
                "move": move,
                "turns": index + 1,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="release/corpus/val.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchors", type=int, default=1700)
    parser.add_argument("--fuel-samples", type=int, default=16)
    parser.add_argument("--anchor-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    fuel, anchors = [], collections.defaultdict(list)
    for line in Path(args.corpus).open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        for cut in cuts(row):
            is_logic = cut["category"] == "logic"
            if cut["move"] in DEPENDENCY or is_logic:
                cut["role"], cut["samples"] = "fuel", args.fuel_samples
                fuel.append(cut)
            elif cut["category"] in ANCHOR_SHARES:
                cut["role"], cut["samples"] = "anchor", args.anchor_samples
                anchors[cut["category"]].append(cut)

    chosen = []
    for category, share in ANCHOR_SHARES.items():
        rows = anchors[category]
        rng.shuffle(rows)
        chosen.extend(rows[: int(round(args.anchors * share))])
    pool = fuel + chosen
    rng.shuffle(pool)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for row in pool:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "prompts": len(pool),
        "fuel": len(fuel),
        "anchors": len(chosen),
        "samples_total": sum(r["samples"] for r in pool),
        "fuel_by_move": dict(collections.Counter(r["move"] for r in fuel).most_common()),
        "fuel_by_category": dict(collections.Counter(r["category"] for r in fuel).most_common()),
        "anchor_by_category": dict(collections.Counter(r["category"] for r in chosen).most_common()),
        "anchor_available": {k: len(v) for k, v in anchors.items()},
        "turns_median": sorted(r["turns"] for r in pool)[len(pool) // 2],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
