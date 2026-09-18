"""Choosing the prompts a policy-optimisation step is allowed to spend time on.

A group of completions only carries a gradient when its members disagree, so
the pool is not "every prompt we have". Two thirds of the SFT corpus's maths
rows come from seventy-four templates and three quarters of its code rows from
twenty; on those the policy retrieves the same memorised answer every time,
the group agrees, and the advantage is zero. The measurement that found this
is the same one used here: strip the digits, count the skeletons, and mark
every prompt that sits inside a heavily repeated one.

Marked rather than dropped. A run that trains only outside the template set
still wants to know what happened inside it, and the tag is what makes that
comparison possible afterwards.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Numbers are the slot a template varies, so removing them is what turns
# "儿子今年 10 岁" and "儿子今年 17 岁" into one shape.
_DIGITS = re.compile(r"[0-9]+|[零一二三四五六七八九十百千万两]+")
_WHITESPACE = re.compile(r"\s+")

# Measured on release/corpus: every healthy category sits under ten rows per
# skeleton, maths averages 162 and the top code templates over 700.
DEFAULT_TEMPLATE_THRESHOLD = 50

IN_TEMPLATE = "in_template_set"
OUTSIDE_TEMPLATE = "outside_template_set"


def skeleton(text: str) -> str:
    """The prompt with its varying numbers removed."""
    return _WHITESPACE.sub("", _DIGITS.sub("#", text))


def first_user_message(messages: Iterable[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _rows(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def memorised_skeletons(
    train_rows: Iterable[dict[str, Any]],
    *,
    threshold: int = DEFAULT_TEMPLATE_THRESHOLD,
) -> set[str]:
    """Skeletons the policy has seen often enough to answer from memory."""
    counts: Counter[str] = Counter()
    for row in train_rows:
        counts[skeleton(first_user_message(row.get("messages") or []))] += 1
    return {shape for shape, count in counts.items() if count >= int(threshold)}


def derive_expectations(row: dict[str, Any]) -> dict[str, Any]:
    """Artefacts an answer to this prompt has to contain.

    Only the checks that can be read off the row without guessing. A wrong
    expectation is worse than a missing one: it teaches the policy to satisfy
    something the prompt never asked for.
    """
    category = str(row.get("category") or "")
    expectations: dict[str, Any] = {}
    if category == "code":
        expectations["code_block"] = True
    elif category == "math":
        expectations["numeral"] = True
    return expectations


@dataclass(frozen=True)
class PoolStats:
    total: int
    inside: int
    outside: int
    skeletons: int
    memorised: int


def build_pool(
    *,
    candidate_rows: Iterable[dict[str, Any]],
    memorised: set[str],
    prefix: str = "p",
) -> tuple[list[dict[str, Any]], PoolStats]:
    """Turn dataset rows into prompt-pool entries, tagged by template status."""
    pool: list[dict[str, Any]] = []
    shapes: set[str] = set()
    inside = 0
    for index, row in enumerate(candidate_rows):
        prompt_messages = [
            {"role": str(message.get("role")), "content": str(message.get("content") or "")}
            for message in row.get("messages") or []
            if message.get("role") in {"user", "assistant", "system"}
        ]
        # The prompt is the conversation up to the last thing the user said;
        # the assistant turn that followed is what the policy has to produce.
        while prompt_messages and prompt_messages[-1]["role"] != "user":
            prompt_messages.pop()
        if not prompt_messages:
            continue
        shape = skeleton(first_user_message(prompt_messages))
        shapes.add(shape)
        tag = IN_TEMPLATE if shape in memorised else OUTSIDE_TEMPLATE
        inside += tag == IN_TEMPLATE
        pool.append(
            {
                "prompt_id": f"{prefix}_{index:06d}",
                "messages": prompt_messages,
                "expectations": derive_expectations(row),
                "tags": [str(row.get("category") or "unknown"), tag],
            }
        )
    stats = PoolStats(
        total=len(pool),
        inside=inside,
        outside=len(pool) - inside,
        skeletons=len(shapes),
        memorised=len(memorised),
    )
    return pool, stats


def write_pool(path: str | Path, pool: Iterable[dict[str, Any]]) -> int:
    written = 0
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for entry in pool:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
    return written


def build_from_dataset(
    *,
    train_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path,
    threshold: int = DEFAULT_TEMPLATE_THRESHOLD,
) -> PoolStats:
    """Build a pool from a split, excluding nothing but marking everything."""
    memorised = memorised_skeletons(_rows(train_path), threshold=threshold)
    pool, stats = build_pool(
        candidate_rows=_rows(candidate_path), memorised=memorised
    )
    write_pool(output_path, pool)
    return stats


__all__ = [
    "DEFAULT_TEMPLATE_THRESHOLD",
    "IN_TEMPLATE",
    "OUTSIDE_TEMPLATE",
    "PoolStats",
    "build_from_dataset",
    "build_pool",
    "derive_expectations",
    "first_user_message",
    "memorised_skeletons",
    "skeleton",
    "write_pool",
]
