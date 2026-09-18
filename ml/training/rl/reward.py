"""The reward a completion earns, and the parts of it that need no model.

The release bar is coherence, not correctness: a wrong-but-well-formed answer
is acceptable, an answer that contradicts itself or spins in place is not. Part
of that bar is decidable from the text alone -- the completion repeats itself,
or it never produced what was asked for -- so it is computed here exactly
rather than learned.

Measured on the twenty-case probe of ckpt_step1700, these terms flag three of
the ten failures and none of the ten passes: high precision, low recall. They
are a hard floor, not the signal. The other seven failures are a boiling point
defined as mass over mass, recursion explained as a shell around code, an
offer to look up a stranger's address -- all fluent, none repetitive. Those
belong to the reward model, which therefore carries most of the discrimination
and has to be held to it.

Every term is returned separately. A single scalar tells you the policy is
improving; the breakdown tells you which term it is improving against, which
is the only way to notice that it has found a way to satisfy the reward
without satisfying the bar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


CODE_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n.*?```", re.DOTALL)
NUMERAL = re.compile(r"[0-9]|[一二三四五六七八九十百千万零两]")
_WHITESPACE = re.compile(r"\s+")


def loop_coverage(text: str, *, min_period: int = 2, max_period: int = 80) -> float:
    """Fraction of the text consumed by a phrase repeated back to back.

    This is the exact shape of the run's worst failure -- "蛋黄的汤汁要清爽"
    ten times over, "12 乘以 8 等于 12 乘以 8 等于 8 乘以 8" -- and it is the
    one failure mode that needs no judgement at all.

    A tandem repeat of period ``p`` shows up as a run of positions where
    ``text[i] == text[i + p]``, so each period costs one linear scan rather
    than a quadratic search over substrings.
    """
    condensed = _WHITESPACE.sub("", text)
    length = len(condensed)
    if length < 2 * min_period:
        return 0.0
    best = 0
    for period in range(min_period, min(max_period, length // 2) + 1):
        covered = 0
        run = 0
        for index in range(length - period):
            if condensed[index] == condensed[index + period]:
                run += 1
                continue
            if run >= period:
                covered += run + period
            run = 0
        if run >= period:
            covered += run + period
        if covered > best:
            best = covered
    return min(best / length, 1.0)


def distinct_ngram_ratio(text: str, *, n: int = 4) -> float:
    """Share of character n-grams that occur exactly once. Low means padding."""
    condensed = _WHITESPACE.sub("", text)
    if len(condensed) <= n:
        return 1.0
    grams = [condensed[i : i + n] for i in range(len(condensed) - n + 1)]
    return len(set(grams)) / len(grams)


def repetition_score(text: str) -> float:
    """0 for clean text, 1 for a completion that is entirely a loop."""
    loop = loop_coverage(text)
    distinct = distinct_ngram_ratio(text)
    # A loop is decisive; sparse n-grams only matter when no loop was found,
    # because a long answer that merely reuses vocabulary is not a defect.
    return max(loop, max(0.0, 1.0 - distinct / 0.6) if loop < 0.2 else loop)


def _normalised(text: str) -> str:
    return _WHITESPACE.sub("", text)


def responsiveness_score(
    text: str,
    *,
    expectations: dict[str, Any] | None = None,
) -> tuple[float, list[str]]:
    """Whether the completion produced the artefact the prompt asked for.

    The prompt pool carries these expectations rather than inferring them
    here: "write a function" is answered with a code block, "rewrite this" is
    answered with text that differs from the source, "compute this" is
    answered with a number. Missing any of them is the failure that a purely
    semantic judge scores as fine -- the sentences are consistent, they are
    just not an answer.
    """
    checks = dict(expectations or {})
    if not checks:
        return 1.0, []
    missed: list[str] = []
    if checks.get("code_block") and not CODE_FENCE.search(text):
        missed.append("code_block")
    if checks.get("numeral") and not NUMERAL.search(text):
        missed.append("numeral")
    source = checks.get("differs_from")
    if source and _normalised(str(source)) in _normalised(text):
        missed.append("differs_from")
    minimum = int(checks.get("min_chars") or 0)
    if minimum and len(_normalised(text)) < minimum:
        missed.append("min_chars")
    satisfied = len(checks) - len(missed)
    return satisfied / len(checks), missed


@dataclass(frozen=True)
class RewardWeights:
    repetition: float = 1.0
    coherence: float = 1.0
    responsiveness: float = 0.5
    unfinished_penalty: float = 0.5
    empty_penalty: float = 1.0


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    repetition: float
    coherence: float
    responsiveness: float
    penalties: float
    missed: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)


DEFAULT_REWARD_WEIGHTS = RewardWeights()


def score_completion(
    text: str,
    *,
    coherence: float = 1.0,
    hit_eos: bool = True,
    expectations: dict[str, Any] | None = None,
    weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
) -> RewardBreakdown:
    """Reward for one completion.

    ``coherence`` comes from the reward model and is expected in [0, 1]; pass
    the default when scoring without one, so that the programmatic terms can
    be measured and tested on their own.
    """
    stripped = text.strip()
    repetition = repetition_score(stripped)
    responsiveness, missed = responsiveness_score(stripped, expectations=expectations)
    penalties = 0.0
    if not stripped:
        penalties += weights.empty_penalty
    if not hit_eos:
        # Running into the token ceiling is not a style choice; every
        # completion in the probe that failed to stop was mid-loop.
        penalties += weights.unfinished_penalty
    total = (
        weights.repetition * (1.0 - repetition)
        + weights.coherence * float(coherence)
        + weights.responsiveness * responsiveness
        - penalties
    )
    return RewardBreakdown(
        total=total,
        repetition=repetition,
        coherence=float(coherence),
        responsiveness=responsiveness,
        penalties=penalties,
        missed=tuple(missed),
        notes={
            "chars": len(stripped),
            "loop_coverage": loop_coverage(stripped),
            "distinct_4gram": distinct_ngram_ratio(stripped),
        },
    )


__all__ = [
    "RewardBreakdown",
    "RewardWeights",
    "distinct_ngram_ratio",
    "loop_coverage",
    "repetition_score",
    "responsiveness_score",
    "score_completion",
]
