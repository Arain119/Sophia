"""Pareto-frontier selection shared by the same-chain machine-selection tools.

The pretrain / SFT machine-selection commands reduce a list of run
summaries to the
non-dominated frontier over a set of ``(metric, direction)`` objectives. That
frontier math is identical across them; only the per-command metric
*flattening* differs (different metric names), so that part stays in each
command.
"""

from __future__ import annotations

import math
from typing import Any

# Each objective is ``(metric_key, "min" | "max")``.
Objectives = tuple[tuple[str, str], ...]


def dominates(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    objectives: Objectives,
) -> bool:
    """Return True if ``left`` Pareto-dominates ``right`` over ``objectives``.

    Domination requires ``left`` to be no worse on every objective and strictly
    better on at least one. Non-numeric values on either side disqualify the pair.
    """
    better = False
    for key, mode in objectives:
        lv = left.get(key)
        rv = right.get(key)
        if not isinstance(lv, (int, float)) or not isinstance(rv, (int, float)):
            return False
        if mode == "min":
            if float(lv) > float(rv):
                return False
            if float(lv) < float(rv):
                better = True
        elif mode == "max":
            if float(lv) < float(rv):
                return False
            if float(lv) > float(rv):
                better = True
        else:  # pragma: no cover
            raise ValueError(f"unsupported objective mode: {mode}")
    return bool(better)


def _has_all_objectives(row: dict[str, Any], objectives: Objectives) -> bool:
    for key, _mode in objectives:
        value = row.get(key)
        if not isinstance(value, (int, float)):
            return False
        if not math.isfinite(float(value)):
            return False
    return True


def pareto_frontier(
    rows: list[dict[str, Any]],
    *,
    objectives: Objectives,
) -> list[dict[str, Any]]:
    """Return the non-dominated ``status == "ok"`` rows over ``objectives``.

    Rows that are not ``ok`` or that are missing/non-finite on any objective
    metric are excluded before the frontier is computed.
    """
    valid = [
        row
        for row in rows
        if str(row.get("status", "") or "") == "ok" and _has_all_objectives(row, objectives)
    ]
    out: list[dict[str, Any]] = []
    for candidate in valid:
        dominated = False
        for other in valid:
            if other is candidate:
                continue
            if dominates(other, candidate, objectives=objectives):
                dominated = True
                break
        if not dominated:
            out.append(candidate)
    return out
