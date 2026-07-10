"""Unit tests for the shared Pareto-frontier helper (``tooling.core.pareto``).

The pretrain / SFT sweeps delegate frontier selection here, so this
pins the generic behavior in one place.
"""

from __future__ import annotations

import math

from ml.tooling.core.pareto import dominates, pareto_frontier


def test_dominates_requires_no_worse_and_strictly_better() -> None:
    objectives = (("loss", "min"), ("speed", "max"))
    a = {"loss": 1.0, "speed": 10.0}
    b = {"loss": 2.0, "speed": 10.0}  # worse loss, equal speed
    assert dominates(a, b, objectives=objectives)
    assert not dominates(b, a, objectives=objectives)
    # Equal on everything -> neither dominates.
    assert not dominates(a, dict(a), objectives=objectives)


def test_dominates_false_when_metric_non_numeric() -> None:
    objectives = (("loss", "min"),)
    assert not dominates({"loss": None}, {"loss": 1.0}, objectives=objectives)
    assert not dominates({"loss": 1.0}, {"loss": "x"}, objectives=objectives)


def test_pareto_frontier_keeps_only_non_dominated_ok_rows() -> None:
    objectives = (("loss", "min"), ("speed", "max"))
    rows = [
        {"name": "a", "status": "ok", "loss": 1.0, "speed": 10.0},  # frontier
        {"name": "b", "status": "ok", "loss": 2.0, "speed": 20.0},  # frontier (trade-off)
        {"name": "c", "status": "ok", "loss": 2.0, "speed": 5.0},   # dominated by a
        {"name": "d", "status": "failed", "loss": 0.1, "speed": 99.0},  # excluded: not ok
        {"name": "e", "status": "ok", "loss": float("nan"), "speed": 50.0},  # excluded: non-finite
        {"name": "f", "status": "ok", "speed": 30.0},  # excluded: missing objective
    ]
    names = {row["name"] for row in pareto_frontier(rows, objectives=objectives)}
    assert names == {"a", "b"}


def test_pareto_frontier_treats_missing_status_as_not_ok() -> None:
    objectives = (("loss", "min"),)
    rows = [{"name": "a", "loss": 1.0}, {"name": "b", "status": None, "loss": 0.5}]
    assert pareto_frontier(rows, objectives=objectives) == []
    assert math.isfinite(1.0)  # sanity: helper depends on finite checks
