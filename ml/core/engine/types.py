"""Shared engine payload/type contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable


MetricScalar: TypeAlias = str | int | float | bool | None
MetricValue: TypeAlias = MetricScalar | list["MetricValue"] | dict[str, "MetricValue"]
MetricsRow: TypeAlias = dict[str, MetricValue]

StatePayload: TypeAlias = dict[str, object]
RuntimeMetadata: TypeAlias = dict[str, object]

StateValueT = TypeVar("StateValueT")


@runtime_checkable
class Stateful(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


__all__ = [
    "MetricsRow",
    "MetricScalar",
    "MetricValue",
    "RuntimeMetadata",
    "StatePayload",
    "StateValueT",
    "Stateful",
]
