"""Shared engine payload type aliases."""

from __future__ import annotations


type MetricsRow = dict[str, object]
type StatePayload = dict[str, object]
type RuntimeMetadata = dict[str, object]


__all__ = [
    "MetricsRow",
    "RuntimeMetadata",
    "StatePayload",
]
