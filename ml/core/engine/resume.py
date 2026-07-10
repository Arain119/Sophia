"""Shared resumable task-state helpers."""

from __future__ import annotations

from collections.abc import Callable

from ml.core.engine.types import StatePayload, Stateful, StateValueT


def restore_iterator_states(
    *,
    resume_state: StatePayload,
    expected_stage: str,
    iterators: dict[str, Stateful | None],
) -> None:
    if not resume_state:
        return
    stage = str(resume_state.get("stage", "") or "")
    if stage and stage != str(expected_stage):
        raise RuntimeError(
            f"checkpoint stage mismatch for {str(expected_stage).upper()}: {stage!r}"
        )
    for state_key, iterator in iterators.items():
        if iterator is None:
            continue
        iterator_state = resume_state.get(state_key)
        if isinstance(iterator_state, dict):
            iterator.load_state_dict(iterator_state)


def read_resume_state_value(
    *,
    resume_state: StatePayload,
    key: str,
    default: StateValueT,
    cast: Callable[[object], StateValueT],
) -> StateValueT:
    if not resume_state:
        return cast(default)
    return cast(resume_state.get(key, default))


def build_train_state(
    *,
    stage: str,
    iterators: dict[str, Stateful | None],
    extra_state: StatePayload,
) -> StatePayload:
    payload: StatePayload = {"stage": str(stage)}
    for state_key, iterator in iterators.items():
        payload[state_key] = {} if iterator is None else dict(iterator.state_dict())
    payload.update(dict(extra_state))
    return payload


__all__ = [
    "build_train_state",
    "read_resume_state_value",
    "restore_iterator_states",
]
