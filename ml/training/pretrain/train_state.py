from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch


def _is_unset(value: object) -> bool:
    return value is None or value == ""


def _coerce_int(value: object, *, default: int = 0) -> int:
    if _is_unset(value):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"expected integer train state value, got {value!r}") from exc


def _coerce_float(value: object, *, default: float | None = None) -> float | None:
    if _is_unset(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"expected numeric train state value, got {value!r}") from exc


def _coerce_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _tree_snapshot(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _tree_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_tree_snapshot(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return type(value)(_tree_snapshot(item) for item in value)
    return value


@dataclass(frozen=True)
class DataIterResumeState:
    payload: dict[str, object] = field(default_factory=dict)
    order: tuple[int, ...] = ()
    order_pos: int = 0
    current_pos: int | None = None

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object] | None,
    ) -> DataIterResumeState | None:
        if not isinstance(payload, Mapping):
            return None
        state = dict(payload)
        order_raw = state.get("order")
        if not isinstance(order_raw, list):
            return None
        try:
            order = tuple(int(item) for item in order_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError("data iterator order must contain integers") from exc
        current_pos_raw = state.get("current_pos")
        current_pos = (
            None if current_pos_raw is None else _coerce_int(current_pos_raw, default=0)
        )
        return cls(
            payload=state,
            order=order,
            order_pos=_coerce_int(state.get("order_pos"), default=0),
            current_pos=current_pos,
        )

    def to_payload(self) -> dict[str, object]:
        payload = dict(self.payload)
        payload["order"] = list(self.order)
        payload["order_pos"] = int(self.order_pos)
        payload["current_pos"] = self.current_pos
        return payload


@dataclass(frozen=True)
class PretrainTrainState:
    serialized_payload: dict[str, object] = field(default_factory=dict)
    seen_supervised_tokens: int = 0
    best_eval_loss: float | None = None
    seq_len: int | None = None
    lr_decay_start_step: int | None = None
    data_iter_state: DataIterResumeState | None = None

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object] | None,
    ) -> PretrainTrainState:
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("pretrain train state must be a mapping")
        raw_payload = dict(payload)
        data_iter_payload = raw_payload.get("data_iter_state")
        if data_iter_payload is not None and not isinstance(data_iter_payload, Mapping):
            raise TypeError("data_iter_state must be a mapping")
        return cls(
            serialized_payload=raw_payload,
            seen_supervised_tokens=_coerce_int(
                raw_payload.get("seen_supervised_tokens"),
                default=0,
            ),
            best_eval_loss=_coerce_float(
                raw_payload.get("best_eval_loss"),
                default=None,
            ),
            seq_len=_coerce_int(raw_payload.get("seq_len"), default=0)
            if "seq_len" in raw_payload
            else None,
            lr_decay_start_step=_coerce_int(
                raw_payload.get("lr_decay_start_step"),
                default=0,
            )
            if "lr_decay_start_step" in raw_payload
            else None,
            data_iter_state=DataIterResumeState.from_payload(
                data_iter_payload
            ),
        )

    @classmethod
    def resolve(
        cls,
        state: PretrainTrainState | Mapping[str, object] | None,
    ) -> PretrainTrainState:
        if isinstance(state, PretrainTrainState):
            return state
        return cls.from_payload(state)

    def to_payload(self) -> dict[str, object]:
        return dict(self.serialized_payload)

    def has_serialized_state(self) -> bool:
        return bool(self.serialized_payload)

    def cleared_for_stage_transition(
        self,
        *,
        clear_data_iter_state: bool = False,
    ) -> PretrainTrainState:
        payload = dict(self.serialized_payload)
        if bool(clear_data_iter_state):
            payload.pop("data_iter_state", None)
        return PretrainTrainState.from_payload(payload)


__all__ = [
    "DataIterResumeState",
    "PretrainTrainState",
    "_tree_snapshot",
]
