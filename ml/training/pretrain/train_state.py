from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class DataIterResumeState:
    payload: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object] | None,
    ) -> DataIterResumeState | None:
        if not isinstance(payload, Mapping):
            return None
        return cls(payload=dict(payload))

    def to_payload(self) -> dict[str, object]:
        return dict(self.payload)


@dataclass(frozen=True)
class PretrainTrainState:
    serialized_payload: dict[str, object] = field(default_factory=dict)
    seen_supervised_tokens: int = 0
    best_eval_loss: float | None = None
    seq_len: int | None = None
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


__all__ = [
    "DataIterResumeState",
    "PretrainTrainState",
]
