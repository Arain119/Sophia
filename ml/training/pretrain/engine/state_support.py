from __future__ import annotations

from collections.abc import Mapping
from ml.training.pretrain.resources import PretrainDataIter
from ml.training.pretrain.train_state import DataIterResumeState
from ml.training.pretrain.train_state import _tree_snapshot


def _load_data_iter_state(
    data_iter: PretrainDataIter,
    state: DataIterResumeState | Mapping[str, object] | None,
) -> None:
    if state is None:
        return
    load_fn = getattr(data_iter, "load_state_dict", None)
    if not callable(load_fn):
        raise RuntimeError("data iterator does not support load_state_dict for rollback")
    payload = state.to_payload() if isinstance(state, DataIterResumeState) else dict(state)
    load_fn(payload)


def _capture_data_iter_state_or_raise(
    data_iter: PretrainDataIter,
    *,
    feature_name: str,
) -> dict[str, object]:
    try:
        state = data_iter.state_dict()
    except Exception as exc:
        raise RuntimeError(
            f"{feature_name} requires an exact data iterator snapshot, but state capture failed: {exc}"
        ) from exc
    if not isinstance(state, dict):
        raise RuntimeError(
            f"{feature_name} requires a dict-like data iterator state, got {type(state).__name__}"
        )
    payload = DataIterResumeState.from_payload(state)
    if payload is None:
        return dict(state)
    return dict(payload.payload)
__all__ = [
    "_capture_data_iter_state_or_raise",
    "_load_data_iter_state",
    "_tree_snapshot",
]
