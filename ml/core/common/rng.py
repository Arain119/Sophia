from __future__ import annotations

import random

import numpy as np
import torch


def _pack_numpy_rng_state(np_state: object) -> object:
    if not isinstance(np_state, tuple) or len(np_state) != 5:
        return np_state
    bitgen, state_arr, pos, has_gauss, cached_gaussian = np_state
    if not isinstance(bitgen, str) or not hasattr(state_arr, "tolist"):
        return np_state
    return {
        "bit_generator": bitgen,
        "state": state_arr.tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _unpack_numpy_rng_state(obj: object) -> object:
    if isinstance(obj, dict) and obj.get("bit_generator") and obj.get("state") is not None:
        bitgen = str(obj.get("bit_generator"))
        state_list = obj.get("state")
        pos = int(obj.get("pos", 0) or 0)
        has_gauss = int(obj.get("has_gauss", 0) or 0)
        cached_gaussian = float(obj.get("cached_gaussian", 0.0) or 0.0)
        state_arr = np.asarray(state_list, dtype=np.uint32)
        return (bitgen, state_arr, pos, has_gauss, cached_gaussian)
    return obj


def capture_rng_state() -> dict:
    state: dict = {
        "python": random.getstate(),
        "numpy": _pack_numpy_rng_state(np.random.get_state()),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict) -> None:
    if not isinstance(state, dict):
        return
    errors: list[str] = []
    if "python" in state:
        try:
            random.setstate(state["python"])
        except Exception as exc:
            errors.append(f"python: {exc}")
    if "numpy" in state:
        try:
            np.random.set_state(_unpack_numpy_rng_state(state["numpy"]))
        except Exception as exc:
            errors.append(f"numpy: {exc}")
    if "torch" in state:
        try:
            if torch.is_tensor(state["torch"]):
                torch.set_rng_state(state["torch"])
            else:
                raise TypeError(f"expected torch Tensor, got {type(state['torch'])}")
        except Exception as exc:
            errors.append(f"torch: {exc}")
    if "cuda" in state and torch.cuda.is_available():
        try:
            if torch.is_tensor(state["cuda"]):
                torch.cuda.set_rng_state(state["cuda"])
            else:
                raise TypeError(f"expected torch Tensor, got {type(state['cuda'])}")
        except Exception as exc:
            errors.append(f"cuda: {exc}")
    if errors:
        raise RuntimeError("Failed to restore RNG state: " + "; ".join(errors))
