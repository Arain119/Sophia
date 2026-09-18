from __future__ import annotations

import contextlib

import torch

import ml.core.common.device_batch as device_batch_mod


class _FakeLoader:
    def __init__(self) -> None:
        self._rows = [
            {"input_ids": torch.tensor([1], dtype=torch.int32)},
            {"input_ids": torch.tensor([2], dtype=torch.int32)},
            {"input_ids": torch.tensor([3], dtype=torch.int32)},
        ]
        self._pos = 0

    def __iter__(self) -> _FakeLoader:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._pos >= len(self._rows):
            raise StopIteration
        row = self._rows[self._pos]
        self._pos += 1
        return row

    def state_dict(self) -> dict[str, int]:
        return {"pos": int(self._pos)}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._pos = int(state["pos"])


def test_resumable_cuda_prefetcher_preserves_exact_resume_state(monkeypatch) -> None:
    class _FakeCurrentStream:
        def wait_stream(self, _stream) -> None:
            return None

    monkeypatch.setattr(device_batch_mod, "move_batch_to_device", lambda batch, *, device: dict(batch))
    monkeypatch.setattr(device_batch_mod.torch.cuda, "Stream", lambda device=None: object())
    monkeypatch.setattr(device_batch_mod.torch.cuda, "current_stream", lambda device=None: _FakeCurrentStream())
    monkeypatch.setattr(device_batch_mod.torch.cuda, "stream", lambda stream: contextlib.nullcontext())

    loader = _FakeLoader()
    prefetched = device_batch_mod.ResumableCudaPrefetcher(loader, device=torch.device("cuda:0"))

    assert prefetched.state_dict() == {"pos": 0}

    first = next(prefetched)
    assert int(first["input_ids"][0].item()) == 1
    assert prefetched.state_dict() == {"pos": 1}

    resumed = device_batch_mod.ResumableCudaPrefetcher(_FakeLoader(), device=torch.device("cuda:0"))
    resumed.load_state_dict(prefetched.state_dict())
    second = next(resumed)
    assert int(second["input_ids"][0].item()) == 2
    assert resumed.state_dict() == {"pos": 2}


def test_move_batch_to_device_reuses_identical_tensor_objects(monkeypatch) -> None:
    calls: list[int] = []

    def _fake_move_tensor_to_device(value: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        del device
        calls.append(id(value))
        return value + 1

    monkeypatch.setattr(
        device_batch_mod,
        "_move_tensor_to_device",
        _fake_move_tensor_to_device,
    )

    shared = torch.tensor([7], dtype=torch.int32)
    batch = {
        "input_ids": shared,
        "labels": shared,
        "labels_is_input_ids": True,
    }
    moved = device_batch_mod.move_batch_to_device(batch, device=torch.device("cpu"))

    assert calls == [id(shared)]
    assert moved["labels"] is moved["input_ids"]
    assert int(moved["input_ids"][0].item()) == 8
