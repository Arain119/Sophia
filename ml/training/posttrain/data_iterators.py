from __future__ import annotations

import random
from collections.abc import Callable

import torch

from ml.core.engine.types import StatePayload
from ml.training.posttrain.contracts import SupportsPosttrainTokenizer
from ml.training.posttrain.data_types import ChatExample


class ShuffledBatchIterator:
    def __init__(
        self,
        *,
        dataset: list[ChatExample],
        batch_size: int,
        seed: int,
        shuffle: bool,
        repeat: bool,
        drop_last: bool,
    ) -> None:
        if not dataset:
            raise ValueError("dataset must be non-empty")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0")
        if bool(drop_last) and int(len(dataset)) < int(batch_size):
            raise ValueError("drop_last=True requires dataset_size >= batch_size")
        self._dataset = dataset
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._shuffle = bool(shuffle)
        self._repeat = bool(repeat)
        self._drop_last = bool(drop_last)
        self._epoch = 0
        self._position = 0
        self._order = list(range(len(self._dataset)))
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._order = list(range(len(self._dataset)))
        if self._shuffle:
            rng = random.Random(self._seed + self._epoch)
            rng.shuffle(self._order)

    def state_dict(self) -> StatePayload:
        return {
            "dataset_size": int(len(self._dataset)),
            "batch_size": int(self._batch_size),
            "seed": int(self._seed),
            "shuffle": bool(self._shuffle),
            "repeat": bool(self._repeat),
            "drop_last": bool(self._drop_last),
            "epoch": int(self._epoch),
            "position": int(self._position),
            "order": list(self._order),
        }

    def load_state_dict(self, state: StatePayload) -> None:
        if not isinstance(state, dict):
            raise TypeError("iterator state must be a dict")
        dataset_size = int(state.get("dataset_size", -1))
        if dataset_size != int(len(self._dataset)):
            raise ValueError(
                f"iterator dataset_size mismatch: {dataset_size} != {len(self._dataset)}"
            )
        batch_size = int(state.get("batch_size", -1))
        if batch_size != int(self._batch_size):
            raise ValueError(
                f"iterator batch_size mismatch: {batch_size} != {self._batch_size}"
            )
        order = state.get("order")
        if not isinstance(order, list) or len(order) != len(self._dataset):
            raise ValueError("iterator state is missing a valid `order` list")
        self._epoch = int(state.get("epoch", 0) or 0)
        self._position = int(state.get("position", 0) or 0)
        self._order = [int(idx) for idx in order]

    def _next_indices_once(self) -> list[int]:
        total = len(self._dataset)
        if self._position >= total:
            raise StopIteration
        end = min(self._position + self._batch_size, total)
        if self._drop_last and (end - self._position) < self._batch_size:
            self._position = total
            raise StopIteration
        indices = self._order[self._position:end]
        self._position = end
        return indices

    def next_batch(self) -> list[ChatExample]:
        while True:
            try:
                indices = self._next_indices_once()
                return [self._dataset[idx] for idx in indices]
            except StopIteration:
                if not self._repeat:
                    raise
            self._epoch += 1
            self._position = 0
            self._reshuffle()


class SupervisedBatchIteratorCore:
    def __init__(
        self,
        *,
        dataset: list[ChatExample],
        tokenizer: SupportsPosttrainTokenizer,
        batch_size: int,
        max_seq_len: int,
        crop_policy: str,
        pad_to_multiple_of: int,
        seed: int,
        shuffle: bool,
        repeat: bool,
        drop_last: bool,
        encode_supervised_batch_fn: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        self._tokenizer = tokenizer
        self._max_seq_len = int(max_seq_len)
        self._crop_policy = str(crop_policy)
        self._pad_to_multiple_of = int(pad_to_multiple_of)
        self._encode_supervised_batch_fn = encode_supervised_batch_fn
        self._iter = ShuffledBatchIterator(
            dataset=list(dataset),
            batch_size=int(batch_size),
            seed=int(seed),
            shuffle=bool(shuffle),
            repeat=bool(repeat),
            drop_last=bool(drop_last),
        )

    def state_dict(self) -> StatePayload:
        return {
            "max_seq_len": int(self._max_seq_len),
            "crop_policy": str(self._crop_policy),
            "pad_to_multiple_of": int(self._pad_to_multiple_of),
            "iterator": self._iter.state_dict(),
        }

    def load_state_dict(self, state: StatePayload) -> None:
        if not isinstance(state, dict):
            raise TypeError("supervised iterator state must be a dict")
        max_seq_len = int(state.get("max_seq_len", -1))
        if max_seq_len != int(self._max_seq_len):
            raise ValueError(
                f"supervised iterator max_seq_len mismatch: {max_seq_len} != {self._max_seq_len}"
            )
        crop_policy = str(state.get("crop_policy", "") or "")
        if crop_policy != str(self._crop_policy):
            raise ValueError(
                f"supervised iterator crop_policy mismatch: {crop_policy!r} != {self._crop_policy!r}"
            )
        multiple = int(state.get("pad_to_multiple_of", -1))
        if multiple != int(self._pad_to_multiple_of):
            raise ValueError(
                "supervised iterator pad_to_multiple_of mismatch: "
                f"{multiple} != {self._pad_to_multiple_of}"
            )
        self._iter.load_state_dict(dict(state.get("iterator") or {}))

    def __iter__(self) -> SupervisedBatchIteratorCore:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        examples = self._iter.next_batch()
        return self._encode_supervised_batch_fn(
            tokenizer=self._tokenizer,
            examples=examples,
            max_seq_len=int(self._max_seq_len),
            crop_policy=str(self._crop_policy),
            pad_to_multiple_of=int(self._pad_to_multiple_of),
        )


__all__ = ["ShuffledBatchIterator", "SupervisedBatchIteratorCore"]
