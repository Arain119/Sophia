from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch


class PretrainManifestShardLike(Protocol):
    path: str
    tokens: int


class PretrainManifestLike(Protocol):
    dtype: str
    shards: tuple[PretrainManifestShardLike, ...] | list[PretrainManifestShardLike]
    eos_token_id: int | None
    tokenizer_sha1: str

    @property
    def total_tokens(self) -> int: ...


class PretrainTokenizerLike(Protocol):
    model_max_length: int
    bos_token_id: int | None
    eos_token_id: int | None
    unk_token_id: int | None
    pad_token_id: int | None


PretrainBatchValue = torch.Tensor | bool
PretrainBatch = dict[str, PretrainBatchValue]


class PretrainDataIter(Protocol):
    def __iter__(self) -> PretrainDataIter: ...

    def __next__(self) -> PretrainBatch: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: dict[str, object]) -> None: ...

    def close(self) -> None: ...


PretrainDataIterFactory = Callable[[int, int, int, int, int], PretrainDataIter]


@dataclass(frozen=True)
class LoadedPretrainManifest:
    path: str
    manifest: PretrainManifestLike
    tokenizer_path: str = ""


@dataclass(frozen=True)
class LoadedPretrainTokenizer:
    tokenizer: PretrainTokenizerLike
    path: str
    seq_len: int


__all__ = [
    "LoadedPretrainManifest",
    "LoadedPretrainTokenizer",
    "PretrainBatch",
    "PretrainBatchValue",
    "PretrainDataIter",
    "PretrainDataIterFactory",
    "PretrainManifestLike",
    "PretrainManifestShardLike",
    "PretrainTokenizerLike",
]
