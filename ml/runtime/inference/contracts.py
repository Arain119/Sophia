from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class SupportsModelMaxLength(Protocol):
    model_max_length: int


class SupportsTokenizerEncodeDecode(Protocol):
    eos_token_id: int | None

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> Sequence[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str: ...


class InferenceTokenizer(SupportsModelMaxLength, SupportsTokenizerEncodeDecode, Protocol):
    pass


class SupportsConfiguredModel(Protocol):
    config: object | None


__all__ = [
    "InferenceTokenizer",
    "SupportsConfiguredModel",
    "SupportsModelMaxLength",
    "SupportsTokenizerEncodeDecode",
]
