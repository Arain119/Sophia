from __future__ import annotations

from typing import Protocol, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class SupportsPosttrainTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(
        self,
        token_ids: object,
        *,
        skip_special_tokens: bool = False,
    ) -> str: ...


__all__ = [
    "JsonObject",
    "JsonScalar",
    "JsonValue",
    "SupportsPosttrainTokenizer",
]
