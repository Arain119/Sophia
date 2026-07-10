from __future__ import annotations

from collections.abc import Mapping


def object_mapping(source: object) -> dict[str, object]:
    if isinstance(source, Mapping):
        return {str(key): value for key, value in source.items()}
    try:
        payload = vars(source)
    except TypeError as exc:
        raise TypeError(
            "expected a mapping or an object with a __dict__ for config projection"
        ) from exc
    return {str(key): value for key, value in dict(payload).items()}


__all__ = ["object_mapping"]
