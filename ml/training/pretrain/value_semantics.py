from __future__ import annotations


def is_unset(value: object) -> bool:
    return value is None or value == ""


def coerce_str(value: object, *, default: str = "") -> str:
    if is_unset(value):
        return str(default)
    return str(value)


def normalized_str(value: object, *, default: str = "") -> str:
    text = coerce_str(value, default=default).strip()
    return str(default) if text == "" else text


def coerce_int(value: object, *, default: int = 0) -> int:
    if is_unset(value):
        return int(default)
    return int(value)


def coerce_float(value: object, *, default: float = 0.0) -> float:
    if is_unset(value):
        return float(default)
    return float(value)


def coerce_optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def positive_int_or_default(value: object, *, default: int) -> int:
    resolved = coerce_int(value)
    return int(default) if resolved <= 0 else int(resolved)


def enabled_flag(value: object, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return bool(int(value) == 1)


def is_non_positive_int(value: object) -> bool:
    return coerce_int(value) <= 0


def is_disabled_flag(value: object) -> bool:
    return not enabled_flag(value)


__all__ = [
    "coerce_float",
    "coerce_int",
    "coerce_optional_float",
    "coerce_str",
    "enabled_flag",
    "is_disabled_flag",
    "is_non_positive_int",
    "is_unset",
    "normalized_str",
    "positive_int_or_default",
]
