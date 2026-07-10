from __future__ import annotations

from collections.abc import Mapping


_CONFIG_ATTRIBUTE_ALIASES: dict[str, tuple[str, ...]] = {
    "hidden_size": ("dim", "hidden_size"),
    "num_hidden_layers": ("n_layers", "num_hidden_layers"),
    "num_attention_heads": ("n_heads", "num_attention_heads"),
    "intermediate_size": ("ffn_hidden", "intermediate_size"),
    "max_position_embeddings": ("max_seq_len", "max_position_embeddings"),
}


def resolve_config_attribute(
    config: object | None,
    key: str,
) -> object | None:
    if config is None:
        return None
    for candidate in _CONFIG_ATTRIBUTE_ALIASES.get(str(key), (str(key),)):
        try:
            value = getattr(config, candidate)
        except Exception:
            continue
        if value is not None:
            return value
    if isinstance(config, Mapping):
        for candidate in _CONFIG_ATTRIBUTE_ALIASES.get(str(key), (str(key),)):
            value = config.get(candidate)
            if value is not None:
                return value
    return None


def resolve_config_num_hidden_layers(config: object | None) -> int | None:
    value = resolve_config_attribute(config, "num_hidden_layers")
    if isinstance(value, int) and value > 0:
        return int(value)
    return None


def resolve_config_max_position_embeddings(config: object | None) -> int | None:
    value = resolve_config_attribute(config, "max_position_embeddings")
    if isinstance(value, int) and 0 < int(value) < 1_000_000_000:
        return int(value)
    return None


__all__ = [
    "resolve_config_attribute",
    "resolve_config_max_position_embeddings",
    "resolve_config_num_hidden_layers",
]
