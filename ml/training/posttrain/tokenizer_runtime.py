from __future__ import annotations

from ml.training.posttrain.contracts import SupportsPosttrainTokenizer


def resolve_tokenizer_pad_token_id(tokenizer: SupportsPosttrainTokenizer) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_token_id, int):
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int):
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    return int(eos_token_id)


__all__ = ["resolve_tokenizer_pad_token_id"]
