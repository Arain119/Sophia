from __future__ import annotations

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from ml.modeling.text.tokenizer_local import require_local_tokenizer_dir
from ml.integrations.adapters.hf.tokenizer_template import (
    RuntimeTokenizer,
    ensure_pad_token,
    set_model_max_length,
)


def load_local_tokenizer(
    tokenizer_path: str,
    *,
    padding_side: str | None = None,
    truncation_side: str | None = None,
    model_max_length: int | None = None,
) -> RuntimeTokenizer:
    resolved = require_local_tokenizer_dir(tokenizer_path)

    # Bandit B615: `resolved` is an explicitly local tokenizer bundle path.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(  # nosec B615
        resolved,
        local_files_only=True,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise RuntimeError(f"Unexpected tokenizer type: {type(tokenizer)!r}")

    if isinstance(padding_side, str) and padding_side:
        tokenizer.padding_side = str(padding_side)
    if isinstance(truncation_side, str) and truncation_side:
        tokenizer.truncation_side = str(truncation_side)

    ensure_pad_token(tokenizer)
    if model_max_length is not None:
        set_model_max_length(tokenizer, model_max_length=int(model_max_length))
    return RuntimeTokenizer(tokenizer)


def tokenizer_vocab_size(tokenizer: object) -> int:
    value = getattr(tokenizer, "vocab_size", None)
    if isinstance(value, int) and value > 0:
        return int(value)
    try:
        return int(len(tokenizer))
    except (AttributeError, TypeError):
        return 0


__all__ = ["RuntimeTokenizer", "load_local_tokenizer", "tokenizer_vocab_size"]
