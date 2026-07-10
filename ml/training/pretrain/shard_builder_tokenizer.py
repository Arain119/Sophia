from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ml.errors import SophiaUsageError
from ml.runtime.stack import (
    disable_optional_ml_backends,
    set_low_cpu_noise_env,
)

# Cap CPU noise (compiled libs / tokenizers).
set_low_cpu_noise_env()
# Disable optional Transformers backends (TF/Flax/JAX)
# to avoid importing large, often-misconfigured dependencies.
disable_optional_ml_backends()

import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from ml.data.token_shards.tokenizer_fingerprint import (  # noqa: E402
    TOKENIZER_JSON,
    preferred_fingerprint,
)


@dataclass(frozen=True)
class _OfflineTokenizerAdapter:
    tokenizer: Tokenizer
    eos_token_id: int

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    def get_vocab(self) -> dict[str, int]:
        return dict(self.tokenizer.get_vocab())

    def encode_batch(self, texts: Sequence[str]):
        return self.tokenizer.encode_batch(list(texts))


def _read_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        import json

        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to read tokenizer metadata: {path}") from exc
    if not isinstance(obj, dict):
        raise TypeError(f"tokenizer metadata must be a JSON object: {path}")
    return obj


def _resolve_special_token_string(*, tok_path: str, key: str) -> str:
    root = Path(str(tok_path))
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        payload = _read_json_file(root / name)
        if not isinstance(payload, dict):
            continue
        value = payload.get(str(key))
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            content = value.get("content")
            if isinstance(content, str) and content:
                return content
    return ""


def _resolve_eos_token_id(*, tok_path: str, tokenizer: Tokenizer) -> int:
    eos_token = _resolve_special_token_string(tok_path=str(tok_path), key="eos_token")
    if eos_token:
        eos_id = tokenizer.token_to_id(str(eos_token))
        if isinstance(eos_id, int) and eos_id >= 0:
            return int(eos_id)
    try:
        fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_path, local_files_only=True)  # nosec B615
    except Exception as exc:  # pragma: no cover - exercised only on malformed bundles
        raise RuntimeError(
            "Unable to resolve eos_token_id from tokenizer bundle.\n"
            f"tokenizer_dir={os.path.abspath(str(tok_path))}"
        ) from exc
    eos_id = getattr(fast_tokenizer, "eos_token_id", None)
    if not isinstance(eos_id, int):
        raise RuntimeError(f"tokenizer has no eos_token_id: {os.path.abspath(str(tok_path))}")
    return int(eos_id)


def _load_tokenizer(tok_path: str):
    """
    Load an offline tokenizer from `tok_path`.

    Sophia is pinned to tokenizer.json (fast tokenizer). Keep tokenizer.model alongside it for
    regeneration, but it is not used for runtime tokenization.
    """
    tok_path = os.path.abspath(str(tok_path))
    fp = preferred_fingerprint(tok_path)
    if fp.label == TOKENIZER_JSON:
        tokenizer = Tokenizer.from_file(fp.path)
        eos_id = _resolve_eos_token_id(tok_path=tok_path, tokenizer=tokenizer)
        return _OfflineTokenizerAdapter(tokenizer=tokenizer, eos_token_id=int(eos_id))
    raise RuntimeError(f"Unsupported tokenizer fingerprint: {fp.label!r}")


def _token_id_upper_bound(tokenizer) -> int:
    try:
        vocab_size = int(tokenizer.vocab_size)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("tokenizer vocab_size must be an integer") from exc
    if vocab_size <= 0:
        raise ValueError(f"tokenizer vocab_size must be > 0, got {vocab_size}")
    return vocab_size


def _resolve_out_dtype(*, out_dtype: str, vocab_size: int) -> np.dtype:
    del vocab_size
    out_dtype = str(out_dtype or "").strip()
    if out_dtype == "int32":
        return np.dtype("<i4")
    raise SophiaUsageError(
        f"[ERR] invalid --out_dtype: {out_dtype!r}. Sophia only supports int32 token shards."
    )


def _manifest_dtype_name(dtype: np.dtype) -> str:
    if dtype == np.dtype("<i4"):
        return "int32"
    raise SophiaUsageError(f"[ERR] unsupported output dtype: {dtype!r}")


__all__ = [
    "_OfflineTokenizerAdapter",
    "_load_tokenizer",
    "_manifest_dtype_name",
    "_read_json_file",
    "_resolve_eos_token_id",
    "_resolve_out_dtype",
    "_resolve_special_token_string",
    "_token_id_upper_bound",
]
