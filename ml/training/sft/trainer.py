"""Model construction shared by the SFT and RL CLIs."""

from __future__ import annotations

import json
from pathlib import Path

from ml.modeling import SophiaDecoder, SophiaDecoderConfig


def _model_from_spec(
    spec_path: Path, *, tokenizer: object, batch_size: int
) -> SophiaDecoder:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    values = dict(spec.get("model") or {})
    values["max_batch_size"] = int(batch_size)
    tokenizer_vocab = int(getattr(tokenizer, "vocab_size", values.get("vocab_size", 0)))
    if tokenizer_vocab != int(values.get("vocab_size", 0)):
        raise RuntimeError("tokenizer and model spec vocab sizes differ")
    config = SophiaDecoderConfig.from_mapping(values)
    for name in ("bos_token_id", "eos_token_id", "unk_token_id", "pad_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int):
            setattr(config, name, value)
    config.return_logits_in_train = False
    config.use_cache = False
    return SophiaDecoder(config)
