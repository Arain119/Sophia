from __future__ import annotations

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

from ml.tooling.scripts.data.audit_token_shards import safe_decode


def _write_test_tokenizer_bundle(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    vocab = {
        "<|pad|>": 0,
        "<|unk|>": 1,
        "<s>": 2,
        "</s>": 3,
        "hello": 4,
        "world": 5,
        "测试": 6,
    }
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path / "tokenizer.json"))
    return path


def test_safe_decode_preserves_cjk_text(tmp_path: Path) -> None:
    tok_dir = _write_test_tokenizer_bundle(tmp_path / "tok")
    tokenizer = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    text = "hello 测试 world"
    ids = np.asarray(tokenizer.encode(text).ids, dtype=np.int64)

    decoded = safe_decode(tokenizer, ids)

    assert "测试" in decoded
    assert "Ġ" not in decoded
    assert "å" not in decoded
