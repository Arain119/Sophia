"""Guard for the shared default tokenizer-dir resolver.

Shard audit / alignment / decode tooling all fall back to the in-package
tokenizer bundle via ``default_modeling_tokenizer_dir``; this pins that it points
at a real bundle and is resolved relative to the package (not the repo root).
"""

from __future__ import annotations

import os

from ml.data.token_shards.tokenizer_fingerprint import (
    TOKENIZER_JSON,
    default_modeling_tokenizer_dir,
    resolve_matching_tokenizer_dir,
)


def test_default_modeling_tokenizer_dir_points_at_real_bundle() -> None:
    tokenizer_dir = default_modeling_tokenizer_dir()
    assert os.path.isabs(tokenizer_dir)
    assert tokenizer_dir.replace("\\", "/").endswith("ml/modeling/text")
    assert os.path.isdir(tokenizer_dir)
    assert os.path.isfile(os.path.join(tokenizer_dir, TOKENIZER_JSON))


def test_resolve_matching_tokenizer_dir_errors_without_any_candidate(tmp_path) -> None:
    manifest_path = tmp_path / "train" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")

    try:
        resolve_matching_tokenizer_dir(
            manifest_path=str(manifest_path),
            expected_sha1="deadbeef",
            explicit_tokenizer_dir="",
            default_tokenizer_dir="",
        )
    except ValueError as exc:
        assert "Unable to resolve a tokenizer bundle" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected resolve_matching_tokenizer_dir to fail")
