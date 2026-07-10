import json
import os
import tempfile
import unittest

from ml.errors import SophiaUsageError
from ml.data.token_shards.shard_manifest import (
    TokenShard,
    TokenShardManifest,
    load_manifest,
    save_manifest,
    validate_tokenizer_fingerprint,
)
from ml.training.pretrain.shard_builder import _resolve_out_dtype
from ml.data.token_shards.tokenizer_fingerprint import (
    CHAT_TEMPLATE_JINJA,
    TOKENIZER_JSON,
    TOKENIZER_CONFIG_JSON,
    compute_tokenizer_bundle_sha1,
    preferred_fingerprint,
    tokenizer_bundle_matches_sha1,
)


class TestShardManifest(unittest.TestCase):
    def test_manifest_roundtrip_and_optional_sha1(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manifest_path = os.path.join(td, "manifest.json")
            shard = TokenShard(path="shard_00000.bin", tokens=16)
            m = TokenShardManifest(
                dtype="int32", shards=(shard,), eos_token_id=2, tokenizer_sha1=""
            )
            save_manifest(manifest_path, m)
            loaded = load_manifest(manifest_path)
            self.assertEqual(loaded.dtype, "int32")
            self.assertEqual(len(loaded.shards), 1)
            self.assertEqual(loaded.shards[0].path, "shard_00000.bin")
            self.assertEqual(int(loaded.shards[0].tokens), 16)
            self.assertEqual(loaded.eos_token_id, 2)
            self.assertEqual(loaded.tokenizer_sha1, "")

    def test_manifest_rejects_non_int32_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manifest_path = os.path.join(td, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dtype": "invalid_dtype",
                        "eos_token_id": 2,
                        "tokenizer_sha1": "",
                        "shards": [{"path": "shard_00000.bin", "tokens": 16}],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            with self.assertRaisesRegex(ValueError, "only supports int32"):
                load_manifest(manifest_path)

    def test_resolve_out_dtype_rejects_non_int32_values(self) -> None:
        self.assertEqual(_resolve_out_dtype(out_dtype="int32", vocab_size=32000).str, "<i4")
        for dtype in ("", "INT32", "invalid_dtype", "i4"):
            with self.assertRaisesRegex(SophiaUsageError, "only supports int32 token shards"):
                _resolve_out_dtype(out_dtype=dtype, vocab_size=32000)

    def test_validate_tokenizer_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tok_dir = os.path.join(td, "tok")
            os.makedirs(tok_dir, exist_ok=True)
            with open(os.path.join(tok_dir, TOKENIZER_JSON), "wb") as f:
                f.write(b"dummy-tokenizer-json")
            with open(os.path.join(tok_dir, TOKENIZER_CONFIG_JSON), "wb") as f:
                f.write(b"{}")
            with open(os.path.join(tok_dir, CHAT_TEMPLATE_JINJA), "wb") as f:
                f.write(b"{{ bos_token }}")
            expected = compute_tokenizer_bundle_sha1(tok_dir)
            validate_tokenizer_fingerprint(
                tokenizer_dir=tok_dir, expected_sha1=expected
            )
            with self.assertRaises(ValueError):
                validate_tokenizer_fingerprint(
                    tokenizer_dir=tok_dir, expected_sha1="0" * 40
                )


class TestTokenizerFingerprint(unittest.TestCase):
    def test_preferred_fingerprint_requires_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(td, exist_ok=True)
            open(os.path.join(td, TOKENIZER_JSON), "wb").close()
            open(os.path.join(td, TOKENIZER_CONFIG_JSON), "wb").close()
            open(os.path.join(td, CHAT_TEMPLATE_JINJA), "wb").close()
            fp = preferred_fingerprint(td)
            self.assertEqual(fp.label, TOKENIZER_JSON)

    def test_tokenizer_bundle_matches_sha1_returns_false_on_oserror(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(td, exist_ok=True)
            with open(os.path.join(td, TOKENIZER_JSON), "wb") as f:
                f.write(b"x")
            with open(os.path.join(td, TOKENIZER_CONFIG_JSON), "wb") as f:
                f.write(b"x")
            with open(os.path.join(td, CHAT_TEMPLATE_JINJA), "wb") as f:
                f.write(b"x")
            os.remove(os.path.join(td, TOKENIZER_JSON))
            os.mkdir(os.path.join(td, TOKENIZER_JSON))

            self.assertFalse(
                tokenizer_bundle_matches_sha1(
                    tokenizer_dir=td,
                    expected_sha1="0" * 40,
                )
            )


class TestPackedManifestMeta(unittest.TestCase):
    def test_load_manifest_accepts_extra_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "manifest.json")
            payload = {
                "format": "packed",
                "seq_len": 8,
                "pack_factor": 2,
                "dtype": "int32",
                "pad_token_id": 0,
                "eos_token_id": 2,
                "tokenizer_sha1": "",
                "total_sequences": 1,
                "shards": [{"path": "shard_00000.bin", "tokens": 16}],
            }
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            loaded = load_manifest(p)
            self.assertEqual(loaded.dtype, "int32")
            self.assertEqual(len(loaded.shards), 1)
