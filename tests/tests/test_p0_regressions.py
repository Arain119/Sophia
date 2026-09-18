import tempfile
import unittest
import json
import argparse
from pathlib import Path
from unittest import mock

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

from ml.errors import SophiaUsageError
from ml.training.pretrain import shard_builder as shard_builder_mod
from ml.training.pretrain.shard_builder import ShardWriter, encode_batches


class _FakeTokenizer:
    def __call__(self, texts, **kwargs):
        ids = [list(range(1, len(str(t)) + 1)) for t in texts]
        return {'input_ids': ids}


def _write_test_tokenizer_bundle(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    vocab = {
        "<|pad|>": 0,
        "<|unk|>": 1,
        "<s>": 2,
        "</s>": 3,
        "hello": 4,
        "world": 5,
        "测": 6,
        "试": 7,
    }
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<|unk|>",
                "pad_token": "<|pad|>",
            }
        ),
        encoding="utf-8",
    )
    (path / "chat_template.jinja").write_text("", encoding="utf-8")
    return path


def _iter_batches(batch_size: int, total_docs: int):
    docs = [f'doc_{i}' for i in range(int(total_docs))]
    batch = []
    for doc in docs:
        batch.append(doc)
        if len(batch) >= int(batch_size):
            yield list(batch)
            batch = []
    if batch:
        yield list(batch)


class TestP0Regressions(unittest.TestCase):
    def test_shard_builder_tokenizer_loads_offline_bundle_and_resolves_eos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tok_dir = _write_test_tokenizer_bundle(Path(tmp) / "tok")
            loaded = shard_builder_mod._load_tokenizer(str(tok_dir))

        self.assertEqual(int(loaded.eos_token_id), 3)
        encs = loaded.encode_batch(["hello", "测试"])
        self.assertEqual([len(enc.ids) for enc in encs], [5, 2])

    def test_shard_builder_repo_tokenizer_keeps_cjk_tokens(self) -> None:
        tok_dir = Path(__file__).resolve().parents[2] / "ml" / "modeling" / "text"
        loaded = shard_builder_mod._load_tokenizer(str(tok_dir))
        arr = shard_builder_mod._encode_batch_flat(
            loaded,
            ["测试测试测试", "你好，世界"],
            eos_id=int(loaded.eos_token_id),
            dtype=np.dtype("<i4"),
            max_doc_tokens=4096,
            min_chunk_tokens=1,
        )

        self.assertGreater(int(arr.size), 0)

    def test_shard_builder_config_from_args_normalizes_cli_namespace(self) -> None:
        config = shard_builder_mod.ShardBuilderConfig.from_args(
            argparse.Namespace(
                jsonl=["a.jsonl", "b.jsonl"],
                out_dir="dataset/pretrain_tokens",
                overwrite_output_dir=1,
                tokenizer_path="tok",
                out_dtype="int32",
                shard_size_tokens=4096,
                batch_texts=32,
                num_proc=4,
                max_docs=100,
                max_doc_chars=2000,
                min_chunk_chars=100,
                max_doc_tokens=2048,
                min_chunk_tokens=8,
                progress_every_docs=500,
            )
        )

        self.assertEqual(config.jsonl, ("a.jsonl", "b.jsonl"))
        self.assertTrue(bool(config.overwrite_output_dir))
        self.assertEqual(int(config.shard_size_tokens), 4096)
        self.assertEqual(int(config.progress_every_docs), 500)

    def test_shard_builder_config_from_args_accepts_mapping(self) -> None:
        config = shard_builder_mod.ShardBuilderConfig.from_args(
            {
                "jsonl": ["a.jsonl"],
                "out_dir": "dataset/pretrain_tokens",
                "overwrite_output_dir": 1,
                "tokenizer_path": "tok",
                "batch_texts": 16,
            }
        )

        self.assertEqual(config.jsonl, ("a.jsonl",))
        self.assertTrue(bool(config.overwrite_output_dir))
        self.assertEqual(config.tokenizer_path, "tok")
        self.assertEqual(int(config.batch_texts), 16)

    def test_encode_batches_respects_max_docs_strictly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = ShardWriter(tmp, shard_size_tokens=1024, dtype=np.dtype('<i4'))
            total_tokens, total_docs = encode_batches(
                batches=_iter_batches(batch_size=4, total_docs=9),
                writer=writer,
                tokenizer=_FakeTokenizer(),
                tokenizer_path=tmp,
                eos_id=2,
                dtype=np.dtype('<i4'),
                max_doc_tokens=0,
                min_chunk_tokens=1,
                num_proc=1,
                max_docs=5,
                progress_every_docs=0,
            )
            writer.close()

        self.assertEqual(total_docs, 5)
        self.assertGreater(total_tokens, 0)

    def test_encode_batches_without_max_docs_keeps_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = ShardWriter(tmp, shard_size_tokens=1024, dtype=np.dtype('<i4'))
            total_tokens, total_docs = encode_batches(
                batches=_iter_batches(batch_size=4, total_docs=9),
                writer=writer,
                tokenizer=_FakeTokenizer(),
                tokenizer_path=tmp,
                eos_id=2,
                dtype=np.dtype('<i4'),
                max_doc_tokens=0,
                min_chunk_tokens=1,
                num_proc=1,
                max_docs=0,
                progress_every_docs=0,
            )
            writer.close()

        self.assertEqual(total_docs, 9)
        self.assertGreater(total_tokens, 0)

    def test_shard_builder_rejects_non_empty_output_dir_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "dataset"
            out_dir.mkdir()
            (out_dir / "keep.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(SophiaUsageError, "out_dir is non-empty"):
                shard_builder_mod._prepare_staging_out_dir(
                    out_dir=str(out_dir),
                    overwrite_output_dir=False,
                )

    def test_shard_builder_promotes_staging_dir_without_touching_live_output_until_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "dataset"
            out_dir.mkdir()
            (out_dir / "manifest.json").write_text("live", encoding="utf-8")
            (out_dir / "shard_00000.bin").write_bytes(b"live")
            shard_builder_mod._write_output_dir_marker(str(out_dir))

            staging_dir = shard_builder_mod._prepare_staging_out_dir(
                out_dir=str(out_dir),
                overwrite_output_dir=True,
            )
            self.assertTrue((out_dir / "manifest.json").exists())

            staging_manifest = Path(staging_dir) / "manifest.json"
            staging_manifest.write_text("new", encoding="utf-8")
            (Path(staging_dir) / "shard_00000.bin").write_bytes(b"new")

            shard_builder_mod._promote_staging_out_dir(
                staging_out_dir=str(staging_dir),
                final_out_dir=str(out_dir),
                overwrite_output_dir=True,
            )

            self.assertEqual((out_dir / "manifest.json").read_text(encoding="utf-8"), "new")
            self.assertEqual((out_dir / "shard_00000.bin").read_bytes(), b"new")

    def test_shard_builder_restores_live_output_if_promote_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "dataset"
            out_dir.mkdir()
            (out_dir / "manifest.json").write_text("live", encoding="utf-8")
            shard_builder_mod._write_output_dir_marker(str(out_dir))

            staging_dir = shard_builder_mod._prepare_staging_out_dir(
                out_dir=str(out_dir),
                overwrite_output_dir=True,
            )
            staging_manifest = Path(staging_dir) / "manifest.json"
            staging_manifest.write_text("new", encoding="utf-8")

            real_replace = shard_builder_mod.os.replace
            calls = {"count": 0}

            def flaky_replace(src: str, dst: str) -> None:
                calls["count"] += 1
                if calls["count"] == 2:
                    raise OSError("boom")
                real_replace(src, dst)

            with mock.patch.object(shard_builder_mod.os, "replace", side_effect=flaky_replace):
                with self.assertRaisesRegex(OSError, "boom"):
                    shard_builder_mod._promote_staging_out_dir(
                        staging_out_dir=str(staging_dir),
                        final_out_dir=str(out_dir),
                        overwrite_output_dir=True,
                    )

            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertEqual((out_dir / "manifest.json").read_text(encoding="utf-8"), "live")

    def test_shard_builder_recovers_interrupted_backup_before_new_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "dataset"
            backup_dir = Path(shard_builder_mod._promotion_backup_dir(str(out_dir)))
            backup_dir.mkdir(parents=True)
            (backup_dir / "manifest.json").write_text("backup", encoding="utf-8")
            shard_builder_mod._write_output_dir_marker(str(backup_dir))

            with self.assertRaisesRegex(SophiaUsageError, "out_dir is non-empty"):
                shard_builder_mod._prepare_staging_out_dir(
                    out_dir=str(out_dir),
                    overwrite_output_dir=False,
                )
            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertFalse(backup_dir.exists())

    def test_shard_builder_refuses_overwrite_of_unmarked_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "dataset"
            out_dir.mkdir()
            (out_dir / "manifest.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(SophiaUsageError, "does not look like a Sophia token-shards dataset"):
                shard_builder_mod._prepare_staging_out_dir(
                    out_dir=str(out_dir),
                    overwrite_output_dir=True,
                )
