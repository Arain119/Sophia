import os
import tempfile

import numpy as np
import pytest
import torch

from ml.data.token_shards.shard_manifest import TokenShard, TokenShardManifest, save_manifest
from ml.data.token_shards import TokenStreamDataset


def _write_int32_shard(path: str, *, tokens: int) -> None:
    arr = np.arange(int(tokens), dtype=np.int32)
    arr.tofile(path)


def _write_single_shard_manifest(tmp_dir: str, *, tokens: int) -> str:
    shard_name = "shard_00000.bin"
    shard_path = os.path.join(tmp_dir, shard_name)
    _write_int32_shard(shard_path, tokens=int(tokens))

    manifest_path = os.path.join(tmp_dir, "manifest.json")
    manifest = TokenShardManifest(
        dtype="int32",
        shards=(TokenShard(path=shard_name, tokens=int(tokens)),),
        eos_token_id=2,
        tokenizer_sha1="",
    )
    save_manifest(manifest_path, manifest)
    return manifest_path


def test_token_stream_dataset_yields_shapes_and_monotonic_windows() -> None:
    with tempfile.TemporaryDirectory() as td:
        manifest_path = _write_single_shard_manifest(td, tokens=64)

        ds = TokenStreamDataset(manifest_path, seq_len=8, seed=123, batch_size=0)
        it = iter(ds)
        batch = next(it)
        raw_ids = batch["input_ids"]
        assert batch["labels"] is raw_ids
        ids = raw_ids.clone()
        it.close()
        assert isinstance(ids, torch.Tensor)
        assert ids.shape == (8,)
        assert bool(batch["labels_is_input_ids"]) is True
        assert ids.dtype == torch.int32
        assert torch.all(ids[1:] == ids[:-1] + 1)
        del raw_ids, batch, it, ds

        ds_batched = TokenStreamDataset(manifest_path, seq_len=8, seed=123, batch_size=2)
        it2 = iter(ds_batched)
        batch2 = next(it2)
        raw_ids2 = batch2["input_ids"]
        assert batch2["labels"] is raw_ids2
        ids2 = raw_ids2.clone()
        it2.close()
        assert isinstance(ids2, torch.Tensor)
        assert ids2.shape == (2, 8)
        assert bool(batch2["labels_is_input_ids"]) is True
        assert ids2.dtype == torch.int32
        flat = ids2.reshape(-1)
        assert torch.all(flat[1:] == flat[:-1] + 1)
        del raw_ids2, batch2, it2, ds_batched


def test_token_stream_dataset_state_dict_resume_continues_stream() -> None:
    with tempfile.TemporaryDirectory() as td:
        manifest_path = _write_single_shard_manifest(td, tokens=64)

        ds = TokenStreamDataset(manifest_path, seq_len=8, seed=123, batch_size=0)
        it = iter(ds)
        first = next(it)["input_ids"].clone()
        state = it.state_dict()
        second = next(it)["input_ids"].clone()
        it.close()

        ds_resume = TokenStreamDataset(manifest_path, seq_len=8, seed=123, batch_size=0)
        it_resume = iter(ds_resume)
        it_resume.load_state_dict(state)
        resumed = next(it_resume)["input_ids"].clone()
        it_resume.close()

        assert torch.equal(first[1:], first[:-1] + 1)
        assert torch.equal(second, resumed)


def test_token_stream_dataset_rejects_shards_shorter_than_block() -> None:
    with tempfile.TemporaryDirectory() as td:
        manifest_path = _write_single_shard_manifest(td, tokens=7)

        with pytest.raises(ValueError, match="block_tokens=8"):
            TokenStreamDataset(manifest_path, seq_len=8, seed=123, batch_size=0)


def test_token_stream_dataset_rejects_non_int32_manifest_dtype() -> None:
    with tempfile.TemporaryDirectory() as td:
        shard_name = "shard_00000.bin"
        shard_path = os.path.join(td, shard_name)
        np.arange(16, dtype=np.int32).tofile(shard_path)
        manifest_path = os.path.join(td, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(
                '{\n'
                '  "dtype": "invalid_dtype",\n'
                '  "eos_token_id": 2,\n'
                '  "tokenizer_sha1": "",\n'
                '  "shards": [{"path": "shard_00000.bin", "tokens": 16}]\n'
                '}\n'
            )

        with pytest.raises(ValueError, match="only supports int32"):
            TokenStreamDataset(manifest_path, seq_len=8, seed=123, batch_size=0)
