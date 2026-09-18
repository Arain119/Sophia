from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from ml.data.token_shards.shard_manifest import TokenShard, TokenShardManifest, save_manifest
from ml.training.pretrain.engine.data.pretrain import build_pretrain_data_iter


def _write_manifest(tmp_path, *, tokens: int) -> str:
    return _write_manifest_with_shards(tmp_path, shard_tokens=[int(tokens)])


def _write_manifest_with_shards(tmp_path, *, shard_tokens: list[int]) -> str:
    shards: list[TokenShard] = []
    offset = 0
    for idx, tokens in enumerate(shard_tokens):
        shard_path = tmp_path / f"shard_{idx:05d}.bin"
        np.arange(offset, offset + int(tokens), dtype=np.int32).tofile(shard_path)
        shards.append(TokenShard(path=os.path.basename(shard_path), tokens=int(tokens)))
        offset += int(tokens)
    manifest_path = tmp_path / "manifest.json"
    save_manifest(
        str(manifest_path),
        TokenShardManifest(
            dtype="int32",
            shards=tuple(shards),
            eos_token_id=2,
            tokenizer_sha1="",
        ),
    )
    return str(manifest_path)


def test_build_pretrain_data_iter_resume_state_continues_exact_batch(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path, tokens=64)
    kwargs = dict(
        manifest_path=manifest_path,
        seq_len=8,
        batch_size=0,
        seed=123,
        device=torch.device("cpu"),
        num_workers=0,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=None,
        shard_preload=0,
        shard_preload_bytes=0,
    )

    data_iter = build_pretrain_data_iter(**kwargs)
    _ = next(data_iter)["input_ids"].clone()
    state = data_iter.state_dict()
    second = next(data_iter)["input_ids"].clone()
    close_fn = getattr(data_iter, "close", None)
    if callable(close_fn):
        close_fn()

    resumed_iter = build_pretrain_data_iter(**kwargs, resume_state=state)
    resumed = next(resumed_iter)["input_ids"].clone()
    close_fn = getattr(resumed_iter, "close", None)
    if callable(close_fn):
        close_fn()

    assert torch.equal(second, resumed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shard_preload", -1, "shard_preload must be >= 0"),
        ("shard_preload_bytes", -1, "shard_preload_bytes must be >= 0"),
    ],
)
def test_build_pretrain_data_iter_rejects_negative_preload_config(
    tmp_path,
    field: str,
    value: int,
    message: str,
) -> None:
    manifest_path = _write_manifest(tmp_path, tokens=64)
    kwargs = dict(
        manifest_path=manifest_path,
        seq_len=8,
        batch_size=0,
        seed=123,
        device=torch.device("cpu"),
        num_workers=0,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=None,
        shard_preload=0,
        shard_preload_bytes=0,
    )
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        build_pretrain_data_iter(**kwargs)


def test_build_pretrain_data_iter_parallel_resume_state_continues_exact_batch(tmp_path) -> None:
    manifest_path = _write_manifest_with_shards(tmp_path, shard_tokens=[128, 128, 128])
    kwargs = dict(
        manifest_path=manifest_path,
        seq_len=8,
        batch_size=0,
        seed=123,
        device=torch.device("cpu"),
        num_workers=2,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=1,
        shard_preload=0,
        shard_preload_bytes=0,
    )

    data_iter = build_pretrain_data_iter(**kwargs)
    _ = next(data_iter)["input_ids"].clone()
    state = data_iter.state_dict()
    second = next(data_iter)["input_ids"].clone()
    close_fn = getattr(data_iter, "close", None)
    if callable(close_fn):
        close_fn()

    resumed_iter = build_pretrain_data_iter(**kwargs, resume_state=state)
    resumed = next(resumed_iter)["input_ids"].clone()
    close_fn = getattr(resumed_iter, "close", None)
    if callable(close_fn):
        close_fn()

    assert torch.equal(second, resumed)


def test_build_pretrain_data_iter_rejects_sequence_geometry_change(
    tmp_path,
) -> None:
    manifest_path = _write_manifest(tmp_path, tokens=256)
    base_iter = build_pretrain_data_iter(
        manifest_path=manifest_path,
        seq_len=8,
        batch_size=0,
        seed=123,
        device=torch.device("cpu"),
        num_workers=0,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=None,
        shard_preload=0,
        shard_preload_bytes=0,
    )
    _ = next(base_iter)["input_ids"].clone()
    state = base_iter.state_dict()
    close_fn = getattr(base_iter, "close", None)
    if callable(close_fn):
        close_fn()

    assert str(state["cursor_semantics"]) == "next_unread_window_index"
    assert int(state["seq_len"]) == 8
    assert int(state["batch_size"]) == 0

    with pytest.raises(ValueError, match="same sequence geometry"):
        build_pretrain_data_iter(
            manifest_path=manifest_path,
            seq_len=16,
            batch_size=0,
            seed=123,
            device=torch.device("cpu"),
            num_workers=0,
            dataloader_prefetch_factor=2,
            dataloader_persistent_workers=None,
            shard_preload=0,
            shard_preload_bytes=0,
            resume_state=state,
        )


def test_build_pretrain_data_iter_globally_shuffles_complete_windows(tmp_path) -> None:
    manifest_path = _write_manifest_with_shards(
        tmp_path,
        shard_tokens=[32, 32, 32, 32],
    )
    data_iter = build_pretrain_data_iter(
        manifest_path=manifest_path,
        seq_len=8,
        batch_size=0,
        seed=42,
        device=torch.device("cpu"),
        num_workers=0,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=None,
        shard_preload=0,
        shard_preload_bytes=0,
    )

    starts = [int(next(data_iter)["input_ids"][0]) for _ in range(16)]
    state = data_iter.state_dict()
    data_iter.close()

    assert sorted(starts) == list(range(0, 128, 8))
    assert len({start // 32 for start in starts[:8]}) > 1
    order = state["window_order"]
    assert torch.is_tensor(order)
    torch.testing.assert_close(
        torch.sort(order).values,
        torch.arange(16, dtype=torch.int32),
    )
    assert int(state["window_order_pos"]) == 16


def test_build_pretrain_data_iter_rejects_corrupt_window_permutation(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path, tokens=64)
    data_iter = build_pretrain_data_iter(
        manifest_path=manifest_path,
        seq_len=8,
        batch_size=0,
        seed=42,
        device=torch.device("cpu"),
        num_workers=0,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=None,
        shard_preload=0,
        shard_preload_bytes=0,
    )
    next(data_iter)
    state = data_iter.state_dict()
    data_iter.close()
    order = state["window_order"]
    assert torch.is_tensor(order)
    order[0] = order[1]

    with pytest.raises(ValueError, match="not a permutation"):
        build_pretrain_data_iter(
            manifest_path=manifest_path,
            seq_len=8,
            batch_size=0,
            seed=42,
            device=torch.device("cpu"),
            num_workers=0,
            dataloader_prefetch_factor=2,
            dataloader_persistent_workers=None,
            shard_preload=0,
            shard_preload_bytes=0,
            resume_state=state,
        )


def test_build_pretrain_data_iter_restores_initial_empty_window_order(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path, tokens=64)
    kwargs = dict(
        manifest_path=manifest_path,
        seq_len=8,
        batch_size=0,
        seed=123,
        device=torch.device("cpu"),
        num_workers=0,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=None,
        shard_preload=0,
        shard_preload_bytes=0,
    )
    base_iter = build_pretrain_data_iter(**kwargs)
    state = base_iter.state_dict()
    expected = next(base_iter)["input_ids"].clone()
    base_iter.close()

    resumed_iter = build_pretrain_data_iter(**kwargs, resume_state=state)
    resumed = next(resumed_iter)["input_ids"].clone()
    resumed_iter.close()

    torch.testing.assert_close(resumed, expected)
