from __future__ import annotations

import os
import pytest
import torch

from ml.runtime.generation.prefix_cache import (
    PrefixCache,
    PrefixCacheSnapshotError,
    PrefixSnapshot,
    model_prefix_cache_namespace,
)
from ml.runtime.generation import (
    generate,
    prefix_cache_path,
    save_snapshot_to_disk,
)
from ml.runtime.model.transformer import ModelArgs, Transformer
from ml.runtime.model.state import RuntimeCacheSnapshot


def _tiny_args() -> ModelArgs:
    return ModelArgs(
        max_batch_size=1,
        max_seq_len=64,
        vocab_size=128,
        dim=96,
        n_layers=2,
        n_heads=4,
        head_dim=24,
        num_key_value_heads=2,
        rope_head_dim=16,
        ffn_hidden=256,
    )


def test_disk_prefix_cache_full_prompt_matches_plain_greedy_cpu(tmp_path) -> None:
    torch.manual_seed(1234)
    args = _tiny_args()
    model_plain = Transformer(args).eval()
    state = model_plain.state_dict()
    model_cached = Transformer(args).eval()
    model_cached.load_state_dict(state, strict=True)

    eos_id = 127
    max_new = 12
    prefill_chunk = 3
    cache_dir = str(tmp_path)

    prompt = [3, 7, 11, 20, 24, 30]

    out_plain = generate(
        model_plain,
        [prompt],
        max_new_tokens=max_new,
        eos_id=eos_id,
        temperature=0.0,
        prefill_chunk_size=prefill_chunk,
    )

    # Populate disk cache with the prompt so it can be replayed on the next call.
    pc1 = PrefixCache(max_entries=8)
    _ = generate(
        model_cached,
        [prompt],
        max_new_tokens=max_new,
        eos_id=eos_id,
        temperature=0.0,
        prefix_cache=pc1,
        prefix_cache_min_tokens=1,
        prefix_cache_dir=cache_dir,
        prefill_chunk_size=prefill_chunk,
    )

    # New in-memory cache: should load the longest cached prefix from disk.
    pc2 = PrefixCache(max_entries=8)
    out_disk = generate(
        model_cached,
        [prompt],
        max_new_tokens=max_new,
        eos_id=eos_id,
        temperature=0.0,
        prefix_cache=pc2,
        prefix_cache_min_tokens=1,
        prefix_cache_dir=cache_dir,
        prefill_chunk_size=prefill_chunk,
    )

    assert out_plain == out_disk


def test_disk_prefix_cache_reuses_longest_cached_prefix_cpu(tmp_path) -> None:
    torch.manual_seed(1234)
    args = _tiny_args()
    model_plain = Transformer(args).eval()
    state = model_plain.state_dict()
    model_cached = Transformer(args).eval()
    model_cached.load_state_dict(state, strict=True)

    cache_dir = str(tmp_path)
    prefix_prompt = [3, 7, 11, 20]
    full_prompt = [3, 7, 11, 20, 24, 30]
    eos_id = 127
    max_new = 12

    _ = generate(
        model_cached,
        [prefix_prompt],
        max_new_tokens=max_new,
        eos_id=eos_id,
        temperature=0.0,
        prefix_cache=PrefixCache(max_entries=8),
        prefix_cache_min_tokens=1,
        prefix_cache_dir=cache_dir,
    )

    files_after_prefix = sorted(path.name for path in tmp_path.glob("*.pt"))
    assert len(files_after_prefix) == 1

    out_plain = generate(
        model_plain,
        [full_prompt],
        max_new_tokens=max_new,
        eos_id=eos_id,
        temperature=0.0,
    )
    out_cached = generate(
        model_cached,
        [full_prompt],
        max_new_tokens=max_new,
        eos_id=eos_id,
        temperature=0.0,
        prefix_cache=PrefixCache(max_entries=8),
        prefix_cache_min_tokens=1,
        prefix_cache_dir=cache_dir,
    )

    files_after_full = sorted(path.name for path in tmp_path.glob("*.pt"))
    assert len(files_after_full) == 2
    assert out_plain == out_cached


def test_disk_prefix_cache_generate_restores_training_state_cpu(tmp_path) -> None:
    model = Transformer(_tiny_args()).train()
    cache_dir = str(tmp_path)

    _ = generate(
        model,
        [[3, 7, 11]],
        max_new_tokens=2,
        eos_id=127,
        temperature=0.0,
        prefix_cache=PrefixCache(max_entries=8),
        prefix_cache_min_tokens=1,
        prefix_cache_dir=cache_dir,
    )

    assert model.training is True


def test_disk_prefix_cache_isolated_by_model_namespace_cpu(tmp_path) -> None:
    torch.manual_seed(1234)
    prompt = [3, 7, 11, 20, 24, 30]
    cache_dir = str(tmp_path)

    args_a = _tiny_args()
    args_b = _tiny_args()
    args_b.max_seq_len = 128
    model_a = Transformer(args_a).eval()
    model_b = Transformer(args_b).eval()

    _ = generate(
        model_a,
        [prompt],
        max_new_tokens=2,
        eos_id=127,
        temperature=0.0,
        prefix_cache=PrefixCache(max_entries=8),
        prefix_cache_min_tokens=1,
        prefix_cache_dir=cache_dir,
    )

    files_after_a = sorted(path.name for path in tmp_path.glob("*.pt"))
    assert len(files_after_a) == 1

    _ = generate(
        model_b,
        [prompt],
        max_new_tokens=2,
        eos_id=127,
        temperature=0.0,
        prefix_cache=PrefixCache(max_entries=8),
        prefix_cache_min_tokens=1,
        prefix_cache_dir=cache_dir,
    )

    files_after_b = sorted(path.name for path in tmp_path.glob("*.pt"))
    assert len(files_after_b) == 2
    assert files_after_b[0] != files_after_b[1]


def test_disk_prefix_cache_isolated_across_same_shape_model_instances_cpu(tmp_path) -> None:
    prompt = [3, 7, 11, 20, 24, 30]
    cache_dir = str(tmp_path)
    model_a = Transformer(_tiny_args()).eval()
    model_b = Transformer(_tiny_args()).eval()

    _ = generate(
        model_a,
        [prompt],
        max_new_tokens=2,
        eos_id=127,
        temperature=0.0,
        prefix_cache=PrefixCache(max_entries=8),
        prefix_cache_min_tokens=1,
        prefix_cache_dir=cache_dir,
    )
    _ = generate(
        model_b,
        [prompt],
        max_new_tokens=2,
        eos_id=127,
        temperature=0.0,
        prefix_cache=PrefixCache(max_entries=8),
        prefix_cache_min_tokens=1,
        prefix_cache_dir=cache_dir,
    )

    files = sorted(path.name for path in tmp_path.glob("*.pt"))
    assert len(files) == 2
    assert files[0] != files[1]


def test_disk_prefix_cache_corrupt_snapshot_fails_explicitly_cpu(tmp_path) -> None:
    model = Transformer(_tiny_args()).eval()
    prompt = [3, 7, 11]
    cache_file = prefix_cache_path(
        tmp_path,
        prompt,
        namespace=model_prefix_cache_namespace(model),
    )
    cache_file.write_bytes(b"not a torch payload")

    with pytest.raises(PrefixCacheSnapshotError, match="failed to load prefix cache snapshot"):
        generate(
            model,
            [prompt],
            max_new_tokens=1,
            temperature=0.0,
            prefix_cache_dir=str(tmp_path),
        )

    assert cache_file.exists()
    assert cache_file.stat().st_size > 0


def test_disk_prefix_cache_invalid_schema_fails_explicitly_cpu(tmp_path) -> None:
    model = Transformer(_tiny_args()).eval()
    prompt = [3, 7, 11]
    cache_file = prefix_cache_path(
        tmp_path,
        prompt,
        namespace=model_prefix_cache_namespace(model),
    )
    torch.save({"prefix_len": 3, "cache": {}}, cache_file)

    with pytest.raises(PrefixCacheSnapshotError, match="missing required keys"):
        generate(
            model,
            [prompt],
            max_new_tokens=1,
            temperature=0.0,
            prefix_cache_dir=str(tmp_path),
        )

    assert cache_file.exists()
    assert cache_file.stat().st_size > 0


def test_disk_prefix_cache_invalid_cache_payload_fails_explicitly_cpu(tmp_path) -> None:
    model = Transformer(_tiny_args()).eval()
    prompt = [3, 7, 11, 20, 24, 30]
    cache_file = prefix_cache_path(
        tmp_path,
        prompt,
        namespace=model_prefix_cache_namespace(model),
    )
    torch.save(
        {
            "prefix_len": len(prompt),
            "replay_len": 4,
            "cache": {
                "layer_0_kv": torch.zeros((1, 999, model.args.head_dim)),
            },
        },
        cache_file,
    )

    with pytest.raises(RuntimeError, match="prefix cache snapshot replay failed"):
        generate(
            model,
            [prompt],
            max_new_tokens=1,
            temperature=0.0,
            prefix_cache_dir=str(tmp_path),
        )

    assert cache_file.exists()
    assert cache_file.stat().st_size > 0


def test_disk_prefix_cache_atomic_save_uses_unique_temp_paths_cpu(tmp_path, monkeypatch) -> None:
    model = Transformer(_tiny_args()).eval()
    prompt = [3, 7, 11]
    snapshot = PrefixSnapshot(
        prefix_len=3,
        cache=RuntimeCacheSnapshot(),
        replay_len=2,
    )
    temp_paths: list[str] = []
    original_replace = os.replace

    def _record_replace(src, dst):  # noqa: ANN001
        temp_paths.append(str(src))
        return original_replace(src, dst)

    monkeypatch.setattr(os, "replace", _record_replace)

    save_snapshot_to_disk(
        tmp_path,
        prompt,
        snapshot,
        namespace=model_prefix_cache_namespace(model),
    )
    save_snapshot_to_disk(
        tmp_path,
        prompt,
        snapshot,
        namespace=model_prefix_cache_namespace(model),
    )

    assert len(temp_paths) == 2
    assert temp_paths[0] != temp_paths[1]
