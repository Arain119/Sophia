from __future__ import annotations

import pytest
import torch

from ml.runtime.model.transformer import ModelArgs, Transformer


def _model_args(**overrides: object) -> ModelArgs:
    base: dict[str, object] = {
        "max_batch_size": 2,
        "max_seq_len": 128,
        "vocab_size": 1024,
        "dim": 256,
        "n_layers": 2,
        "n_heads": 4,
        "head_dim": 32,
        "num_key_value_heads": 2,
        "rope_head_dim": 32,
        "ffn_hidden": 768,
    }
    base.update(overrides)
    return ModelArgs(**base)


def test_sophia_incremental_cache_matches_prefix_recompute_cpu() -> None:
    args = _model_args()
    baseline_model = Transformer(args).eval()
    state = baseline_model.state_dict()
    incremental_model = Transformer(args).eval()
    incremental_model.load_state_dict(state, strict=True)

    torch.manual_seed(123)
    tokens = torch.randint(0, args.vocab_size, (2, 25), dtype=torch.long)

    # Baseline: recompute from scratch for each prefix.
    baseline: list[torch.Tensor] = []
    for t in range(1, int(tokens.size(1)) + 1):
        baseline_model.reset_runtime_cache()
        logits, _ = baseline_model.forward_with_last_hidden(
            tokens[:, :t],
            start_pos=0,
            return_all_logits=False,
        )
        baseline.append(logits.detach())

    # Incremental: feed one token at a time with strict start_pos.
    inc: list[torch.Tensor] = []
    incremental_model.reset_runtime_cache()
    for pos in range(int(tokens.size(1))):
        logits, _ = incremental_model.forward_with_last_hidden(
            tokens[:, pos : pos + 1],
            start_pos=int(pos),
            return_all_logits=False,
        )
        inc.append(logits.detach())

    assert len(baseline) == len(inc)
    for a, b in zip(baseline, inc, strict=True):
        torch.testing.assert_close(a, b)


def test_sophia_incremental_chunking_matches_single_shot_cpu() -> None:
    args = _model_args()
    full_model = Transformer(args).eval()
    state = full_model.state_dict()
    chunked_model = Transformer(args).eval()
    chunked_model.load_state_dict(state, strict=True)

    torch.manual_seed(456)
    tokens = torch.randint(0, args.vocab_size, (2, 33), dtype=torch.long)

    one_shot, _ = full_model.replay_with_cache(tokens, start_pos=0, return_all_logits=True)
    one_shot = one_shot.detach()

    # Chunked prefill: split the prompt into two parts while keeping incremental cache semantics.
    split = 13
    _ = chunked_model.replay_with_cache(tokens[:, :split], start_pos=0, return_all_logits=False)
    chunked, _ = chunked_model.replay_with_cache(
        tokens[:, split:],
        start_pos=split,
        return_all_logits=True,
    )
    chunked = chunked.detach()

    torch.testing.assert_close(one_shot[:, split:, :], chunked)


def test_sophia_multi_token_incremental_forward_accepts_window_local_chunk_cpu() -> None:
    args = _model_args()
    model = Transformer(args).eval()
    baseline = Transformer(args).eval()
    baseline.load_state_dict(model.state_dict(), strict=True)
    tokens = torch.randint(0, args.vocab_size, (2, 9), dtype=torch.long)

    _ = baseline.replay_with_cache(tokens[:, :3], start_pos=0, return_all_logits=False)
    expected, _ = baseline.replay_with_cache(
        tokens[:, 3:9],
        start_pos=3,
        return_all_logits=True,
    )

    _ = model.replay_with_cache(tokens[:, :3], start_pos=0, return_all_logits=False)
    actual, _ = model.replay_with_cache(
        tokens[:, 3:9],
        start_pos=3,
        return_all_logits=True,
    )

    torch.testing.assert_close(expected, actual)


def test_sophia_multi_token_incremental_forward_rejects_cached_tail_cpu() -> None:
    args = _model_args()
    model = Transformer(args).eval()
    tokens = torch.randint(0, args.vocab_size, (2, 4), dtype=torch.long)

    with pytest.raises(ValueError, match="multi-token incremental forward is not supported"):
        model(tokens, start_pos=14)


def test_sophia_multi_token_incremental_forward_rejects_start_pos_cpu() -> None:
    args = _model_args()
    model = Transformer(args).eval()
    tokens = torch.randint(0, args.vocab_size, (2, 3), dtype=torch.long)

    with pytest.raises(ValueError, match="multi-token incremental forward is not supported"):
        model(tokens, start_pos=5)
