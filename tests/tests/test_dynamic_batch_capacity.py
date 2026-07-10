from __future__ import annotations

import torch

from ml.runtime.model.transformer import ModelArgs, Transformer


def _model_args(**overrides: object) -> ModelArgs:
    base: dict[str, object] = {
        "max_batch_size": 1,
        "max_seq_len": 64,
        "vocab_size": 256,
        "dim": 128,
        "n_layers": 2,
        "n_heads": 4,
        "head_dim": 32,
        "num_key_value_heads": 2,
        "rope_head_dim": 16,
        "ffn_hidden": 384,
        "use_cache": True,
    }
    base.update(overrides)
    return ModelArgs(**base)


def test_runtime_model_grows_batch_capacity_cpu() -> None:
    args = _model_args()
    model = Transformer(args).eval()

    tokens = torch.randint(0, args.vocab_size, (3, 12), dtype=torch.long)
    logits = model(tokens, start_pos=0)

    assert logits.shape == (3, 12, args.vocab_size)
    assert int(model.args.max_batch_size) >= 3


def test_runtime_model_warm_growth_after_small_batch_cpu() -> None:
    args = _model_args()
    model = Transformer(args).eval()

    small = torch.randint(0, args.vocab_size, (1, 12), dtype=torch.long)
    large = torch.randint(0, args.vocab_size, (3, 12), dtype=torch.long)

    _ = model(small, start_pos=0)
    logits = model(large, start_pos=0)

    assert logits.shape == (3, 12, args.vocab_size)
    assert int(model.args.max_batch_size) >= 3


def test_model_args_runtime_controls_use_explicit_methods() -> None:
    args = ModelArgs(max_batch_size=1, max_seq_len=64)

    assert args.ensure_runtime_batch_capacity(3) == 3
    assert int(args.max_batch_size) == 3
    assert args.ensure_runtime_sequence_capacity(96) == 96
    assert int(args.max_seq_len) == 96
    assert args.runtime_rope_kwargs(max_seq_len=128)["max_seq_len"] == 128


def test_runtime_no_cache_model_full_sequence_eval_stays_stateless_cpu(
    monkeypatch,
) -> None:
    args = _model_args(use_cache=False)
    model = Transformer(args).eval()

    def _unexpected_capacity_call(*_args, **_kwargs):
        raise AssertionError("stateless full-sequence forward should not touch cache capacity")

    monkeypatch.setattr(model, "ensure_batch_capacity", _unexpected_capacity_call)
    for layer in model.layers:
        monkeypatch.setattr(layer.attn, "ensure_batch_capacity", _unexpected_capacity_call)

    tokens = torch.randint(0, args.vocab_size, (3, 12), dtype=torch.long)
    logits = model(tokens, start_pos=0)

    assert logits.shape == (3, 12, args.vocab_size)
    for layer in model.layers:
        assert torch.count_nonzero(layer.attn.kv_cache) == 0


def test_runtime_no_cache_model_replay_enables_cache_cpu() -> None:
    args = _model_args(use_cache=False)
    model = Transformer(args).eval()
    prompt = torch.randint(0, args.vocab_size, (1, 12), dtype=torch.long)

    logits, _ = model.replay_with_cache(prompt, start_pos=0, return_all_logits=False)

    assert logits.shape == (1, args.vocab_size)
    assert model.args.use_cache is True
    assert all(layer.attn.use_cache for layer in model.layers)
    assert int(model.layers[0].attn.kv_cache.size(0)) >= 1

    next_token = torch.randint(0, args.vocab_size, (1, 1), dtype=torch.long)
    step_logits, _ = model.forward_with_last_hidden(
        next_token,
        start_pos=12,
        return_all_logits=False,
    )
    assert step_logits.shape == (1, args.vocab_size)


def test_runtime_model_warm_growth_cache_roundtrip_cpu() -> None:
    args = _model_args()
    model = Transformer(args).eval()

    small = torch.randint(0, args.vocab_size, (1, 12), dtype=torch.long)
    large = torch.randint(0, args.vocab_size, (3, 12), dtype=torch.long)
    _ = model(small, start_pos=0)
    _ = model(large, start_pos=0)

    cache = model.cache_dump(device="cpu", cache_pos=12, batch_size=3)
    restored = Transformer(args).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.cache_load(cache)
    next_token = torch.randint(0, args.vocab_size, (3, 1), dtype=torch.long)
    logits, _ = restored.forward_with_last_hidden(
        next_token,
        start_pos=12,
        return_all_logits=False,
    )

    assert logits.shape == (3, args.vocab_size)
