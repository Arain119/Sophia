import torch


def _build_tiny_model():
    from ml.runtime.model.transformer import ModelArgs, Transformer

    args = ModelArgs(
        max_batch_size=1,
        max_seq_len=256,
        vocab_size=512,
        dim=128,
        n_layers=6,
        n_heads=8,
        head_dim=16,
        num_key_value_heads=2,
        rope_head_dim=16,
        ffn_hidden=384,
    )
    model = Transformer(args)
    model.eval()
    return model


@torch.inference_mode()
def test_cache_dump_prefix_roundtrip_recompute_matches_full_dump():
    from ml.runtime.generation.prefix_cache import PrefixSnapshot

    torch.manual_seed(0)
    model = _build_tiny_model()

    prompt_len = 120
    toks = torch.randint(0, int(model.args.vocab_size), (1, prompt_len), dtype=torch.long)

    logits_full, _last_hidden = model.forward_with_last_hidden(
        toks,
        start_pos=0,
        return_all_logits=False,
    )
    full_dump = model.cache_dump(device="cpu")

    prefix_dump = model.cache_dump_prefix(device="cpu")
    snap = PrefixSnapshot(
        prefix_len=prompt_len,
        cache=prefix_dump,
        replay_len=int(model.args.max_seq_len) * int(model.args.n_layers),
    )

    model2 = _build_tiny_model()
    model2.load_state_dict(model.state_dict(), strict=True)
    model2.cache_load(snap.cache)
    replay_len = min(int(snap.replay_len), int(prompt_len))
    replay_start = int(prompt_len) - int(replay_len)
    replay = toks[:, replay_start:prompt_len]
    logits_replay, _last_hidden2 = model2.recompute_state_cache(
        replay,
        start_pos=int(replay_start),
    )

    next_tok = torch.randint(0, int(model.args.vocab_size), (1, 1), dtype=torch.long)
    logits_next_1, _ = model.forward_with_last_hidden(
        next_tok,
        start_pos=prompt_len,
        return_all_logits=False,
    )
    logits_next_2, _ = model2.forward_with_last_hidden(
        next_tok,
        start_pos=prompt_len,
        return_all_logits=False,
    )

    assert torch.allclose(logits_full, logits_replay, atol=1e-4, rtol=1e-4)
    assert torch.allclose(logits_next_1, logits_next_2, atol=1e-4, rtol=1e-4)

    model3 = _build_tiny_model()
    model3.load_state_dict(model.state_dict(), strict=True)
    model3.cache_load(full_dump)
    logits_next_3, _ = model3.forward_with_last_hidden(
        next_tok,
        start_pos=prompt_len,
        return_all_logits=False,
    )
    assert torch.allclose(logits_next_1, logits_next_3, atol=1e-4, rtol=1e-4)
