from __future__ import annotations

import torch

import ml.runtime.generation.decode as decode_mod


def _greedy_logits(token_id: int, *, vocab_size: int = 8) -> torch.Tensor:
    logits = torch.full((1, vocab_size), -1e9, dtype=torch.float32)
    logits[:, int(token_id)] = 0.0
    return logits


def test_decode_from_prefill_grows_without_torch_cat_cpu(monkeypatch) -> None:
    class _Model:
        def replay_with_cache(self, _token, *, start_pos: int, return_all_logits: bool):
            del start_pos, return_all_logits
            return _greedy_logits(1), None

    prefill_logits = torch.stack(
        [_greedy_logits(1).squeeze(0), _greedy_logits(1).squeeze(0)],
        dim=0,
    )

    def _forbid_cat(*_args, **_kwargs):
        raise AssertionError("decode path should not call torch.cat for token growth")

    monkeypatch.setattr(decode_mod.torch, "cat", _forbid_cat)

    output = decode_mod.decode_from_prefill(
        model=_Model(),
        input_ids=torch.tensor([[3, 7], [11, 19]], dtype=torch.long),
        prefill_logits=prefill_logits,
        model_device=torch.device("cpu"),
        prompt_len=2,
        batch_size=2,
        max_new_tokens=3,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        eos_id=None,
    )

    assert output == [[3, 7, 1, 1, 1], [11, 19, 1, 1, 1]]
