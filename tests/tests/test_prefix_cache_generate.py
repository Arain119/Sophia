from __future__ import annotations

import time
from threading import Barrier, Event, RLock, Thread

import torch
import pytest

from ml.runtime.generation.prefix_cache import (
    PrefixCache,
    PrefixSnapshot,
    model_prefix_cache_namespace,
)
from ml.runtime.generation import generate
from ml.runtime.generation.prefill import prefill_prompt
from ml.runtime.model.state import RuntimeCacheSnapshot
from ml.runtime.model.transformer import ModelArgs, Transformer


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


def test_prefix_cache_full_prompt_matches_plain_greedy_cpu() -> None:
    torch.manual_seed(1234)
    args = _tiny_args()
    model_plain = Transformer(args).eval()
    state = model_plain.state_dict()
    model_cached = Transformer(args).eval()
    model_cached.load_state_dict(state, strict=True)

    eos_id = 127
    max_new = 12
    prefill_chunk = 3

    prompt = [3, 7, 11, 20, 24, 30]

    # Plain generation (no prefix cache).
    out_plain = generate(
        model_plain,
        [prompt],
        max_new_tokens=max_new,
        eos_id=eos_id,
        temperature=0.0,
        prefill_chunk_size=prefill_chunk,
    )

    # Warm cache with the prompt so the same prompt can be replayed from cache.
    pc = PrefixCache(max_entries=64)
    _ = generate(
        model_cached,
        [prompt],
        max_new_tokens=max_new,
        eos_id=eos_id,
        temperature=0.0,
        prefix_cache=pc,
        prefix_cache_min_tokens=1,
        prefill_chunk_size=prefill_chunk,
    )

    snap = pc.get(prompt, namespace=model_prefix_cache_namespace(model_cached))
    assert snap is not None
    assert int(snap.prefix_len) == len(prompt)

    out_cached = generate(
        model_cached,
        [prompt],
        max_new_tokens=max_new,
        eos_id=eos_id,
        temperature=0.0,
        prefix_cache=pc,
        prefix_cache_min_tokens=1,
        prefill_chunk_size=prefill_chunk,
    )

    assert out_plain == out_cached


def test_prefix_cache_reuses_longest_cached_prefix_cpu() -> None:
    torch.manual_seed(1234)
    args = _tiny_args()
    model_plain = Transformer(args).eval()
    state = model_plain.state_dict()
    model_cached = Transformer(args).eval()
    model_cached.load_state_dict(state, strict=True)

    prefix_prompt = [3, 7, 11, 20]
    full_prompt = [3, 7, 11, 20, 24, 30]
    eos_id = 127
    max_new = 12

    cache = PrefixCache(max_entries=64)
    _ = generate(
        model_cached,
        [prefix_prompt],
        max_new_tokens=max_new,
        eos_id=eos_id,
        temperature=0.0,
        prefix_cache=cache,
        prefix_cache_min_tokens=1,
    )
    snapshot = cache.get(prefix_prompt, namespace=model_prefix_cache_namespace(model_cached))
    assert snapshot is not None
    assert cache.get_longest_prefix(
        full_prompt,
        namespace=model_prefix_cache_namespace(model_cached),
        min_tokens=1,
    )[0] == len(prefix_prompt)

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
        prefix_cache=cache,
        prefix_cache_min_tokens=1,
    )

    assert out_plain == out_cached


def test_generate_restores_training_state_cpu() -> None:
    model = Transformer(_tiny_args()).train()

    _ = generate(
        model,
        [[3, 7, 11]],
        max_new_tokens=2,
        eos_id=127,
        temperature=0.0,
    )

    assert model.training is True


def test_generate_keeps_eval_state_inside_runtime_lock_cpu() -> None:
    class _TrainingSensitiveModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._param = torch.nn.Parameter(torch.zeros(1))
            self.runtime_lock = RLock()
            self._first_entered = Event()
            self._release_first = Event()
            self._calls = 0

        def runtime_max_seq_len(self) -> int:
            return 16

        def reset_runtime_cache(self) -> None:
            return None

        def _logits(self, input_ids: torch.Tensor) -> torch.Tensor:
            token = 1 if not self.training else 2
            logits = torch.full(
                (int(input_ids.size(0)), 4),
                -1e9,
                dtype=torch.float32,
                device=input_ids.device,
            )
            logits[:, token] = 0.0
            return logits

        def replay_with_cache(self, input_ids, start_pos=0, return_all_logits=False):
            del start_pos
            self._calls += 1
            if self._calls == 1:
                self._first_entered.set()
                self._release_first.wait(timeout=5.0)
            logits = self._logits(input_ids)
            return (logits if not return_all_logits else logits[:, None, :]), None

        def forward_with_last_hidden(self, input_ids, start_pos=0, return_all_logits=False):
            return self.replay_with_cache(
                input_ids,
                start_pos=start_pos,
                return_all_logits=return_all_logits,
            )

    model = _TrainingSensitiveModel().train()
    results: dict[str, list[list[int]]] = {}
    errors: list[BaseException] = []

    def _run(name: str) -> None:
        try:
            results[name] = generate(
                model,
                [[5]],
                max_new_tokens=1,
                eos_id=None,
                temperature=0.0,
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread_a = Thread(target=_run, args=("a",))
    thread_b = Thread(target=_run, args=("b",))
    thread_a.start()
    assert model._first_entered.wait(timeout=2.0)
    thread_b.start()
    time.sleep(0.05)
    model._release_first.set()
    thread_a.join()
    thread_b.join()

    if errors:
        raise AssertionError(f"concurrent generate failed: {errors!r}") from errors[0]

    assert results["a"] == [[5, 1]]
    assert results["b"] == [[5, 1]]


def test_generate_first_token_matches_prompt_logit_cpu() -> None:
    torch.manual_seed(1234)
    model = Transformer(_tiny_args()).eval()

    prompt = [3, 7, 11, 19, 23]
    prompt_tensor = torch.tensor([prompt], dtype=torch.long)
    prompt_logits, _ = model.forward_with_last_hidden(
        prompt_tensor,
        start_pos=0,
        return_all_logits=False,
    )
    expected = int(prompt_logits.argmax(dim=-1).item())

    out = generate(
        model,
        [prompt],
        max_new_tokens=1,
        eos_id=127,
        temperature=0.0,
    )

    assert out[0][-1] == expected


def test_prefill_prompt_defaults_to_single_replay_when_no_override_cpu() -> None:
    class _Model:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[int, ...], int, bool]] = []

        def runtime_max_seq_len(self) -> int:
            return 4

        def replay_with_cache(self, input_ids, start_pos=0, return_all_logits=False):
            self.calls.append(
                (
                    tuple(int(token) for token in input_ids[0].tolist()),
                    int(start_pos),
                    bool(return_all_logits),
                )
            )
            logits = torch.randn(
                int(input_ids.size(0)),
                17,
                dtype=torch.float32,
                device=input_ids.device,
            )
            return logits, None

    model = _Model()
    input_ids = torch.tensor([[3, 7, 11, 19, 23, 29]], dtype=torch.long)

    logits = prefill_prompt(
        model,
        input_ids=input_ids,
        input_ids_list=None,
        model_device=torch.device("cpu"),
        prefix_cache=None,
        prefix_cache_min_tokens=1,
        prefix_cache_dir=None,
        prefill_chunk_size=None,
    )

    assert tuple(logits.shape) == (1, 17)
    assert model.calls == [((3, 7, 11, 19, 23, 29), 0, False)]


def test_prefill_prompt_honors_explicit_chunk_override_cpu() -> None:
    class _Model:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[int, ...], int, bool]] = []

        def runtime_max_seq_len(self) -> int:
            return 8

        def replay_with_cache(self, input_ids, start_pos=0, return_all_logits=False):
            self.calls.append(
                (
                    tuple(int(token) for token in input_ids[0].tolist()),
                    int(start_pos),
                    bool(return_all_logits),
                )
            )
            logits = torch.randn(
                int(input_ids.size(0)),
                17,
                dtype=torch.float32,
                device=input_ids.device,
            )
            return logits, None

    model = _Model()
    input_ids = torch.tensor([[3, 7, 11, 19, 23, 29]], dtype=torch.long)

    logits = prefill_prompt(
        model,
        input_ids=input_ids,
        input_ids_list=None,
        model_device=torch.device("cpu"),
        prefix_cache=None,
        prefix_cache_min_tokens=1,
        prefix_cache_dir=None,
        prefill_chunk_size=4,
    )

    assert tuple(logits.shape) == (1, 17)
    assert model.calls == [
        ((3, 7, 11, 19), 0, False),
        ((23, 29), 4, False),
    ]


def test_generate_resets_internal_cache_between_calls_cpu() -> None:
    torch.manual_seed(1234)
    args = _tiny_args()
    model_reused = Transformer(args).eval()
    state = model_reused.state_dict()

    _ = generate(
        model_reused,
        [[3, 7, 11, 19, 23, 29, 31, 37, 41, 43]],
        max_new_tokens=4,
        eos_id=127,
        temperature=0.0,
    )
    out_reused = generate(
        model_reused,
        [[5, 9, 13, 17, 21, 25]],
        max_new_tokens=4,
        eos_id=127,
        temperature=0.0,
    )

    model_fresh = Transformer(args).eval()
    model_fresh.load_state_dict(state, strict=True)
    out_fresh = generate(
        model_fresh,
        [[5, 9, 13, 17, 21, 25]],
        max_new_tokens=4,
        eos_id=127,
        temperature=0.0,
    )

    assert out_reused == out_fresh


def test_generate_rejects_overlong_request_cpu() -> None:
    model = Transformer(_tiny_args()).eval()
    prompt = list(range(60))

    try:
        _ = generate(
            model,
            [prompt],
            max_new_tokens=5,
            eos_id=127,
            temperature=0.0,
        )
    except ValueError as exc:
        assert "exceeds max_seq_len" in str(exc)
        return
    raise AssertionError("expected overlong request to be rejected")


def test_generate_stops_each_sequence_at_its_own_eos_cpu() -> None:
    class _BatchEosModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._param = torch.nn.Parameter(torch.zeros(1))
            self._step = 0

        def runtime_max_seq_len(self) -> int:
            return 16

        def reset_runtime_cache(self) -> None:
            self._step = 0

        def replay_with_cache(self, input_ids, start_pos=0, return_all_logits=False):
            table = [
                torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]], dtype=torch.float32, device=input_ids.device),
                torch.tensor([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]], dtype=torch.float32, device=input_ids.device),
            ]
            if int(start_pos) == 0:
                logits = torch.tensor(
                    [[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                    dtype=torch.float32,
                    device=input_ids.device,
                )
                self._step = 0
            else:
                logits = table[min(self._step, len(table) - 1)]
                self._step += 1
            return (logits if not return_all_logits else logits[:, None, :]), None

    model = _BatchEosModel().eval()
    out = generate(
        model,
        [[5], [6]],
        max_new_tokens=3,
        eos_id=2,
        temperature=0.0,
    )

    assert out == [[5, 1, 2], [6, 2]]
    assert all(row.count(2) == 1 for row in out)


def test_generate_rejects_negative_temperature_cpu() -> None:
    model = Transformer(_tiny_args()).eval()

    with pytest.raises(ValueError, match="temperature must be >= 0"):
        generate(
            model,
            [[3, 7, 11]],
            max_new_tokens=2,
            temperature=-0.1,
        )


def test_generate_rejects_negative_top_k_cpu() -> None:
    model = Transformer(_tiny_args()).eval()

    with pytest.raises(ValueError, match="top_k must be >= 0"):
        generate(
            model,
            [[3, 7, 11]],
            max_new_tokens=2,
            temperature=0.0,
            top_k=-1,
        )


def test_generate_rejects_non_rectangular_list_input_cpu() -> None:
    model = Transformer(_tiny_args()).eval()

    with pytest.raises(ValueError, match="rectangular"):
        generate(
            model,
            [[3, 7, 11], [5, 9]],
            max_new_tokens=1,
            temperature=0.0,
        )


def test_prefix_cache_requires_positive_capacity_cpu() -> None:
    with pytest.raises(ValueError, match="max_entries must be > 0"):
        PrefixCache(max_entries=0)


def test_prefix_cache_namespace_isolation_cpu() -> None:
    args_a = _tiny_args()
    args_b = _tiny_args()
    args_b.max_seq_len = 128
    model_a = Transformer(args_a).eval()
    model_b = Transformer(args_b).eval()
    cache = PrefixCache(max_entries=8)
    prompt = [3, 7, 11]

    snapshot = PrefixSnapshot(
        prefix_len=len(prompt),
        cache=RuntimeCacheSnapshot(),
        replay_len=2,
    )
    cache.put(prompt, snapshot, namespace=model_prefix_cache_namespace(model_a))

    assert cache.get(prompt, namespace=model_prefix_cache_namespace(model_a)) is not None
    assert cache.get(prompt, namespace=model_prefix_cache_namespace(model_b)) is None


def test_prefix_cache_isolated_across_same_shape_model_instances_cpu() -> None:
    model_a = Transformer(_tiny_args()).eval()
    model_b = Transformer(_tiny_args()).eval()
    cache = PrefixCache(max_entries=8)
    prompt = [3, 7, 11]
    snapshot = PrefixSnapshot(
        prefix_len=len(prompt),
        cache=RuntimeCacheSnapshot(),
        replay_len=2,
    )
    cache.put(prompt, snapshot, namespace=model_prefix_cache_namespace(model_a))

    assert cache.get(prompt, namespace=model_prefix_cache_namespace(model_a)) is not None
    assert cache.get(prompt, namespace=model_prefix_cache_namespace(model_b)) is None


def test_generate_tensor_input_without_prefix_cache_does_not_require_cpu_materialization_cpu(monkeypatch) -> None:
    model = Transformer(_tiny_args()).eval()
    prompt = torch.tensor([[3, 7, 11, 19]], dtype=torch.long)

    original_tolist = torch.Tensor.tolist
    calls: list[tuple[int, ...]] = []

    def _record_tolist(self):  # noqa: ANN001
        calls.append(tuple(int(v) for v in self.shape))
        return original_tolist(self)

    monkeypatch.setattr(torch.Tensor, "tolist", _record_tolist)
    out = generate(
        model,
        prompt,
        max_new_tokens=1,
        temperature=0.0,
    )

    assert len(out) == 1
    assert out[0][: int(prompt.size(1))] == [3, 7, 11, 19]
    # The decoder path should not emit extra small host transfers during prefix replay.
    # The remaining .tolist() call is the single output-token conversion
    # `(batch=1, prompt_len + max_new_tokens=5)`.
    host_visible_calls = [c for c in calls if len(c) != 1]
    assert host_visible_calls == [(1, 5)]


def test_prefix_cache_shared_writers_preserve_capacity_cpu() -> None:
    cache = PrefixCache(max_entries=3)
    snapshot = PrefixSnapshot(
        prefix_len=3,
        cache=RuntimeCacheSnapshot(),
        replay_len=2,
    )
    errors: list[BaseException] = []

    def _writer(offset: int) -> None:
        try:
            for idx in range(16):
                cache.put([offset, idx, idx + 1], snapshot, namespace="shared")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [Thread(target=_writer, args=(worker * 100,)) for worker in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        raise AssertionError(f"concurrent cache writes failed: {errors!r}") from errors[0]

    assert len(cache.cache) <= cache.max_entries
    assert len(cache._token_keys) <= cache.max_entries  # noqa: SLF001


def test_prefix_cache_invalid_cached_payload_fails_explicitly_cpu() -> None:
    model = Transformer(_tiny_args()).eval()
    cache = PrefixCache(max_entries=8)
    prompt = [3, 7, 11, 20, 24, 30]
    namespace = model_prefix_cache_namespace(model)
    bad_snapshot = PrefixSnapshot(
        prefix_len=len(prompt),
        replay_len=4,
        cache=RuntimeCacheSnapshot.from_payload(
            {
                "layer_0_kv": torch.zeros(
                    (1, 999, model.args.head_dim)
                )
            }
        ),
    )
    cache.put(
        prompt,
        bad_snapshot,
        namespace=namespace,
    )

    with pytest.raises(RuntimeError, match="prefix cache snapshot replay failed"):
        generate(
            model,
            [prompt],
            max_new_tokens=1,
            temperature=0.0,
            prefix_cache=cache,
            prefix_cache_min_tokens=1,
        )


def test_generate_rejects_prefix_cache_without_explicit_namespace_cpu() -> None:
    class _NoNamespaceModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.runtime_lock = RLock()
            self._param = torch.nn.Parameter(torch.zeros(1))

        def runtime_max_seq_len(self) -> int:
            return 16

        def reset_runtime_cache(self) -> None:
            return None

        def replay_with_cache(self, input_ids, start_pos=0, return_all_logits=False):
            del start_pos
            logits = torch.zeros(
                int(input_ids.size(0)),
                4,
                dtype=torch.float32,
                device=input_ids.device,
            )
            return (logits if not return_all_logits else logits[:, None, :]), None

    with pytest.raises(TypeError, match="prefix_cache_namespace"):
        generate(
            _NoNamespaceModel().eval(),
            [[1, 2, 3]],
            max_new_tokens=1,
            temperature=0.0,
            prefix_cache=PrefixCache(max_entries=4),
            prefix_cache_min_tokens=1,
        )


def test_generate_supports_concurrent_calls_on_same_runtime_instance_cpu() -> None:
    torch.manual_seed(1234)
    model = Transformer(_tiny_args()).eval()
    state = model.state_dict()
    prompt_a = [3, 7, 11, 19, 23]
    prompt_b = [5, 9, 13, 17, 21]
    barrier = Barrier(2)
    results: dict[str, list[list[int]]] = {}
    errors: list[BaseException] = []

    def _run(name: str, prompt: list[int]) -> None:
        try:
            barrier.wait(timeout=5)
            results[name] = generate(
                model,
                [prompt],
                max_new_tokens=4,
                eos_id=127,
                temperature=0.0,
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread_a = Thread(target=_run, args=("a", prompt_a))
    thread_b = Thread(target=_run, args=("b", prompt_b))
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    if errors:
        raise AssertionError(f"concurrent generate failed: {errors!r}") from errors[0]

    fresh_a = Transformer(_tiny_args()).eval()
    fresh_a.load_state_dict(state, strict=True)
    expected_a = generate(
        fresh_a,
        [prompt_a],
        max_new_tokens=4,
        eos_id=127,
        temperature=0.0,
    )

    fresh_b = Transformer(_tiny_args()).eval()
    fresh_b.load_state_dict(state, strict=True)
    expected_b = generate(
        fresh_b,
        [prompt_b],
        max_new_tokens=4,
        eos_id=127,
        temperature=0.0,
    )

    assert results["a"] == expected_a
    assert results["b"] == expected_b
