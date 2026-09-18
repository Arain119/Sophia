from __future__ import annotations

from threading import RLock

import torch
from torch import nn

from ml.runtime import (
    RuntimeCacheSnapshot,
    resolve_runtime_host,
    resolve_runtime_lock,
)


class _RuntimeModel:
    def __init__(self) -> None:
        self.runtime = object()
        self.runtime_lock = object()
        self.runtime_host = self.runtime
        self.runtime_model = self


class _RuntimeWrapper:
    def __init__(self, runtime_model: object, runtime_host: object, runtime_lock: object) -> None:
        self.runtime_model = runtime_model
        self.runtime_host = runtime_host
        self.runtime_lock = runtime_lock


def test_resolve_runtime_prefers_explicit_contract() -> None:
    runtime_model = _RuntimeModel()
    wrapper = _RuntimeWrapper(
        runtime_model,
        runtime_model.runtime,
        runtime_model.runtime_lock,
    )

    assert resolve_runtime_host(wrapper) is runtime_model.runtime
    assert resolve_runtime_lock(wrapper) is runtime_model.runtime_lock


def test_resolve_runtime_accepts_runtime_model_directly() -> None:
    runtime_model = _RuntimeModel()

    assert resolve_runtime_host(runtime_model) is runtime_model.runtime
    assert resolve_runtime_lock(runtime_model) is runtime_model.runtime_lock


def test_decoder_runtime_mixin_delegates_generation_methods() -> None:
    from ml.modeling.decoder_runtime import (
        DecoderModelMixin,
        DecoderRecipeMixin,
        DecoderRuntimeMixin,
    )

    class _RuntimeModel:
        tok_embeddings = object()
        output = object()
        gradient_checkpointing = False
        gradient_checkpointing_exclude_first = 0
        gradient_checkpointing_exclude_last = 0

    class _RuntimeHost:
        def runtime_max_seq_len(self) -> int:
            return 128

        def replay_with_cache(
            self,
            input_ids,
            start_pos: int = 0,
            return_all_logits: bool = True,
        ):
            return ("replay", (input_ids,), {"start_pos": start_pos, "return_all_logits": return_all_logits})

        def forward_with_last_hidden(
            self,
            input_ids,
            start_pos: int = 0,
            return_all_logits: bool = True,
        ):
            return ("forward", (input_ids,), {"start_pos": start_pos, "return_all_logits": return_all_logits})

        def cache_dump(
            self,
            device: str = "cpu",
            *,
            cache_pos: int | None = None,
            batch_size: int | None = None,
        ):
            return ("dump", (), {"device": device, "cache_pos": cache_pos, "batch_size": batch_size})

        def cache_load(self, snapshot) -> None:
            return ("load", (snapshot,), {})

    class _Wrapper(
        DecoderModelMixin,
        DecoderRuntimeMixin,
        DecoderRecipeMixin,
    ):
        def __init__(self) -> None:
            self.model = _RuntimeModel()
            self.runtime = _RuntimeHost()
            self._model_runtime_lock = RLock()

    wrapper = _Wrapper()

    assert wrapper.runtime_model is wrapper.model
    assert wrapper.runtime_host is wrapper.runtime
    assert wrapper.runtime_lock is wrapper._model_runtime_lock
    assert wrapper.replay_with_cache(1, start_pos=2) == (
        "replay",
        (1,),
        {"start_pos": 2, "return_all_logits": True},
    )
    assert wrapper.forward_with_last_hidden(3, return_all_logits=True) == (
        "forward",
        (3,),
        {"start_pos": 0, "return_all_logits": True},
    )
    assert wrapper.cache_dump(device="cpu") == (
        "dump",
        (),
        {"device": "cpu", "cache_pos": None, "batch_size": None},
    )
    snapshot = RuntimeCacheSnapshot()
    assert wrapper.cache_load(snapshot) is None
    assert wrapper.runtime_max_seq_len() == 128


def test_decoder_runtime_mixin_to_rebuilds_runtime_buffers() -> None:
    from ml.modeling.decoder_runtime import (
        DecoderModelMixin,
        DecoderRecipeMixin,
        DecoderRuntimeMixin,
    )

    class _RuntimeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok_embeddings = nn.Embedding(8, 4)
            self.output = nn.Linear(4, 8, bias=False)
            self.register_buffer("freqs_cis", torch.ones(4, 2, 2), persistent=False)
            self.register_buffer(
                "complex_buf",
                torch.ones(2, dtype=torch.complex64),
                persistent=False,
            )
            self.alias = torch.zeros_like(self.freqs_cis)
            self.rebuild_calls = 0

    class _RuntimeHost:
        def __init__(self, model: _RuntimeModel) -> None:
            self.model = model

        def rebuild_runtime_buffers(self) -> None:
            self.model.rebuild_calls += 1
            self.model.alias = self.model.freqs_cis

    class _Wrapper(
        DecoderModelMixin,
        DecoderRuntimeMixin,
        DecoderRecipeMixin,
        nn.Module,
    ):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Cfg", (), {})()
            self.model = _RuntimeModel()
            self.runtime = _RuntimeHost(self.model)
            self._model_runtime_lock = RLock()

    wrapper = _Wrapper()

    wrapper.to(device="cpu")

    assert wrapper.model.rebuild_calls == 1
    assert wrapper.model.alias is wrapper.model.freqs_cis
    assert torch.is_complex(wrapper.model.complex_buf)
    assert wrapper.model.complex_buf.device.type == "cpu"


def test_decoder_runtime_mixin_load_state_dict_resets_cache_and_rebuilds() -> None:
    from ml.modeling.decoder_runtime import (
        DecoderModelMixin,
        DecoderRecipeMixin,
        DecoderRuntimeMixin,
    )

    class _RuntimeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok_embeddings = nn.Embedding(8, 4)
            self.output = nn.Linear(4, 8, bias=False)
            self.register_buffer("freqs_cis", torch.ones(4, 2, 2), persistent=False)
            self.reset_calls = 0
            self.rebuild_calls = 0
            self.alias = torch.zeros_like(self.freqs_cis)

    class _RuntimeHost:
        def __init__(self, model: _RuntimeModel) -> None:
            self.model = model

        def rebuild_runtime_buffers(self) -> None:
            self.model.rebuild_calls += 1
            self.model.alias = self.model.freqs_cis

        def reset_runtime_cache(self) -> None:
            self.model.reset_calls += 1

    class _Wrapper(
        DecoderModelMixin,
        DecoderRuntimeMixin,
        DecoderRecipeMixin,
        nn.Module,
    ):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Cfg", (), {})()
            self.model = _RuntimeModel()
            self.runtime = _RuntimeHost(self.model)
            self._model_runtime_lock = RLock()

    wrapper = _Wrapper()
    state_dict = wrapper.state_dict()

    wrapper.load_state_dict(state_dict)

    assert wrapper.model.reset_calls == 1
    assert wrapper.model.rebuild_calls == 1
    assert wrapper.model.alias is wrapper.model.freqs_cis


def test_decoder_runtime_model_mixin_reties_word_embeddings_after_setters() -> None:
    from ml.modeling.decoder_runtime import (
        DecoderModelMixin,
        DecoderRecipeMixin,
        DecoderRuntimeMixin,
    )

    class _RuntimeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok_embeddings = nn.Embedding(8, 4)
            self.output = nn.Linear(4, 8, bias=False)

    class _RuntimeHost:
        def rebuild_runtime_buffers(self) -> None:
            return None

        def reset_runtime_cache(self) -> None:
            return None

    class _Wrapper(
        DecoderModelMixin,
        DecoderRuntimeMixin,
        DecoderRecipeMixin,
        nn.Module,
    ):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Cfg", (), {"tie_word_embeddings": True})()
            self.model = _RuntimeModel()
            self.runtime = _RuntimeHost()
            self._model_runtime_lock = RLock()

    wrapper = _Wrapper()
    new_input = nn.Embedding(8, 4)
    new_output = nn.Linear(4, 8, bias=False)

    wrapper.set_input_embeddings(new_input)
    assert wrapper.model.output.weight is wrapper.model.tok_embeddings.weight

    wrapper.set_output_embeddings(new_output)
    assert wrapper.model.output.weight is wrapper.model.tok_embeddings.weight
