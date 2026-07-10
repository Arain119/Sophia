from __future__ import annotations

import importlib.util
import types

import pytest

from ml.errors import SophiaUsageError
if importlib.util.find_spec("torch") is None:
    pytest.skip("torch is not installed in this environment", allow_module_level=True)

from ml.runtime.inference import eval as eval_mod
from ml.runtime.inference.eval import generation_runtime as eval_generation_runtime
from ml.runtime.inference.eval import model_io as eval_model_io
from ml.runtime.inference.eval.runtime_config import (
    EvalRuntimeConfig,
    eval_runtime_config_from_args,
    resolve_eval_runtime_config,
)
from ml.runtime.generation import sample_next_token as runtime_sample_next_token


def test_eval_local_model_load_kwargs_use_dtype_object() -> None:
    kwargs = eval_mod.local_model_load_kwargs()

    assert kwargs["local_files_only"] is True
    assert kwargs["dtype"] == eval_mod.torch.float32
    assert "torch_dtype" not in kwargs


def test_init_model_missing_export_dir_exits_cleanly(tmp_path) -> None:
    config = EvalRuntimeConfig(
        export_dir="",
        save_dir=str(tmp_path / "out"),
        device="cpu",
    )

    with pytest.raises(SophiaUsageError, match="Unable to locate a local exported Sophia model"):
        eval_mod.init_model(config)


def test_init_model_uses_context_window_for_tokenizer(monkeypatch, tmp_path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "config.json").write_text("{}", encoding="utf-8")
    (export_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    config = EvalRuntimeConfig(
        export_dir=str(export_dir),
        save_dir=str(tmp_path / "out"),
        device="cpu",
    )

    model = types.SimpleNamespace(
        config=types.SimpleNamespace(max_position_embeddings=4096),
        eval=lambda: types.SimpleNamespace(to=lambda *_args, **_kwargs: model),
    )
    captured: dict[str, object] = {}

    def _load_model(*_args, **_kwargs):
        return model

    def _load_tokenizer(path: str, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(eval_model_io, "load_trainable_decoder", _load_model)
    monkeypatch.setattr(eval_model_io, "load_local_export_tokenizer", _load_tokenizer)

    _loaded_model, _tokenizer = eval_mod.init_model(config)

    assert captured["path"] == str(export_dir)
    assert captured["model_max_length"] == 4096
    assert captured["padding_side"] == "left"
    assert captured["truncation_side"] == "left"


def test_init_model_auto_detects_sophia_export_dir(monkeypatch, tmp_path) -> None:
    save_dir = tmp_path / "out"
    export_dir = save_dir / f"sophia_{int(eval_mod.SOPHIA_DIM)}_export"
    export_dir.mkdir(parents=True)
    (export_dir / "config.json").write_text("{}", encoding="utf-8")
    (export_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    config = EvalRuntimeConfig(
        export_dir="",
        save_dir=str(save_dir),
        device="cpu",
    )

    model = types.SimpleNamespace(
        config=types.SimpleNamespace(max_position_embeddings=4096),
        eval=lambda: types.SimpleNamespace(to=lambda *_args, **_kwargs: model),
    )
    captured: dict[str, object] = {}

    def _load_model(*_args, **kwargs):
        captured["export_dir"] = kwargs["export_dir"]
        return model

    def _load_tokenizer(path: str, **_kwargs):
        captured["tokenizer_dir"] = path
        return object()

    monkeypatch.setattr(eval_model_io, "load_trainable_decoder", _load_model)
    monkeypatch.setattr(eval_model_io, "load_local_export_tokenizer", _load_tokenizer)

    eval_mod.init_model(config)

    assert captured["export_dir"] == str(export_dir)
    assert captured["tokenizer_dir"] == str(export_dir)


def test_init_model_enables_runtime_cache_for_sophia(monkeypatch, tmp_path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "config.json").write_text("{}", encoding="utf-8")
    (export_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    model = types.SimpleNamespace(
        config=types.SimpleNamespace(use_cache=True, max_seq_len=4096),
        eval=lambda: types.SimpleNamespace(to=lambda *_args, **_kwargs: model),
    )

    def _load_model(*_args, **_kwargs):
        return model

    monkeypatch.setattr(eval_model_io, "load_trainable_decoder", _load_model)
    monkeypatch.setattr(
        eval_model_io,
        "load_local_export_tokenizer",
        lambda *_args, **_kwargs: object(),
    )

    loaded_model, _ = eval_mod.init_model(
        EvalRuntimeConfig(
            export_dir=str(export_dir),
            save_dir=str(tmp_path / "out"),
            device="cpu",
            use_cache=1,
        )
    )

    assert loaded_model.config.use_cache is True


def test_resolve_eval_runtime_config_rounds_history_and_normalizes_sampling(monkeypatch) -> None:
    monkeypatch.setattr("ml.runtime.inference.eval.runtime_config.require_cuda", lambda device: device)
    monkeypatch.setattr("ml.runtime.inference.eval.runtime_config.setup_torch_backends", lambda: None)
    seen_seed: dict[str, int] = {}
    monkeypatch.setattr(
        "ml.runtime.inference.eval.runtime_config.setup_seed",
        lambda seed: seen_seed.setdefault("seed", int(seed)),
    )

    resolved = resolve_eval_runtime_config(
        EvalRuntimeConfig(
            device="cpu",
            history_turns=5,
            seed=7,
            do_sample=False,
            temperature=0.0,
        )
    )

    assert resolved.device == "cpu"
    assert resolved.history_turns == 4
    assert seen_seed["seed"] == 7


def test_eval_runtime_config_from_args_accepts_mapping() -> None:
    config = eval_runtime_config_from_args(
        {
            "save_dir": "out",
            "export_dir": "export",
            "seed": 9,
            "max_new_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": 0,
            "top_k": 11,
            "use_cache": 1,
            "max_cache_len": 2048,
            "history_turns": 2,
            "device": "cpu",
            "mode": "manual",
        }
    )

    assert config.export_dir == "export"
    assert config.seed == 9
    assert config.do_sample is False
    assert config.top_k == 11
    assert config.mode == "manual"


def test_eval_sample_next_token_shares_runtime_sampling_semantics() -> None:
    logits = eval_mod.torch.tensor([[10.0, 9.0, 1.0, 0.0, -5.0]])

    eval_token = eval_generation_runtime.sample_next_token(
        logits.clone(),
        do_sample=False,
        temperature=1.0,
        top_p=0.5,
        top_k=0,
    )
    runtime_token = runtime_sample_next_token(
        logits.clone(),
        temperature=0.0,
        top_p=0.5,
        top_k=0,
    ).squeeze(-1)

    assert eval_mod.torch.equal(eval_token, runtime_token)


def test_generate_with_kv_cache_routes_unpadded_batch_through_runtime_generate() -> None:
    captured: dict[str, object] = {}

    def _fake_generate(model, input_ids, **kwargs):
        del model
        captured["input_ids"] = input_ids
        captured["kwargs"] = kwargs
        return [list(row) + [99] for row in input_ids]

    generated = eval_generation_runtime.generate_with_kv_cache(
        model=object(),
        input_ids=eval_mod.torch.tensor([[1, 2], [3, 4]], dtype=eval_mod.torch.long),
        attention_mask=None,
        max_new_tokens=1,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        eos_token_id=None,
        max_cache_len=8,
        runtime_generate_fn=_fake_generate,
    )

    assert captured["input_ids"] == [[1, 2], [3, 4]]
    assert tuple(generated.shape) == (2, 3)
    assert generated.tolist() == [[1, 2, 99], [3, 4, 99]]


def test_generate_with_kv_cache_routes_padded_rows_independently() -> None:
    calls: list[list[list[int]]] = []

    def _fake_generate(model, input_ids, **kwargs):
        del model, kwargs
        calls.append(input_ids)
        return [list(input_ids[0]) + [77]]

    generated = eval_generation_runtime.generate_with_kv_cache(
        model=object(),
        input_ids=eval_mod.torch.tensor([[0, 0, 5, 6], [7, 8, 0, 0]], dtype=eval_mod.torch.long),
        attention_mask=eval_mod.torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=eval_mod.torch.long),
        max_new_tokens=1,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        eos_token_id=9,
        max_cache_len=8,
        runtime_generate_fn=_fake_generate,
    )

    assert calls == [[[5, 6]], [[7, 8]]]
    assert generated.tolist() == [[5, 6, 77], [7, 8, 77]]
