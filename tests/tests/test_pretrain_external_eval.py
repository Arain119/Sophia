from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ml.training.pretrain import external_eval as mod
from ml.training.pretrain.external_eval import build_pretrain_external_evals


def _args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "external_eval_contamination_interval": 0,
        "save_interval": 100,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_pretrain_external_evals_are_contamination_scan() -> None:
    evals = build_pretrain_external_evals(
        args=_args(),
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        output_dir="out",
        device=SimpleNamespace(),
        max_steps=10,
    )

    intervals = {name: interval for name, interval, _fn in evals}

    assert intervals == {
        "pretrain_contamination_safe_rate": 10,
    }


def test_pretrain_external_eval_intervals_honor_explicit_overrides() -> None:
    evals = build_pretrain_external_evals(
        args=_args(
            external_eval_contamination_interval=50,
        ),
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        output_dir="out",
        device=SimpleNamespace(),
        max_steps=100,
    )

    intervals = {name: interval for name, interval, _fn in evals}

    assert intervals == {
        "pretrain_contamination_safe_rate": 50,
    }


def test_external_eval_counts_all_flagged_samples_but_previews_eight(monkeypatch) -> None:
    class _Dataset:
        def __iter__(self):
            return self

        def __next__(self):
            return {"input_ids": torch.tensor([1, 2, 3])}

    monkeypatch.setattr(mod, "TokenStreamDataset", lambda **_kwargs: _Dataset())
    monkeypatch.setattr(mod.manifest_policy, "resolve_manifest_path", lambda path: path)
    monkeypatch.setattr(mod, "is_poison_repetitive_text", lambda _text: True)
    monkeypatch.setattr(mod, "has_repeated_sentences", lambda _text: False)
    monkeypatch.setattr(mod, "normalize_text", lambda text: text)

    result = mod._sample_split_contamination(
        split_name="train",
        split_path="data",
        tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: "flagged"),
        seed=1,
        sample_count=10,
        sample_seq_len=8,
    )

    assert result["samples"] == 10
    assert result["flagged_count"] == 10
    assert result["flagged_samples"] == 10
    assert result["safe_rate"] == 0.0
    assert len(result["examples"]) == 8


def test_pretrain_external_eval_rejects_negative_interval() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        build_pretrain_external_evals(
            args=_args(external_eval_contamination_interval=-1),
            model=SimpleNamespace(),
            tokenizer=SimpleNamespace(),
            output_dir="out",
            device=SimpleNamespace(),
            max_steps=100,
        )
