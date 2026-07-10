from __future__ import annotations

from types import SimpleNamespace

import pytest

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
