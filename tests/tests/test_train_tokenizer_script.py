from __future__ import annotations

from pathlib import Path

import pytest

from ml.tooling.scripts.data import train_tokenizer


def test_resolve_limit_args_refuses_unbounded_without_opt_in() -> None:
    parser = train_tokenizer.build_parser()
    args = parser.parse_args(
        [
            "--pretrain_max_bytes",
            "0",
            "--sft_max_bytes",
            "1",
            "--max_text_chars",
            "1",
        ]
    )

    with pytest.raises(SystemExit, match="refusing unbounded tokenizer training"):
        train_tokenizer._resolve_limit_args(args)


def test_iter_training_texts_keeps_separate_pretrain_and_sft_budgets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        train_tokenizer,
        "_iter_parquet_texts",
        lambda _root: iter(["a" * 10, "b" * 10]),
    )
    monkeypatch.setattr(
        train_tokenizer,
        "_iter_sft_texts",
        lambda _root: iter(["c" * 10, "d" * 10]),
    )
    stats = train_tokenizer.TrainingTextStats()

    texts = list(
        train_tokenizer.iter_training_texts(
            pretrain_dir=tmp_path / "pretrain",
            sft_dir=tmp_path / "sft",
            pretrain_max_texts=0,
            sft_max_texts=0,
            pretrain_max_bytes=15,
            sft_max_bytes=12,
            max_text_chars=6,
            stats=stats,
        )
    )

    assert texts == ["a" * 6, "b" * 6, "c" * 6, "d" * 6]
    assert stats.pretrain_texts == 2
    assert stats.pretrain_bytes == 12
    assert stats.sft_texts == 2
    assert stats.sft_bytes == 12
    assert stats.total_texts == 4
    assert stats.truncated_texts == 4


def test_iter_stratified_training_texts_allocates_bucket_budgets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pretrain_dir = tmp_path / "pretrain"
    sft_dir = tmp_path / "sft"
    p_zh = pretrain_dir / "train" / "books" / "zh" / "a.parquet"
    p_code = pretrain_dir / "train" / "code" / "python" / "b.parquet"
    s_train = sft_dir / "train.jsonl"

    monkeypatch.setattr(
        train_tokenizer,
        "_pretrain_bucket_files",
        lambda _root: {
            "pretrain/books/zh": [p_zh],
            "pretrain/code/python": [p_code],
        },
    )
    monkeypatch.setattr(
        train_tokenizer,
        "_sft_bucket_files",
        lambda _root: {"sft/train": [s_train]},
    )
    monkeypatch.setattr(
        train_tokenizer,
        "_iter_round_robin_texts",
        lambda files, *, kind: iter(
            ["z" * 10, "z" * 10]
            if files == [p_zh]
            else (["p" * 10, "p" * 10] if files == [p_code] else ["s" * 10, "s" * 10])
        ),
    )
    stats = train_tokenizer.TrainingTextStats()

    texts = list(
        train_tokenizer.iter_stratified_training_texts(
            pretrain_dir=pretrain_dir,
            sft_dir=sft_dir,
            pretrain_max_texts=0,
            sft_max_texts=0,
            pretrain_max_bytes=20,
            sft_max_bytes=10,
            max_text_chars=0,
            bucket_min_bytes=0,
            stats=stats,
        )
    )

    assert texts == ["z" * 10, "p" * 10, "s" * 10]
    assert stats.pretrain_bytes == 20
    assert stats.sft_bytes == 10
    assert stats.buckets == {
        "pretrain/books/zh": {"texts": 1, "bytes": 10},
        "pretrain/code/python": {"texts": 1, "bytes": 10},
        "sft/train": {"texts": 1, "bytes": 10},
    }


def test_bucket_for_pretrain_file_uses_split_independent_leaf_path(tmp_path: Path) -> None:
    root = tmp_path / "pretrain"
    path = root / "train" / "code" / "source=Python" / "part.parquet"

    assert train_tokenizer._bucket_for_pretrain_file(path, root) == "pretrain/code/source=Python"


def test_allocate_bucket_budgets_uses_floor_then_proportional_remainder() -> None:
    budgets = train_tokenizer._allocate_bucket_budgets(
        100,
        {"small": 1, "large": 9},
        min_bucket_bytes=10,
    )

    assert budgets == {"large": 82, "small": 18}
