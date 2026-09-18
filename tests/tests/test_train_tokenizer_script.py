from __future__ import annotations

from pathlib import Path

import pytest

from ml.tooling.scripts.data import train_tokenizer


def test_repo_root_contains_project_metadata() -> None:
    root = train_tokenizer._repo_root()

    assert (root / "pyproject.toml").is_file()
    assert (root / "ml" / "modeling" / "text" / "chat_template.jinja").is_file()


def test_resolve_limit_args_refuses_unbounded_without_opt_in() -> None:
    parser = train_tokenizer.build_parser()
    args = parser.parse_args(
        [
            "--pretrain_max_bytes",
            "0",
            "--max_text_chars",
            "1",
        ]
    )

    with pytest.raises(SystemExit, match="refusing unbounded tokenizer training"):
        train_tokenizer._resolve_limit_args(args)


def test_iter_training_texts_applies_pretrain_budget(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        train_tokenizer,
        "_iter_parquet_texts",
        lambda _root: iter(["a" * 10, "b" * 10]),
    )
    stats = train_tokenizer.TrainingTextStats()

    texts = list(
        train_tokenizer.iter_training_texts(
            pretrain_dir=tmp_path / "pretrain",
            pretrain_max_texts=0,
            pretrain_max_bytes=15,
            max_text_chars=6,
            stats=stats,
        )
    )

    assert texts == ["a" * 6, "b" * 6]
    assert stats.pretrain_texts == 2
    assert stats.pretrain_bytes == 12
    assert stats.total_texts == 2
    assert stats.truncated_texts == 2


def test_iter_stratified_training_texts_allocates_bucket_budgets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pretrain_dir = tmp_path / "pretrain"
    p_zh = pretrain_dir / "train" / "books" / "zh" / "a.parquet"
    p_code = pretrain_dir / "train" / "code" / "python" / "b.parquet"

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
        "_iter_round_robin_texts",
        lambda files: iter(
            ["z" * 10, "z" * 10]
            if files == [p_zh]
            else ["p" * 10, "p" * 10]
        ),
    )
    stats = train_tokenizer.TrainingTextStats()

    texts = list(
        train_tokenizer.iter_stratified_training_texts(
            pretrain_dir=pretrain_dir,
            pretrain_max_texts=0,
            pretrain_max_bytes=20,
            max_text_chars=0,
            bucket_min_bytes=0,
            stats=stats,
        )
    )

    assert texts == ["z" * 10, "p" * 10]
    assert stats.pretrain_bytes == 20
    assert stats.buckets == {
        "pretrain/books/zh": {"texts": 1, "bytes": 10},
        "pretrain/code/python": {"texts": 1, "bytes": 10},
    }


def test_stratified_training_rejects_duplicate_buckets_across_roots(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        train_tokenizer,
        "_pretrain_bucket_files",
        lambda root: {"pretrain/shared": [root / "part.parquet"]},
    )
    stats = train_tokenizer.TrainingTextStats()

    with pytest.raises(ValueError, match="duplicate tokenizer source bucket"):
        list(
            train_tokenizer.iter_stratified_training_texts(
                pretrain_dir=tmp_path / "one",
                additional_pretrain_dirs=(tmp_path / "two",),
                pretrain_max_texts=0,
                pretrain_max_bytes=10,
                max_text_chars=0,
                bucket_min_bytes=0,
                stats=stats,
            )
        )


def test_stratified_training_requires_exact_bucket_budget_keys(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        train_tokenizer,
        "_pretrain_bucket_files",
        lambda _root: {"pretrain/source": [tmp_path / "part.parquet"]},
    )
    stats = train_tokenizer.TrainingTextStats()

    with pytest.raises(ValueError, match="must exactly match discovered buckets"):
        list(
            train_tokenizer.iter_stratified_training_texts(
                pretrain_dir=tmp_path,
                pretrain_max_texts=0,
                pretrain_max_bytes=10,
                max_text_chars=0,
                bucket_min_bytes=0,
                stats=stats,
                pretrain_bucket_budgets={"pretrain/other": 10},
            )
        )


def test_load_bucket_budgets_reads_positive_mapping(tmp_path: Path) -> None:
    path = tmp_path / "budgets.json"
    path.write_text(
        '{"bucket_budgets":{"pretrain/source":123}}',
        encoding="utf-8",
    )

    assert train_tokenizer._load_bucket_budgets(str(path)) == {
        "pretrain/source": 123
    }


def test_bucket_for_pretrain_file_uses_split_independent_leaf_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pretrain"
    path = root / "train" / "code" / "source=Python" / "part.parquet"

    assert (
        train_tokenizer._bucket_for_pretrain_file(path, root)
        == "pretrain/code/source=Python"
    )


def test_clean_corpus_layout_uses_source_bucket_and_train_only(tmp_path: Path) -> None:
    root = tmp_path / "clean"
    train = root / "source_zh" / "train" / "part.parquet"
    val = root / "source_zh" / "val" / "part.parquet"
    train.parent.mkdir(parents=True)
    val.parent.mkdir(parents=True)
    train.touch()
    val.touch()

    assert train_tokenizer._bucket_for_pretrain_file(train, root) == "pretrain/source_zh"
    assert train_tokenizer._is_pretrain_split_file(train, root, split="train")
    assert not train_tokenizer._is_pretrain_split_file(val, root, split="train")


def test_allocate_bucket_budgets_uses_floor_then_proportional_remainder() -> None:
    budgets = train_tokenizer._allocate_bucket_budgets(
        100,
        {"small": 1, "large": 9},
        min_bucket_bytes=10,
    )

    assert budgets == {"large": 82, "small": 18}
