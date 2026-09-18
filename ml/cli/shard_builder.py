from __future__ import annotations

import argparse
from collections.abc import Sequence

from ml.cli.support import run_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build token shard dataset for Sophia pretraining"
    )
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--overwrite_output_dir",
        type=int,
        default=0,
        choices=[0, 1],
        help="Delete an existing Sophia token-shards output directory before promoting a fresh build.",
    )
    parser.add_argument(
        "--shard_size_tokens",
        type=int,
        default=20_000_000,
        help="Tokens per shard (approx)",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        default="int32",
        choices=["int32"],
        help=(
            "Output shard dtype. Sophia only supports the canonical int32 training format."
        ),
    )
    parser.add_argument(
        "--batch_texts", type=int, default=512, help="Documents per tokenizer batch"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Tokenizer worker processes",
    )
    parser.add_argument(
        "--progress_every_docs",
        type=int,
        default=200_000,
        help="Print progress every N docs",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=0,
        help="Optional cap on documents (0=unlimited)",
    )
    parser.add_argument(
        "--max_doc_chars",
        type=int,
        default=0,
        help="Optional: split long docs before tokenization.",
    )
    parser.add_argument(
        "--min_chunk_chars",
        type=int,
        default=200,
        help="Drop split chunks shorter than this many chars (requires --max_doc_chars>0).",
    )
    parser.add_argument(
        "--max_doc_tokens",
        type=int,
        default=0,
        help="Optional: split long docs by token length after tokenization (0 disables).",
    )
    parser.add_argument(
        "--min_chunk_tokens",
        type=int,
        default=1,
        help="When --max_doc_tokens>0: merge a short final chunk (<min_chunk_tokens) into the previous chunk.",
    )
    parser.add_argument("--jsonl", type=str, action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    from ml.training.pretrain.shard_builder import ShardBuilderConfig, run

    run(ShardBuilderConfig.from_args(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(run_cli(lambda: main()))
