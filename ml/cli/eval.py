from __future__ import annotations

import argparse
from collections.abc import Sequence

from ml.cli.support import run_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sophia model inference / chat")
    parser.add_argument(
        "--save_dir",
        default="out",
        type=str,
        help="Search root for auto-detecting `pretrain_check/` first, then the canonical `sophia_<dim>_export/` bundle.",
    )
    parser.add_argument(
        "--export_dir",
        default="",
        type=str,
        help="Optional local Sophia export directory. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--seed",
        default=2026,
        type=int,
        help="Random seed. `-1` disables explicit seeding.",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=256,
        type=int,
        help="Maximum number of generated tokens. Clamped to the model context window.",
    )
    parser.add_argument(
        "--temperature",
        default=0.6,
        type=float,
        help="Sampling temperature. Ignored when `--do_sample 0`; `0` is accepted for greedy decoding.",
    )
    parser.add_argument(
        "--top_p",
        default=0.8,
        type=float,
        help="Top-p (nucleus) sampling threshold in `(0, 1]`.",
    )
    parser.add_argument(
        "--do_sample",
        default=1,
        type=int,
        choices=[0, 1],
        help="1=sample, 0=greedy.",
    )
    parser.add_argument(
        "--top_k",
        default=0,
        type=int,
        help="Top-k sampling. `0` disables it.",
    )
    parser.add_argument(
        "--use_cache",
        default=1,
        type=int,
        choices=[0, 1],
        help="Enable KV-cache incremental decoding.",
    )
    parser.add_argument(
        "--max_cache_len",
        default=0,
        type=int,
        help="Maximum KV-cache length in tokens. `0` uses `config.max_position_embeddings`.",
    )
    parser.add_argument(
        "--history_turns",
        dest="history_turns",
        default=0,
        type=int,
        help="Number of history turns to keep. Must be even; `0` disables history.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="Execution device. Use `cpu` for validation or `cuda:N` for accelerated inference.",
    )
    parser.add_argument(
        "--mode",
        default="ask",
        choices=["ask", "auto", "manual"],
        help="Run mode: `ask` prompts at startup, `auto` uses built-in prompts, `manual` reads interactive input.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parsed = build_parser().parse_args(argv)

    from ml.runtime.stack import ensure_standard_stack

    ensure_standard_stack(mode="infer")
    from ml.runtime.inference.eval import (
        eval_runtime_config_from_args,
        resolve_eval_runtime_config,
        run,
    )

    config = eval_runtime_config_from_args(parsed)
    run(resolve_eval_runtime_config(config))


if __name__ == "__main__":
    raise SystemExit(run_cli(lambda: main()))
