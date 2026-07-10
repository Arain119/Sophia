#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ml.tooling.pipelines.posttrain_length_profile import (
    build_posttrain_length_profile,
    write_posttrain_length_profile,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile true rendered token lengths for SFT datasets."
    )
    parser.add_argument("--dataset_root", type=str, default="dataset")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=os.path.join("ml", "modeling", "text"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("out", "posttrain_length_profile.json"),
    )
    parser.add_argument("--max_model_length", type=int, default=4096)
    parser.add_argument(
        "--mode",
        type=str,
        default="tokens",
        choices=["tokens", "chars"],
        help="`tokens` uses the real tokenizer; `chars` is a fast rendered-character profile.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="sft",
        help="Comma-separated subset of: sft.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated subset of: train,val,test.",
    )
    parser.add_argument(
        "--sample_rows_per_split",
        type=int,
        default=0,
        help="Deterministically profile only the first N rows per split when >0.",
    )
    parser.add_argument(
        "--top_examples",
        type=int,
        default=32,
        help="Keep N longest examples per split in the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output).resolve()
    requested_datasets = tuple(
        item.strip()
        for item in str(args.datasets or "").split(",")
        if item.strip()
    )
    requested_splits = tuple(
        item.strip()
        for item in str(args.splits or "").split(",")
        if item.strip()
    )
    report = build_posttrain_length_profile(
        dataset_root=Path(args.dataset_root).resolve(),
        tokenizer_path=Path(args.tokenizer_path).resolve(),
        max_model_length=int(args.max_model_length),
        mode=str(args.mode),
        requested_datasets=requested_datasets,
        requested_splits=requested_splits,
        sample_rows_per_split=int(args.sample_rows_per_split),
        top_examples=int(args.top_examples),
    )
    for dataset_name in requested_datasets:
        dataset_report = report["datasets"].get(dataset_name, {})
        for split in requested_splits:
            split_report = dataset_report.get(split)
            if not isinstance(split_report, dict):
                continue
            print(
                "[PROFILE] "
                f"{dataset_name}/{split} rows={int(split_report['rows']):,} "
                f"avg={split_report['avg']} p95={int(split_report['p95']):,} "
                f"p99={int(split_report['p99']):,} max={int(split_report['max']):,} "
                f">=4k={int(split_report['thresholds']['>=4096']['rows']):,}",
                flush=True,
            )
    write_posttrain_length_profile(output_path=output_path, report=report)
    print(f"[OK] wrote: {output_path}", flush=True)


if __name__ == "__main__":
    main()
