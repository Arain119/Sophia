#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ml.tooling.pipelines.posttrain_length_curriculum import (
    write_posttrain_length_curriculum,
)
from ml.training.posttrain.defaults import DEFAULT_POSTTRAIN_CURRICULUM_PATH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the production post-training length curriculum plan."
    )
    parser.add_argument("--sft_dir", type=str, default=os.path.join("dataset", "sft"))
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_POSTTRAIN_CURRICULUM_PATH,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = Path(args.output).resolve()
    plan = write_posttrain_length_curriculum(
        sft_dir=Path(args.sft_dir).resolve(),
        output=output,
    )
    print(json.dumps({"ok": True, "output": str(output), "kind": plan["kind"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
