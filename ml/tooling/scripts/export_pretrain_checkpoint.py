from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ml.core.common.io import write_json_atomic
from ml.core.engine.checkpointing import load_checkpoint
from ml.integrations.export.artifacts import export_model_artifacts
from ml.integrations.export.model_dir import (
    load_local_export_tokenizer,
    load_trainable_decoder_from_export_dir,
)


EXPORT_SCHEMA = "sophia_pretrain_checkpoint_export_v3"


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_recipe_sha256(checkpoint: object) -> str | None:
    args = getattr(checkpoint, "args", None)
    if not isinstance(args, Mapping):
        return None
    value = str(args.get("_sophia_machine_recipe_sha256") or "").lower()
    if not value:
        return None
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("checkpoint machine recipe SHA-256 is invalid")
    return value


def export_checkpoint(
    *,
    checkpoint_path: str,
    parent_export: str,
    output_dir: str,
) -> dict[str, Any]:
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    parent = Path(parent_export).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_file}")
    for filename in ("config.json", "model.safetensors"):
        if not (parent / filename).is_file():
            raise FileNotFoundError(f"parent export is incomplete: {parent / filename}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output_dir must be empty: {output}")

    checkpoint = load_checkpoint(str(checkpoint_file))
    model = load_trainable_decoder_from_export_dir(
        export_dir=str(parent),
        device=torch.device("cpu"),
        base_dtype=torch.bfloat16,
        gradient_checkpointing=False,
        gradient_checkpointing_exclude_first=0,
        gradient_checkpointing_exclude_last=0,
        apply_gradient_checkpointing_fn=lambda *_args, **_kwargs: None,
    )
    model.load_state_dict(checkpoint.model, strict=True)
    model.eval()

    output.mkdir(parents=True, exist_ok=True)
    export_model_artifacts(
        model=model,
        tokenizer=load_local_export_tokenizer(str(parent)),
        output_dir=str(output),
        safe_serialization=True,
    )
    model_path = output / "model.safetensors"
    if not model_path.is_file():
        raise RuntimeError(f"export did not produce model.safetensors: {output}")

    report: dict[str, Any] = {
        "schema": EXPORT_SCHEMA,
        "checkpoint": {
            "path": str(checkpoint_file),
            "sha256": sha256_file(checkpoint_file),
            "step": int(checkpoint.step),
            "kind": str(checkpoint.kind),
            "machine_recipe_sha256": _checkpoint_recipe_sha256(checkpoint),
        },
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
        "parent": {
            "path": str(parent),
            "config_sha256": sha256_file(parent / "config.json"),
            "model_sha256": sha256_file(parent / "model.safetensors"),
        },
        "output_dir": str(output),
    }
    write_json_atomic(
        output / "checkpoint_export.json",
        report,
        ensure_ascii=False,
        sort_keys=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a Sophia pretrain checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parent_export", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = export_checkpoint(
        checkpoint_path=str(args.checkpoint),
        parent_export=str(args.parent_export),
        output_dir=str(args.output_dir),
    )
    print(
        f"[DONE] step={report['checkpoint']['step']} export={report['output_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
