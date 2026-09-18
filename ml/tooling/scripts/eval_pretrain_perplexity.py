from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import torch

from ml.core.common.io import write_json_atomic
from ml.data.token_shards.shard_manifest import (
    load_manifest,
    sha1_file,
    validate_tokenizer_fingerprint,
)
from ml.training.pretrain.engine.data.pretrain import build_pretrain_data_iter
from ml.training.pretrain.engine.eval import evaluate_loss


def _sha256(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing evaluation binding file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_binding(
    *,
    protocol_json: str | Path | None,
    checkpoint: str | Path | None,
    checkpoint_sha256: str | None,
    eval_export_model: str | Path | None,
    dataset_fingerprints: dict[str, str],
) -> dict[str, Any] | None:
    values = (protocol_json, checkpoint, checkpoint_sha256, eval_export_model)
    if not any(value is not None and str(value).strip() for value in values):
        return None
    if not all(value is not None and str(value).strip() for value in (protocol_json, checkpoint, eval_export_model)):
        raise ValueError(
            "release-bound evaluation requires --protocol_json, --checkpoint, and --eval_export_model"
        )
    protocol_path = Path(str(protocol_json)).expanduser().resolve()
    checkpoint_path = Path(str(checkpoint)).expanduser().resolve()
    model_path = Path(str(eval_export_model)).expanduser().resolve()
    protocol_digest = _sha256(protocol_path)
    checkpoint_digest = _sha256(checkpoint_path)
    model_digest = _sha256(model_path)
    if checkpoint_sha256 and str(checkpoint_sha256).lower() != checkpoint_digest:
        raise ValueError("checkpoint_sha256 does not match checkpoint")
    aggregate = hashlib.sha256()
    for name, digest in sorted(dataset_fingerprints.items()):
        aggregate.update(f"{name}={digest}\n".encode())
    return {
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "eval_export_model_path": str(model_path),
        "eval_export_model_sha256": model_digest,
        "dataset_fingerprints": dict(sorted(dataset_fingerprints.items())),
        "evaluation_asset_sha256": aggregate.hexdigest(),
    }


def parse_dataset_specs(values: list[str]) -> dict[str, str]:
    datasets: dict[str, str] = {}
    for value in values:
        name, separator, path = str(value).partition("=")
        name = name.strip()
        path = path.strip()
        if not separator or not name or not path:
            raise ValueError(f"dataset must use name=/path/to/manifest.json: {value!r}")
        if name in datasets:
            raise ValueError(f"duplicate dataset name: {name}")
        datasets[name] = path
    if not datasets:
        raise ValueError("at least one --dataset is required")
    return datasets


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    from ml.runtime.inference.eval.model_io import init_model
    from ml.runtime.inference.eval.runtime_config import EvalRuntimeConfig

    datasets = parse_dataset_specs(list(args.dataset))
    device = torch.device(str(args.device))
    base_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, _tokenizer = init_model(
        EvalRuntimeConfig(
            export_dir=str(args.export_dir),
            device=str(device),
            seed=int(args.seed),
            use_cache=False,
        )
    )
    control = getattr(model, "apply_runtime_recipe_knobs", None)
    if callable(control):
        control(
            loss_chunk_size=int(args.loss_chunk_size),
            gradient_checkpointing_exclude_first=0,
            gradient_checkpointing_exclude_last=0,
        )
    results: dict[str, dict[str, Any]] = {}
    for name, manifest_value in datasets.items():
        manifest_path = Path(manifest_value).expanduser().resolve()
        manifest = load_manifest(str(manifest_path))
        validate_tokenizer_fingerprint(
            tokenizer_dir=str(args.tokenizer_path),
            expected_sha1=str(manifest.tokenizer_sha1),
        )
        data_iter = build_pretrain_data_iter(
            manifest_path=str(manifest_path),
            seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            device=device,
            num_workers=0,
            dataloader_prefetch_factor=0,
            dataloader_persistent_workers=0,
            shard_preload=0,
            shard_preload_bytes=0,
        )
        try:
            loss = evaluate_loss(
                model=model,
                data_iter=data_iter,
                steps=int(args.steps),
                base_dtype=base_dtype,
            )
        finally:
            data_iter.close()
        results[name] = {
            "manifest_path": str(manifest_path),
            "manifest_sha1": sha1_file(str(manifest_path)),
            "manifest_tokens": int(manifest.total_tokens),
            "loss": float(loss),
            "perplexity": float(math.exp(min(float(loss), 50.0))),
        }
        print(
            f"[EVAL] dataset={name} loss={loss:.6f} "
            f"ppl={results[name]['perplexity']:.4f}",
            flush=True,
        )
    dataset_fingerprints = {
        name: str(results[name]["manifest_sha1"]) for name in sorted(results)
    }
    report = {
        "schema": "sophia_pretrain_perplexity_eval_v1",
        "evaluation_binding": _evaluation_binding(
            protocol_json=args.protocol_json,
            checkpoint=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            eval_export_model=args.eval_export_model,
            dataset_fingerprints=dataset_fingerprints,
        ),
        "model_dir": os.path.abspath(str(args.export_dir)),
        "tokenizer_path": os.path.abspath(str(args.tokenizer_path)),
        "settings": {
            "seq_len": int(args.seq_len),
            "batch_size": int(args.batch_size),
            "steps": int(args.steps),
            "seed": int(args.seed),
            "loss_chunk_size": int(args.loss_chunk_size),
        },
        "datasets": results,
    }
    write_json_atomic(
        args.output,
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Sophia token-shard losses")
    parser.add_argument("--export_dir", required=True)
    parser.add_argument("--tokenizer_path", default="ml/modeling/text")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--loss_chunk_size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--protocol_json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint_sha256")
    parser.add_argument("--eval_export_model")
    return parser


def main() -> None:
    run_evaluation(build_parser().parse_args())


if __name__ == "__main__":
    main()
