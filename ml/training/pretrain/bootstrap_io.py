from __future__ import annotations

import os
from dataclasses import replace

from ml.errors import SophiaUsageError
from ml.data.token_shards.shard_manifest import (
    load_manifest,
    validate_tokenizer_fingerprint,
)
from ml.data.token_shards.tokenizer_fingerprint import (
    default_modeling_tokenizer_dir,
    resolve_matching_tokenizer_dir,
)
from ml.training.pretrain.resources import (
    LoadedPretrainManifest,
    LoadedPretrainTokenizer,
    PretrainManifestLike,
)
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.value_semantics import coerce_str, normalized_str
from ml.training.pretrain import manifest_policy
from ml.training.pretrain.model_setup import load_tokenizer


def prepare_pretrain_eval_data(
    *,
    args: PretrainRunConfig,
    output_dir: str,
    resume_path: str | None,
) -> PretrainRunConfig:
    del output_dir, resume_path
    cfg = args
    raw = normalized_str(cfg.data_path)
    if not raw:
        raise SophiaUsageError("[ERR] data_path is empty.")

    raw_abs = os.path.abspath(raw)
    if raw_abs.endswith(os.sep + "manifest.json") and os.path.isfile(raw_abs):
        split_dir = os.path.dirname(raw_abs) or "."
    else:
        split_dir = raw_abs

    split_dir_name = os.path.basename(split_dir).lower().strip()
    if split_dir_name in ("train", "val", "test"):
        root = os.path.dirname(split_dir) or "."
    else:
        root = split_dir

    train_manifest = os.path.join(root, "train", "manifest.json")
    val_manifest = os.path.join(root, "val", "manifest.json")
    test_manifest = os.path.join(root, "test", "manifest.json")
    if (
        not os.path.exists(train_manifest)
        or not os.path.exists(val_manifest)
        or not os.path.exists(test_manifest)
    ):
        raise SophiaUsageError(
            "[ERR] Missing required train/val/test split.\n"
            "Expected:\n"
            f"  {train_manifest}\n"
            f"  {val_manifest}\n"
            f"  {test_manifest}\n"
            "Pass --data_path pointing to the dataset root (containing train/, val/, test/), or the train split directory."
        )

    cfg = replace(
        cfg,
        data_path=str(os.path.dirname(os.path.abspath(train_manifest))),
        eval_data_path=str(os.path.dirname(os.path.abspath(val_manifest))),
        test_data_path=str(os.path.dirname(os.path.abspath(test_manifest))),
    )
    print(
        f"[INFO] dataset split | train={cfg.data_path} | val={cfg.eval_data_path} | test={cfg.test_data_path}",
        flush=True,
    )
    return cfg


def load_manifest_or_die(*, args: PretrainRunConfig) -> LoadedPretrainManifest:
    manifest_path = manifest_policy.resolve_manifest_path(args.data_path)
    if not manifest_path or not os.path.exists(manifest_path):
        raise SophiaUsageError(f"manifest.json not found: {manifest_path}")
    manifest = load_manifest(manifest_path)
    resolved_tokenizer_dir = resolve_matching_tokenizer_dir(
        manifest_path=str(manifest_path),
        expected_sha1=str(manifest.tokenizer_sha1),
        explicit_tokenizer_dir=coerce_str(args.tokenizer_path),
        default_tokenizer_dir=default_modeling_tokenizer_dir(),
    )
    try:
        validate_tokenizer_fingerprint(
            tokenizer_dir=str(resolved_tokenizer_dir),
            expected_sha1=str(manifest.tokenizer_sha1),
        )
    except (OSError, ValueError) as exc:
        raise SophiaUsageError(str(exc)) from exc
    return LoadedPretrainManifest(
        path=str(manifest_path),
        manifest=manifest,
        tokenizer_path=str(resolved_tokenizer_dir),
    )


def load_tokenizer_and_validate_manifest(
    *,
    args: PretrainRunConfig,
    manifest: PretrainManifestLike,
) -> LoadedPretrainTokenizer:
    seq_len = int(args.seq_len)
    manifest_path = manifest_policy.resolve_manifest_path(args.data_path)
    if not manifest_path or not os.path.exists(manifest_path):
        raise SophiaUsageError(f"manifest.json not found: {manifest_path}")
    tok_path = resolve_matching_tokenizer_dir(
        manifest_path=str(manifest_path),
        expected_sha1=str(getattr(manifest, "tokenizer_sha1", "") or ""),
        explicit_tokenizer_dir=coerce_str(args.tokenizer_path),
        default_tokenizer_dir=default_modeling_tokenizer_dir(),
    )
    try:
        validate_tokenizer_fingerprint(
            tokenizer_dir=str(tok_path),
            expected_sha1=str(getattr(manifest, "tokenizer_sha1", "") or ""),
        )
    except (OSError, ValueError) as exc:
        raise SophiaUsageError(str(exc)) from exc
    tokenizer = load_tokenizer(tok_path, model_max_length=int(seq_len))
    tok_eos = tokenizer.eos_token_id
    if tok_eos is None:
        raise SophiaUsageError(
            "Tokenizer has no eos_token_id; token-shard pretraining requires EOS-separated documents."
        )
    if getattr(manifest, "eos_token_id", None) is not None and int(
        manifest.eos_token_id
    ) != int(tok_eos):
        raise SophiaUsageError(
            "Token shard manifest eos_token_id mismatch.\n"
            f"manifest.eos_token_id={manifest.eos_token_id}\n"
            f"tokenizer.eos_token_id={tok_eos}\n"
            "Rebuild shards with the same tokenizer, or point --tokenizer_path to the matching tokenizer."
        )
    return LoadedPretrainTokenizer(
        tokenizer=tokenizer,
        path=str(tok_path),
        seq_len=int(seq_len),
    )


__all__ = [
    "load_manifest_or_die",
    "load_tokenizer_and_validate_manifest",
    "prepare_pretrain_eval_data",
]
