from __future__ import annotations

import os
from dataclasses import dataclass, replace

from ml.data.token_shards.shard_manifest import sha1_file
from ml.core.engine.checkpointing import EngineCheckpoint
from ml.errors import SophiaUsageError
from ml.training.pretrain.resources import (
    PretrainManifestLike,
    PretrainTokenizerLike,
)
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.tasks.pretrain.session import PretrainPipelineSnapshot
from ml.training.pretrain.value_semantics import coerce_int, coerce_str, normalized_str
from ml.training.pretrain import manifest_policy

_BOOTSTRAP_SNAPSHOT_SCALARS: tuple[tuple[str, str], ...] = (
    ("output_dir", "output_dir"),
    ("resume_path", "resume_path"),
    ("manifest_path", "manifest_path"),
    ("manifest", "manifest"),
    ("tokenizer", "tokenizer"),
    ("tokenizer_path", "tokenizer_path"),
    ("seq_len", "seq_len"),
)
_BOOTSTRAP_CONFIG_SCALARS: tuple[tuple[str, str], ...] = (
    ("resolved_resume_checkpoint", "_sophia_resolved_resume_checkpoint"),
    ("data_path", "data_path"),
    ("eval_data_path", "eval_data_path"),
    ("test_data_path", "test_data_path"),
    ("total_tokens", "total_tokens"),
    ("train_manifest_sha1", "_sophia_train_manifest_sha1"),
    ("val_manifest_sha1", "_sophia_val_manifest_sha1"),
    ("test_manifest_sha1", "_sophia_test_manifest_sha1"),
)
_BOOTSTRAP_CONFIG_PROJECT_FIELDS: tuple[tuple[str, str], ...] = (
    ("output_dir", "output_dir"),
    ("data_path", "data_path"),
    ("eval_data_path", "eval_data_path"),
    ("test_data_path", "test_data_path"),
    ("total_tokens", "total_tokens"),
    ("tokenizer_path", "tokenizer_path"),
    ("resolved_resume_checkpoint", "_sophia_resolved_resume_checkpoint"),
    ("train_manifest_sha1", "_sophia_train_manifest_sha1"),
    ("val_manifest_sha1", "_sophia_val_manifest_sha1"),
    ("test_manifest_sha1", "_sophia_test_manifest_sha1"),
)


def infer_total_tokens_from_manifest(manifest: PretrainManifestLike) -> int:
    try:
        total = coerce_int(manifest.total_tokens)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SophiaUsageError("[ERR] manifest total_tokens must be an integer") from exc
    if total < 0:
        raise SophiaUsageError("[ERR] manifest total_tokens must be >= 0")
    if total > 0:
        return int(total)
    try:
        shards = list(manifest.shards)
    except (AttributeError, TypeError) as exc:
        raise SophiaUsageError("[ERR] manifest shards must be a sequence") from exc
    shard_tokens: list[int] = []
    for index, shard in enumerate(shards):
        try:
            tokens = coerce_int(shard.tokens)
        except (AttributeError, TypeError, ValueError) as exc:
            raise SophiaUsageError(
                f"[ERR] manifest shard {index} tokens must be an integer"
            ) from exc
        if tokens < 0:
            raise SophiaUsageError(
                f"[ERR] manifest shard {index} tokens must be >= 0"
            )
        shard_tokens.append(tokens)
    total = sum(shard_tokens)
    return int(total)


def manifest_sha1_if_configured(path: str) -> str:
    resolved = normalized_str(path)
    if not resolved:
        return ""
    if not os.path.isfile(resolved):
        raise SophiaUsageError(f"[ERR] manifest file not found: {resolved}")
    try:
        return sha1_file(resolved).lower()
    except Exception as exc:
        raise SophiaUsageError(
            f"[ERR] failed to fingerprint manifest: {resolved}: {exc}"
        ) from exc


def record_current_manifest_fingerprints(
    *,
    args: PretrainRunConfig,
    manifest_path: str,
) -> PretrainRunConfig:
    train_manifest_sha1 = sha1_file(str(manifest_path)).lower()
    eval_manifest_path = manifest_policy.resolve_manifest_path(
        str(args.eval_data_path)
    )
    test_manifest_path = manifest_policy.resolve_manifest_path(
        str(args.test_data_path)
    )
    decay_manifest_sha1 = ""
    decay_data_path = normalized_str(getattr(args, "decay_data_path", ""))
    if decay_data_path:
        decay_manifest_path = manifest_policy.resolve_manifest_path(decay_data_path)
        decay_manifest_sha1 = manifest_sha1_if_configured(decay_manifest_path)
    return replace(
        args,
        _sophia_train_manifest_sha1=str(train_manifest_sha1),
        _sophia_val_manifest_sha1=manifest_sha1_if_configured(eval_manifest_path),
        _sophia_test_manifest_sha1=manifest_sha1_if_configured(test_manifest_path),
        _sophia_decay_manifest_sha1=str(decay_manifest_sha1),
    )


def validate_resume_manifest_fingerprints(
    *,
    args: PretrainRunConfig,
    resume_args: dict[str, object] | None,
) -> None:
    if not isinstance(resume_args, dict):
        return
    prev_train_sha1 = normalized_str(
        resume_args.get("_sophia_train_manifest_sha1")
    ).lower()
    cur_train_sha1 = normalized_str(args._sophia_train_manifest_sha1).lower()
    if prev_train_sha1 and cur_train_sha1 and prev_train_sha1 != cur_train_sha1:
        raise SophiaUsageError(
            "[ERR] resume checkpoint train manifest fingerprint mismatch.\n"
            f"Checkpoint: {prev_train_sha1}\n"
            f"Current:    {cur_train_sha1}\n"
            "Delete the output_dir and restart after rebuilding the dataset."
        )

    for label, prev_sha1, cur_sha1 in (
        (
            "val",
            normalized_str(resume_args.get("_sophia_val_manifest_sha1")).lower(),
            normalized_str(args._sophia_val_manifest_sha1).lower(),
        ),
        (
            "test",
            normalized_str(resume_args.get("_sophia_test_manifest_sha1")).lower(),
            normalized_str(args._sophia_test_manifest_sha1).lower(),
        ),
        (
            "decay",
            normalized_str(resume_args.get("_sophia_decay_manifest_sha1")).lower(),
            normalized_str(args._sophia_decay_manifest_sha1).lower(),
        ),
    ):
        if prev_sha1 and cur_sha1 and prev_sha1 != cur_sha1:
            raise SophiaUsageError(
                f"[ERR] resume checkpoint {label} manifest fingerprint mismatch.\n"
                f"Checkpoint: {prev_sha1}\n"
                f"Current:    {cur_sha1}\n"
                "Delete the output_dir and restart after rebuilding the evaluation dataset."
            )


def _bootstrap_payload_from_snapshot(
    snapshot: PretrainPipelineSnapshot,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for field_name, source_name in _BOOTSTRAP_SNAPSHOT_SCALARS:
        value = getattr(snapshot, source_name)
        if field_name in {"resume_path", "manifest"}:
            payload[field_name] = value
            continue
        payload[field_name] = (
            coerce_int(value)
            if field_name == "seq_len"
            else coerce_str(value)
        )
    return payload


def _bootstrap_payload_from_config(cfg: PretrainRunConfig) -> dict[str, object]:
    payload: dict[str, object] = {}
    for field_name, source_name in _BOOTSTRAP_CONFIG_SCALARS:
        value = getattr(cfg, source_name)
        payload[field_name] = (
            coerce_int(value) if field_name == "total_tokens" else coerce_str(value)
        )
    return payload


def _project_bootstrap_run_config(
    *,
    bootstrap: PretrainRuntimeBootstrap,
    cfg: PretrainRunConfig,
) -> PretrainRunConfig:
    updates: dict[str, object] = {}
    for field_name, config_name in _BOOTSTRAP_CONFIG_PROJECT_FIELDS:
        value = getattr(bootstrap, field_name)
        updates[config_name] = int(value) if field_name == "total_tokens" else str(value)
    return replace(cfg, **updates)


@dataclass(frozen=True)
class PretrainRuntimeBootstrap:
    output_dir: str
    resume_path: str | None
    resume_checkpoint: EngineCheckpoint | None
    resolved_resume_checkpoint: str
    data_path: str
    eval_data_path: str
    test_data_path: str
    manifest_path: str
    manifest: PretrainManifestLike
    total_tokens: int
    tokenizer: PretrainTokenizerLike
    tokenizer_path: str
    seq_len: int
    train_manifest_sha1: str
    val_manifest_sha1: str
    test_manifest_sha1: str

    @classmethod
    def from_snapshot(
        cls,
        *,
        snapshot: PretrainPipelineSnapshot,
        cfg: PretrainRunConfig,
        resume_checkpoint: EngineCheckpoint | None = None,
    ) -> PretrainRuntimeBootstrap:
        payload = _bootstrap_payload_from_snapshot(snapshot)
        payload.update(_bootstrap_payload_from_config(cfg))
        payload["resume_checkpoint"] = resume_checkpoint
        return cls(**payload)

    def project_run_config(self, cfg: PretrainRunConfig) -> PretrainRunConfig:
        return _project_bootstrap_run_config(bootstrap=self, cfg=cfg)


def prepare_pretrain_runtime_bootstrap(
    *,
    args: PretrainRunConfig,
    prepare_output_dir_and_resume,
    load_checkpoint,
    prepare_pretrain_eval_data,
    load_manifest_or_die,
    load_tokenizer_and_validate_manifest,
) -> PretrainRuntimeBootstrap:
    shadow_args = replace(args)
    output_resolution = prepare_output_dir_and_resume(shadow_args)
    shadow_args = output_resolution.args
    output_dir = str(output_resolution.output_dir)
    resume_path = output_resolution.resume_path
    shadow_args = replace(
        shadow_args,
        _sophia_resolved_resume_checkpoint=(
            "" if resume_path is None else str(resume_path)
        ),
    )
    resume_checkpoint = (
        load_checkpoint(str(resume_path)) if resume_path is not None else None
    )
    resume_args = (
        resume_checkpoint.args
        if resume_checkpoint is not None and isinstance(resume_checkpoint.args, dict)
        else None
    )

    shadow_args = prepare_pretrain_eval_data(
        args=shadow_args,
        output_dir=str(output_dir),
        resume_path=resume_path,
    )

    loaded_manifest = load_manifest_or_die(args=shadow_args)
    if str(loaded_manifest.tokenizer_path or "").strip():
        shadow_args = replace(
            shadow_args,
            tokenizer_path=str(loaded_manifest.tokenizer_path),
        )
    manifest_path = str(loaded_manifest.path)
    manifest = loaded_manifest.manifest

    shadow_args = record_current_manifest_fingerprints(
        args=shadow_args,
        manifest_path=str(manifest_path),
    )
    validate_resume_manifest_fingerprints(
        args=shadow_args,
        resume_args=resume_args,
    )

    if int(shadow_args.total_tokens or 0) <= 0:
        total_tokens = infer_total_tokens_from_manifest(manifest)
        if total_tokens <= 0:
            raise SophiaUsageError(
                "[ERR] unable to infer --total_tokens from manifest (tokens<=0)"
            )
        shadow_args = replace(shadow_args, total_tokens=int(total_tokens))
        print(
            f"[INFO] total_tokens=auto -> {int(total_tokens):,} (from manifest)",
            flush=True,
        )

    loaded_tokenizer = load_tokenizer_and_validate_manifest(
        args=shadow_args,
        manifest=manifest,
    )
    return PretrainRuntimeBootstrap(
        output_dir=coerce_str(output_dir),
        resume_path=resume_path,
        resume_checkpoint=resume_checkpoint,
        resolved_resume_checkpoint=coerce_str(
            shadow_args._sophia_resolved_resume_checkpoint
        ),
        data_path=coerce_str(shadow_args.data_path),
        eval_data_path=coerce_str(shadow_args.eval_data_path),
        test_data_path=coerce_str(shadow_args.test_data_path),
        manifest_path=coerce_str(manifest_path),
        manifest=manifest,
        total_tokens=coerce_int(shadow_args.total_tokens),
        tokenizer=loaded_tokenizer.tokenizer,
        tokenizer_path=coerce_str(loaded_tokenizer.path),
        seq_len=int(loaded_tokenizer.seq_len),
        train_manifest_sha1=coerce_str(shadow_args._sophia_train_manifest_sha1),
        val_manifest_sha1=coerce_str(shadow_args._sophia_val_manifest_sha1),
        test_manifest_sha1=coerce_str(shadow_args._sophia_test_manifest_sha1),
    )


__all__ = [
    "PretrainRuntimeBootstrap",
    "infer_total_tokens_from_manifest",
    "manifest_sha1_if_configured",
    "record_current_manifest_fingerprints",
    "validate_resume_manifest_fingerprints",
    "prepare_pretrain_runtime_bootstrap",
]
