from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from ml.data.token_shards.shard_manifest import sha1_file
from ml.core.engine.checkpointing import EngineCheckpoint, FULL_CHECKPOINT_KIND
from ml.errors import SophiaUsageError
from ml.training.pretrain.resources import (
    PretrainManifestLike,
    PretrainTokenizerLike,
)
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.data_admission import DATASET_MARKER
from ml.training.pretrain.profiles import is_release_pretrain_profile, resolve_pretrain_profile
from ml.core.spec import ModelSpec
from ml.training.pretrain.pipeline_snapshot import PretrainPipelineSnapshot
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
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RELEASE_PROTOCOL_PATH = _REPO_ROOT / "configs/eval/pretrain_release_protocol.json"
_RELEASE_MODEL_CONFIG_PATH = _REPO_ROOT / "configs/model/sophia.json"

_BOOTSTRAP_CONFIG_SCALARS: tuple[tuple[str, str], ...] = (
    ("resolved_resume_checkpoint", "_sophia_resolved_resume_checkpoint"),
    ("data_path", "data_path"),
    ("eval_data_path", "eval_data_path"),
    ("test_data_path", "test_data_path"),
    ("total_tokens", "total_tokens"),
    ("train_manifest_sha1", "_sophia_train_manifest_sha1"),
    ("val_manifest_sha1", "_sophia_val_manifest_sha1"),
    ("test_manifest_sha1", "_sophia_test_manifest_sha1"),
    ("protocol_path", "_sophia_protocol_path"),
    ("protocol_sha256", "_sophia_protocol_sha256"),
    ("model_spec_sha256", "_sophia_model_spec_sha256"),
    ("model_parameter_count", "_sophia_model_parameter_count"),
    ("tokenizer_json_sha1", "_sophia_tokenizer_json_sha1"),
    ("dataset_marker_sha256", "_sophia_dataset_marker_sha256"),
    ("data_admission_sha256", "_sophia_data_admission_sha256"),
    ("machine_recipe_path", "_sophia_machine_recipe_path"),
    ("machine_recipe_sha256", "_sophia_machine_recipe_sha256"),
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
    ("protocol_path", "_sophia_protocol_path"),
    ("protocol_sha256", "_sophia_protocol_sha256"),
    ("model_spec_sha256", "_sophia_model_spec_sha256"),
    ("model_parameter_count", "_sophia_model_parameter_count"),
    ("tokenizer_json_sha1", "_sophia_tokenizer_json_sha1"),
    ("dataset_marker_sha256", "_sophia_dataset_marker_sha256"),
    ("data_admission_sha256", "_sophia_data_admission_sha256"),
    ("machine_recipe_path", "_sophia_machine_recipe_path"),
    ("machine_recipe_sha256", "_sophia_machine_recipe_sha256"),
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_model_sha1(tokenizer_dir: str) -> str:
    tokenizer_json = os.path.join(os.path.abspath(str(tokenizer_dir)), "tokenizer.json")
    if not os.path.isfile(tokenizer_json):
        raise SophiaUsageError(
            f"[ERR] tokenizer directory is missing tokenizer.json: {tokenizer_json}"
        )
    return sha1_file(tokenizer_json).lower()


def _require_hex_fingerprint(*, label: str, value: object, length: int) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != int(length) or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise SophiaUsageError(
            f"[ERR] {label} must be a {int(length)}-character lowercase hexadecimal fingerprint."
        )
    return normalized


def _dataset_marker_path(data_path: str) -> Path:
    current = Path(str(data_path)).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        marker = candidate / DATASET_MARKER
        if marker.is_file():
            return marker
    raise SophiaUsageError(
        "[ERR] admitted pretraining dataset marker is missing above the loaded data split: "
        f"{current / DATASET_MARKER}"
    )


def _data_admission_sha256(*, data_path: str) -> str:
    marker_path = _dataset_marker_path(data_path)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SophiaUsageError(
            f"[ERR] unable to load admitted pretraining dataset marker: {marker_path}"
        ) from exc
    if not isinstance(marker, dict):
        raise SophiaUsageError(
            "[ERR] admitted pretraining dataset marker must be a JSON object."
        )
    lineage = marker.get("lineage")
    if not isinstance(lineage, dict):
        raise SophiaUsageError(
            "[ERR] admitted pretraining dataset marker is missing lineage."
        )
    return _require_hex_fingerprint(
        label="data admission policy SHA-256",
        value=lineage.get("admission_policy_sha256"),
        length=64,
    )


def _load_release_protocol() -> tuple[str, str, dict[str, object]]:
    path = _RELEASE_PROTOCOL_PATH.resolve()
    if not path.is_file():
        raise SophiaUsageError(f"[ERR] formal release protocol is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SophiaUsageError(f"[ERR] unable to load formal release protocol: {path}") from exc
    if not isinstance(payload, dict):
        raise SophiaUsageError("[ERR] formal release protocol must be a JSON object.")
    model = payload.get("model")
    if not isinstance(model, dict):
        raise SophiaUsageError("[ERR] formal release protocol is missing model.")
    protocol_sha256 = _require_hex_fingerprint(
        label="formal release protocol SHA-256", value=sha256_file(str(path)), length=64
    )
    expected_model_hash = _require_hex_fingerprint(
        label="protocol model spec SHA-256", value=model.get("spec_sha256"), length=64
    )
    canonical_hash = hashlib.sha256(
        json.dumps(
            ModelSpec.default().to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    try:
        model_config = json.loads(_RELEASE_MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
        configured_model = model_config["model"]
        configured_parameter_count = model_config["parameter_count"]["unique_parameters"]
        configured_hash = hashlib.sha256(
            json.dumps(configured_model, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SophiaUsageError("[ERR] authoritative release model config is invalid.") from exc
    if canonical_hash != expected_model_hash or configured_hash != expected_model_hash:
        raise SophiaUsageError("[ERR] authoritative model semantics do not match protocol.")
    if int(model.get("parameter_count", 0) or 0) != int(configured_parameter_count):
        raise SophiaUsageError("[ERR] authoritative release model config does not match protocol.")
    return str(path), protocol_sha256, payload


def _machine_recipe_lineage(*, args: PretrainRunConfig) -> tuple[str, str]:
    recipe_value = normalized_str(args.machine_recipe_json)
    if not recipe_value:
        raise SophiaUsageError(
            "[ERR] release pretrain requires --machine_recipe_json."
        )
    recipe_path = Path(recipe_value).expanduser().resolve()
    if not recipe_path.is_file():
        raise SophiaUsageError(
            f"[ERR] release pretrain machine recipe is missing: {recipe_path}"
        )
    try:
        recipe_sha256 = sha256_file(str(recipe_path))
    except OSError as exc:
        raise SophiaUsageError(
            f"[ERR] unable to fingerprint release pretrain machine recipe: {recipe_path}"
        ) from exc
    return str(recipe_path), _require_hex_fingerprint(
        label="machine recipe SHA-256", value=recipe_sha256, length=64
    )


def _release_lineage(
    *, args: PretrainRunConfig, tokenizer_path: str
) -> dict[str, object]:
    protocol_path, protocol_sha256, protocol = _load_release_protocol()
    model = protocol["model"]
    marker_path = _dataset_marker_path(args.data_path)
    tokenizer_sha1 = tokenizer_model_sha1(tokenizer_path)
    machine_recipe_path, machine_recipe_sha256 = _machine_recipe_lineage(args=args)
    return {
        "protocol_path": protocol_path,
        "protocol_sha256": protocol_sha256,
        "model_spec_sha256": str(model["spec_sha256"]),
        "model_parameter_count": int(model["parameter_count"]),
        "tokenizer_json_sha1": tokenizer_sha1,
        "dataset_marker_sha256": _require_hex_fingerprint(
            label="dataset marker SHA-256", value=sha256_file(str(marker_path)), length=64
        ),
        "data_admission_sha256": _data_admission_sha256(data_path=str(args.data_path)),
        "machine_recipe_path": machine_recipe_path,
        "machine_recipe_sha256": machine_recipe_sha256,
    }


def _validate_release_resume_lineage(
    *, args: PretrainRunConfig, resume_args: dict[str, object] | None
) -> None:
    if not is_release_pretrain_profile(resolve_pretrain_profile(args)) or resume_args is None:
        return
    current = {
        "protocol_path": str(_RELEASE_PROTOCOL_PATH.resolve()),
        "protocol_sha256": str(args._sophia_protocol_sha256),
        "model_spec_sha256": str(args._sophia_model_spec_sha256),
        "model_parameter_count": int(args._sophia_model_parameter_count),
        "tokenizer_json_sha1": str(args._sophia_tokenizer_json_sha1),
        "dataset_marker_sha256": str(args._sophia_dataset_marker_sha256),
        "data_admission_sha256": str(args._sophia_data_admission_sha256),
        "machine_recipe_path": str(args._sophia_machine_recipe_path),
        "machine_recipe_sha256": str(args._sophia_machine_recipe_sha256),
    }
    migration_from = normalized_str(
        args._sophia_machine_recipe_migration_from_sha256
    )
    if migration_from:
        migration_from = _require_hex_fingerprint(
            label="machine recipe migration source SHA-256",
            value=migration_from,
            length=64,
        )
    fields = (
        ("protocol_path", str),
        ("protocol_sha256", lambda value: _require_hex_fingerprint(label="checkpoint protocol SHA-256", value=value, length=64)),
        ("model_spec_sha256", lambda value: _require_hex_fingerprint(label="checkpoint model spec SHA-256", value=value, length=64)),
        ("model_parameter_count", int),
        ("tokenizer_json_sha1", lambda value: _require_hex_fingerprint(label="checkpoint tokenizer JSON SHA-1", value=value, length=40)),
        ("dataset_marker_sha256", lambda value: _require_hex_fingerprint(label="checkpoint dataset marker SHA-256", value=value, length=64)),
        ("data_admission_sha256", lambda value: _require_hex_fingerprint(label="checkpoint data admission SHA-256", value=value, length=64)),
        ("machine_recipe_path", str),
        ("machine_recipe_sha256", lambda value: _require_hex_fingerprint(label="checkpoint machine recipe SHA-256", value=value, length=64)),
    )
    for field_name, converter in fields:
        checkpoint_field_name = f"_sophia_{field_name}"
        if checkpoint_field_name not in resume_args:
            raise SophiaUsageError(
                f"[ERR] release resume checkpoint is missing {checkpoint_field_name}."
            )
        try:
            checkpoint_value = converter(resume_args[checkpoint_field_name])
        except (TypeError, ValueError) as exc:
            raise SophiaUsageError(f"[ERR] release resume checkpoint has invalid {field_name}.") from exc
        if field_name == "machine_recipe_sha256" and migration_from:
            if checkpoint_value != migration_from:
                raise SophiaUsageError(
                    "[ERR] machine recipe migration source does not match "
                    "the release resume checkpoint. "
                    f"checkpoint={checkpoint_value!r} migration={migration_from!r}"
                )
            continue
        if field_name == "machine_recipe_path" and migration_from:
            continue
        if (
            field_name == "protocol_path"
            and int(args._sophia_allow_protocol_path_relocation) == 1
        ):
            continue
        if checkpoint_value != current[field_name]:
            raise SophiaUsageError(
                f"[ERR] release resume checkpoint {field_name} drifted. "
                f"checkpoint={checkpoint_value!r} current={current[field_name]!r}"
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
    return replace(
        args,
        _sophia_train_manifest_sha1=str(train_manifest_sha1),
        _sophia_val_manifest_sha1=manifest_sha1_if_configured(eval_manifest_path),
        _sophia_test_manifest_sha1=manifest_sha1_if_configured(test_manifest_path),
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
        if field_name in {"resume_path", "manifest", "tokenizer"}:
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
        updates[config_name] = (
            int(value)
            if field_name in {"total_tokens", "model_parameter_count"}
            else str(value)
        )
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
    protocol_path: str = ""
    protocol_sha256: str = ""
    model_spec_sha256: str = ""
    model_parameter_count: int = 0
    tokenizer_json_sha1: str = ""
    dataset_marker_sha256: str = ""
    data_admission_sha256: str = ""
    machine_recipe_path: str = ""
    machine_recipe_sha256: str = ""

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
    if (
        resume_checkpoint is not None
        and str(resume_checkpoint.kind) != FULL_CHECKPOINT_KIND
    ):
        raise SophiaUsageError(
            "[ERR] pretrain resume requires a full checkpoint; "
            f"found kind={resume_checkpoint.kind!r}"
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
    if (
        is_release_pretrain_profile(resolve_pretrain_profile(shadow_args))
        and str(shadow_args._sophia_run_kind or "pretrain") == "pretrain"
    ):
        shadow_args = replace(
            shadow_args,
            **{
                f"_sophia_{name}": value
                for name, value in _release_lineage(
                    args=shadow_args, tokenizer_path=str(loaded_tokenizer.path)
                ).items()
            },
        )
        _validate_release_resume_lineage(args=shadow_args, resume_args=resume_args)
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
        protocol_path=coerce_str(shadow_args._sophia_protocol_path),
        protocol_sha256=coerce_str(shadow_args._sophia_protocol_sha256),
        model_spec_sha256=coerce_str(shadow_args._sophia_model_spec_sha256),
        model_parameter_count=coerce_int(shadow_args._sophia_model_parameter_count),
        tokenizer_json_sha1=coerce_str(shadow_args._sophia_tokenizer_json_sha1),
        dataset_marker_sha256=coerce_str(shadow_args._sophia_dataset_marker_sha256),
        data_admission_sha256=coerce_str(shadow_args._sophia_data_admission_sha256),
        machine_recipe_path=coerce_str(shadow_args._sophia_machine_recipe_path),
        machine_recipe_sha256=coerce_str(shadow_args._sophia_machine_recipe_sha256),
    )


__all__ = [
    "PretrainRuntimeBootstrap",
    "infer_total_tokens_from_manifest",
    "manifest_sha1_if_configured",
    "record_current_manifest_fingerprints",
    "validate_resume_manifest_fingerprints",
    "prepare_pretrain_runtime_bootstrap",
]
