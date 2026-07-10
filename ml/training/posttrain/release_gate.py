from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import torch

from ml.data.token_shards.shard_manifest import sha1_file
from ml.data.token_shards.tokenizer_fingerprint import (
    compute_tokenizer_bundle_sha1,
)
from ml.errors import SophiaUsageError
from ml.core.engine.checkpointing import EngineCheckpoint
from ml.core.common.io import write_json_atomic
from ml.core.engine.machine_signature import build_machine_adaptive_signature
from ml.runtime.stack import ensure_standard_stack
from ml.runtime.torch_env import require_cuda
from ml.training.posttrain.release_config import build_runtime_sft_release_semantics
from ml.training.posttrain.types import PosttrainStageArgs
from ml.training.posttrain.runtime_config import (
    POSTTRAIN_MACHINE_RECIPE_KIND,
    POSTTRAIN_MACHINE_SIGNATURE_SCHEMA,
)
from ml.training.release_gate_fs_checks import (
    MIN_DISK_FREE_GB,
    cleanup_stale_tmp_files as _shared_cleanup_stale_tmp_files,
    ensure_disk_free as _shared_ensure_disk_free,
)

POSTTRAIN_OUTPUT_RECIPE_SUMMARY = "machine_recipe_summary.json"
POSTTRAIN_OUTPUT_RECIPE_COPY = "machine_recipe.json"
def _compare_float(name: str, expected: object, current: object) -> None:
    try:
        expected_value = float(expected)
        current_value = float(current)
    except (TypeError, ValueError) as exc:
        raise SophiaUsageError(
            "[ERR] invalid release semantic float field.\n"
            f"field={name}\n"
            f"expected={expected!r}\n"
            f"current={current!r}"
        ) from exc
    if not (math.isfinite(expected_value) and math.isfinite(current_value)):
        raise SophiaUsageError(
            "[ERR] non-finite release semantic float field.\n"
            f"field={name}\n"
            f"expected={expected!r}\n"
            f"current={current!r}"
        )
    if abs(expected_value - current_value) > 1e-12:
        raise SophiaUsageError(
            "[ERR] release semantic drift detected.\n"
            f"field={name}\n"
            f"recipe={expected_value!r}\n"
            f"current={current_value!r}"
        )


def _compare_scalar(name: str, expected: object, current: object) -> None:
    if expected != current:
        raise SophiaUsageError(
            "[ERR] release semantic drift detected.\n"
            f"field={name}\n"
            f"recipe={expected!r}\n"
            f"current={current!r}"
        )


def cleanup_stale_tmp_files(*, output_dir: str) -> int:
    return _shared_cleanup_stale_tmp_files(output_dir=output_dir)


def ensure_path_exists(*, path: str, label: str, expect_dir: bool = False) -> str:
    resolved = os.path.abspath(str(path or "").strip())
    if not resolved:
        raise SophiaUsageError(f"[ERR] missing required {label}.")
    if expect_dir:
        if not os.path.isdir(resolved):
            raise SophiaUsageError(f"[ERR] {label} directory does not exist: {resolved}")
    else:
        if not os.path.isfile(resolved):
            raise SophiaUsageError(f"[ERR] {label} file does not exist: {resolved}")
    if not os.access(resolved, os.R_OK):
        raise SophiaUsageError(f"[ERR] {label} is not readable: {resolved}")
    return resolved


def ensure_disk_free(*, path: str, min_free_gb: float = MIN_DISK_FREE_GB) -> float:
    return _shared_ensure_disk_free(path=path, min_free_gb=min_free_gb)


def build_export_signature(*, export_dir: str) -> dict[str, object]:
    resolved = ensure_path_exists(path=export_dir, label="export_dir", expect_dir=True)
    config_path = ensure_path_exists(
        path=os.path.join(resolved, "config.json"),
        label="export config.json",
    )
    return {
        "export_dir": resolved,
        "config_sha1": sha1_file(config_path).lower(),
        "tokenizer_bundle_sha1": compute_tokenizer_bundle_sha1(resolved).lower(),
    }


def build_data_signature(
    *,
    train_data: str,
    eval_data: str,
    test_data: str,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for label, raw_path in (
        ("train", train_data),
        ("eval", eval_data),
        ("test", test_data),
    ):
        normalized = str(raw_path or "").strip()
        if not normalized:
            payload[label] = None
            continue
        resolved = ensure_path_exists(path=normalized, label=f"{label}_data")
        payload[label] = {
            "path": resolved,
            "sha1": sha1_file(resolved).lower(),
        }
    return payload


def build_current_release_semantics(
    *,
    stage: str,
    args: PosttrainStageArgs,
    resolved_max_seq_len: int,
    machine_recipe: Mapping[str, object] | None = None,
) -> dict[str, object]:
    normalized_stage = str(stage or "").strip().lower()
    if normalized_stage == "sft":
        return build_runtime_sft_release_semantics(
            args=args,
            resolved_max_seq_len=int(resolved_max_seq_len),
            machine_recipe=machine_recipe,
        ).to_payload()
    raise ValueError(f"unsupported post-train stage: {stage!r}")


def validate_release_semantics(
    *,
    stage: str,
    recipe_release_semantics: Mapping[str, object],
    current_release_semantics: Mapping[str, object],
) -> None:
    normalized_stage = str(stage or "").strip().lower()
    for field in (
        "curriculum_stage",
        "eval_interval",
        "save_interval",
        "resolved_max_seq_len",
        "max_seq_len_source",
    ):
        _compare_scalar(
            field,
            recipe_release_semantics.get(field),
            current_release_semantics.get(field),
        )
    for field in ("learning_rate",):
        _compare_float(
            field,
            recipe_release_semantics.get(field),
            current_release_semantics.get(field),
        )
    if normalized_stage == "sft":
        for field in ("target_examples_per_update", "weight_decay"):
            value_expected = recipe_release_semantics.get(field)
            value_current = current_release_semantics.get(field)
            if field == "weight_decay":
                _compare_float(field, value_expected, value_current)
            else:
                _compare_scalar(field, value_expected, value_current)
        return
    raise ValueError(f"unsupported post-train stage: {stage!r}")


def validate_signed_machine_recipe(
    *,
    args: PosttrainStageArgs,
    stage: str,
    resolved_max_seq_len: int,
    recipe_payload: Mapping[str, object],
    current_machine_signature: Mapping[str, object],
) -> None:
    if str(recipe_payload.get("kind", "") or "") != str(POSTTRAIN_MACHINE_RECIPE_KIND):
        raise SophiaUsageError(
            "[ERR] unsupported post-train machine recipe kind.\n"
            f"kind={recipe_payload.get('kind')!r}"
        )
    recipe_stage = str(recipe_payload.get("stage", "") or "").strip().lower()
    expected_stage = str(stage or "").strip().lower()
    if recipe_stage and recipe_stage != expected_stage:
        raise SophiaUsageError(
            "[ERR] post-train machine recipe stage mismatch.\n"
            f"recipe={recipe_stage!r}\n"
            f"current={expected_stage!r}"
        )
    recipe_release_semantics = recipe_payload.get("release_semantics")
    if not isinstance(recipe_release_semantics, Mapping):
        raise SophiaUsageError("[ERR] post-train machine recipe is missing release_semantics.")
    recipe_machine = recipe_payload.get("machine_recipe")
    if not isinstance(recipe_machine, Mapping):
        raise SophiaUsageError("[ERR] post-train machine recipe is missing machine_recipe.")
    recipe_machine_signature = recipe_payload.get("machine_signature")
    if not isinstance(recipe_machine_signature, Mapping):
        raise SophiaUsageError("[ERR] post-train machine recipe is missing machine_signature.")
    recipe_export_signature = recipe_payload.get("export_signature")
    if not isinstance(recipe_export_signature, Mapping):
        raise SophiaUsageError("[ERR] post-train machine recipe is missing export_signature.")
    recipe_data_signature = recipe_payload.get("data_signature")
    if not isinstance(recipe_data_signature, Mapping):
        raise SophiaUsageError("[ERR] post-train machine recipe is missing data_signature.")

    current_release_semantics = build_current_release_semantics(
        stage=stage,
        args=args,
        resolved_max_seq_len=int(resolved_max_seq_len),
        machine_recipe=recipe_machine,
    )
    validate_release_semantics(
        stage=stage,
        recipe_release_semantics=recipe_release_semantics,
        current_release_semantics=current_release_semantics,
    )

    current_machine = dict(current_machine_signature)
    if dict(recipe_machine_signature) != current_machine:
        raise SophiaUsageError(
            "[ERR] post-train machine recipe signature mismatch.\n"
            f"recipe={json.dumps(dict(recipe_machine_signature), ensure_ascii=False, sort_keys=True)}\n"
            f"current={json.dumps(current_machine, ensure_ascii=False, sort_keys=True)}"
        )

    current_export_signature = build_export_signature(export_dir=str(args.export_dir))
    if dict(recipe_export_signature) != current_export_signature:
        raise SophiaUsageError(
            "[ERR] post-train export signature mismatch.\n"
            f"recipe={json.dumps(dict(recipe_export_signature), ensure_ascii=False, sort_keys=True)}\n"
            f"current={json.dumps(current_export_signature, ensure_ascii=False, sort_keys=True)}"
        )

    current_data_signature = build_data_signature(
        train_data=str(args.train_data),
        eval_data=str(args.eval_data),
        test_data=str(args.test_data),
    )
    if dict(recipe_data_signature) != current_data_signature:
        raise SophiaUsageError(
            "[ERR] post-train data signature mismatch.\n"
            f"recipe={json.dumps(dict(recipe_data_signature), ensure_ascii=False, sort_keys=True)}\n"
            f"current={json.dumps(current_data_signature, ensure_ascii=False, sort_keys=True)}"
        )


def materialize_posttrain_recipe_artifacts(
    *,
    output_dir: str,
    recipe_payload: Mapping[str, object],
) -> None:
    root = os.path.abspath(str(output_dir or "").strip())
    os.makedirs(root, exist_ok=True)
    source_summary_path = ensure_path_exists(
        path=str(recipe_payload.get("source_summary") or ""),
        label="machine_recipe source_summary",
    )
    try:
        source_summary_payload = json.loads(
            Path(str(source_summary_path)).read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise SophiaUsageError(
            "[ERR] unable to load post-train machine recipe summary.\n"
            f"path={source_summary_path}\n"
            f"error={type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(source_summary_payload, Mapping):
        raise SophiaUsageError(
            "[ERR] post-train machine recipe summary must decode to an object.\n"
            f"path={source_summary_path}"
        )
    write_json_atomic(
        os.path.join(root, POSTTRAIN_OUTPUT_RECIPE_SUMMARY),
        dict(source_summary_payload),
        sort_keys=True,
    )
    write_json_atomic(
        os.path.join(root, POSTTRAIN_OUTPUT_RECIPE_COPY),
        dict(recipe_payload),
        sort_keys=True,
    )


def validate_resume_checkpoint_contract(
    *,
    args: PosttrainStageArgs,
    stage: str,
    resolved_max_seq_len: int,
    resume_checkpoint: EngineCheckpoint | None,
) -> None:
    if resume_checkpoint is None or not isinstance(resume_checkpoint.args, Mapping):
        return

    resume_args = dict(resume_checkpoint.args)
    checkpoint_stage = str(resume_args.get("_sophia_run_kind", "") or "").strip().lower()
    expected_stage = str(stage or "").strip().lower()
    if checkpoint_stage and checkpoint_stage != expected_stage:
        raise SophiaUsageError(
            "[ERR] post-train resume checkpoint stage mismatch.\n"
            f"checkpoint={checkpoint_stage!r}\n"
            f"current={expected_stage!r}"
        )

    current_release_semantics = build_current_release_semantics(
        stage=expected_stage,
        args=args,
        resolved_max_seq_len=int(resolved_max_seq_len),
    )
    checkpoint_release_semantics = build_current_release_semantics(
        stage=expected_stage,
        args=PosttrainStageArgs.from_mapping(resume_args),
        resolved_max_seq_len=int(resume_args.get("_sophia_resolved_max_seq_len") or resolved_max_seq_len),
    )
    validate_release_semantics(
        stage=expected_stage,
        recipe_release_semantics=checkpoint_release_semantics,
        current_release_semantics=current_release_semantics,
    )

    checkpoint_export_signature = build_export_signature(
        export_dir=str(resume_args.get("export_dir") or "")
    )
    current_export_signature = build_export_signature(export_dir=str(args.export_dir))
    if dict(checkpoint_export_signature) != dict(current_export_signature):
        raise SophiaUsageError(
            "[ERR] post-train resume checkpoint export signature mismatch.\n"
            f"checkpoint={json.dumps(dict(checkpoint_export_signature), ensure_ascii=False, sort_keys=True)}\n"
            f"current={json.dumps(dict(current_export_signature), ensure_ascii=False, sort_keys=True)}"
        )

    checkpoint_data_signature = build_data_signature(
        train_data=str(resume_args.get("train_data") or ""),
        eval_data=str(resume_args.get("eval_data") or ""),
        test_data=str(resume_args.get("test_data") or ""),
    )
    current_data_signature = build_data_signature(
        train_data=str(args.train_data),
        eval_data=str(args.eval_data),
        test_data=str(args.test_data),
    )
    if dict(checkpoint_data_signature) != dict(current_data_signature):
        raise SophiaUsageError(
            "[ERR] post-train resume checkpoint data signature mismatch.\n"
            f"checkpoint={json.dumps(dict(checkpoint_data_signature), ensure_ascii=False, sort_keys=True)}\n"
            f"current={json.dumps(dict(current_data_signature), ensure_ascii=False, sort_keys=True)}"
        )


def run_posttrain_release_preflight(
    *,
    args: PosttrainStageArgs,
    stage: str,
    resolved_max_seq_len: int,
    recipe_payload: Mapping[str, object] | None,
) -> dict[str, object]:
    ensure_standard_stack(mode="train", require_tensorboard=True)
    ensure_path_exists(path=str(args.train_data), label="train_data")
    if str(args.eval_data or "").strip():
        ensure_path_exists(path=str(args.eval_data), label="eval_data")
    if str(args.test_data or "").strip():
        ensure_path_exists(path=str(args.test_data), label="test_data")
    ensure_path_exists(path=str(args.export_dir), label="export_dir", expect_dir=True)
    if str(args.curriculum_path or "").strip():
        ensure_path_exists(
            path=str(args.curriculum_path),
            label="curriculum_path",
        )

    output_dir = os.path.abspath(str(args.output_dir or "").strip())
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        deleted = cleanup_stale_tmp_files(output_dir=output_dir)
        if deleted > 0:
            print(
                f"[INFO] preflight cleaned tmp artifacts: count={int(deleted)}",
                flush=True,
            )
        ensure_disk_free(path=output_dir)

    ensure_disk_free(path=str(args.export_dir))
    device_str = require_cuda(str(args.device))
    current_machine_signature = build_machine_adaptive_signature(
        schema=POSTTRAIN_MACHINE_SIGNATURE_SCHEMA,
        device=torch.device(device_str)
    ).to_payload()

    if recipe_payload is None:
        if bool(int(args.allow_missing_machine_recipe) == 1):
            return current_machine_signature
        raise SophiaUsageError(
            "[ERR] release post-training requires --machine_recipe_json.\n"
            f"stage={str(stage or '').strip().lower()}"
        )

    validate_signed_machine_recipe(
        args=args,
        stage=str(stage or "").strip().lower(),
        resolved_max_seq_len=int(resolved_max_seq_len),
        recipe_payload=recipe_payload,
        current_machine_signature=current_machine_signature,
    )
    materialize_posttrain_recipe_artifacts(
        output_dir=output_dir,
        recipe_payload=recipe_payload,
    )
    return current_machine_signature


__all__ = [
    "MIN_DISK_FREE_GB",
    "POSTTRAIN_OUTPUT_RECIPE_COPY",
    "POSTTRAIN_OUTPUT_RECIPE_SUMMARY",
    "build_current_release_semantics",
    "build_data_signature",
    "build_export_signature",
    "cleanup_stale_tmp_files",
    "ensure_disk_free",
    "ensure_path_exists",
    "materialize_posttrain_recipe_artifacts",
    "run_posttrain_release_preflight",
    "validate_resume_checkpoint_contract",
    "validate_release_semantics",
    "validate_signed_machine_recipe",
]
