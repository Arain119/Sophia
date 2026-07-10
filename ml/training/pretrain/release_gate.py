from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import replace

from ml.data.token_shards.shard_manifest import write_json_atomic
from ml.data.token_shards.tokenizer_fingerprint import (
    compute_tokenizer_bundle_sha1,
)
from ml.core.engine.machine_signature import machine_signature_payload
from ml.errors import SophiaUsageError
from ml.training.pretrain.machine_runtime import PretrainMachineRuntime
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.training.pretrain.artifacts import (
    PRETRAIN_MACHINE_RECIPE_ARTIFACT,
    PRETRAIN_MACHINE_RUNTIME_ARTIFACT,
    PRETRAIN_UPDATE_PROFILE_ARTIFACT,
)
from ml.training.pretrain.release_config import (
    CANONICAL_PRETRAIN_MACHINE_RECIPE_NAME,
    CANONICAL_PRETRAIN_MACHINE_RECIPE_PAYLOAD,
    CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD,
    PretrainReleaseSemantics,
)
from ml.training.pretrain import manifest_policy
from ml.training.pretrain.runtime_bootstrap import PretrainRuntimeBootstrap
from ml.training.release_gate_fs_checks import (
    MIN_DISK_FREE_GB,
    cleanup_stale_tmp_files as _shared_cleanup_stale_tmp_files,
    ensure_disk_free as _shared_ensure_disk_free,
)

PRETRAIN_MACHINE_RECIPE_KIND = "pretrain_machine_recipe"
MACHINE_SIGNATURE_MEMORY_TOLERANCE_GB = 0.01
DATA_SIGNATURE_PATH_FIELDS = frozenset(
    {
        "data_path",
        "eval_data_path",
        "test_data_path",
        "train_manifest_path",
        "val_manifest_path",
        "test_manifest_path",
        "tokenizer_path",
    }
)


def _json_payload(name: str, value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SophiaUsageError(f"[ERR] pretrain machine recipe is missing {name}.")
    return {str(key): item for key, item in value.items()}


def _compare_payload(name: str, expected: object, current: object) -> None:
    if expected != current:
        raise SophiaUsageError(
            "[ERR] pretrain release-gate mismatch.\n"
            f"field={name}\n"
            f"recipe={json.dumps(expected, ensure_ascii=False, sort_keys=True)}\n"
            f"current={json.dumps(current, ensure_ascii=False, sort_keys=True)}"
        )


def _same_existing_path(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    left_s = str(left)
    right_s = str(right)
    if left_s == right_s:
        return True
    if not left_s.strip() or not right_s.strip():
        return False
    try:
        return bool(os.path.samefile(left_s, right_s))
    except OSError:
        return False


def _compare_data_signature(
    *,
    expected: Mapping[str, object],
    current: Mapping[str, object],
) -> None:
    expected_payload = dict(expected)
    current_payload = dict(current)
    for field_name in DATA_SIGNATURE_PATH_FIELDS:
        if field_name not in expected_payload and field_name not in current_payload:
            continue
        if _same_existing_path(
            expected_payload.get(field_name),
            current_payload.get(field_name),
        ):
            current_payload[field_name] = expected_payload.get(field_name)
    _compare_payload("data_signature", expected_payload, current_payload)


def _compare_machine_signature(
    *,
    expected: Mapping[str, object],
    current: Mapping[str, object],
) -> None:
    expected_payload = dict(expected)
    current_payload = dict(current)
    expected_memory = expected_payload.pop("cuda_total_memory_gb", None)
    current_memory = current_payload.pop("cuda_total_memory_gb", None)
    if expected_payload != current_payload:
        _compare_payload("machine_signature", dict(expected), dict(current))
    if expected_memory is None and current_memory is None:
        return
    try:
        expected_gb = float(expected_memory)
        current_gb = float(current_memory)
    except (TypeError, ValueError) as exc:
        raise SophiaUsageError(
            "[ERR] pretrain release-gate mismatch.\n"
            "field=machine_signature.cuda_total_memory_gb\n"
            f"recipe={expected_memory!r}\n"
            f"current={current_memory!r}"
        ) from exc
    if abs(expected_gb - current_gb) > float(MACHINE_SIGNATURE_MEMORY_TOLERANCE_GB):
        _compare_payload("machine_signature", dict(expected), dict(current))


def _validate_supported_pretrain_machine_signature(
    *,
    name: str,
    signature: Mapping[str, object],
) -> None:
    device_type = str(signature.get("device_type", "") or "")
    if device_type != "cuda":
        raise SophiaUsageError(
            "[ERR] release pretrain requires a CUDA GPU.\n"
            f"field={name}.device_type\n"
            f"current={device_type!r}"
        )


def _canonical_machine_payload(
    payload: Mapping[str, object],
    *,
    expected: Mapping[str, object],
) -> dict[str, object]:
    return {str(key): payload.get(str(key)) for key in expected}


def _validate_canonical_pretrain_recipe(
    *,
    recipe_payload: Mapping[str, object],
    recipe_machine_signature: Mapping[str, object],
    current_machine_signature: Mapping[str, object],
    recipe_machine: Mapping[str, object],
    recipe_runtime: Mapping[str, object],
) -> None:
    selected_name = str(recipe_payload.get("selected_name", "") or "")
    if selected_name != str(CANONICAL_PRETRAIN_MACHINE_RECIPE_NAME):
        raise SophiaUsageError(
            "[ERR] release pretrain requires the canonical machine recipe.\n"
            "field=selected_name\n"
            f"expected={CANONICAL_PRETRAIN_MACHINE_RECIPE_NAME!r}\n"
            f"current={selected_name!r}"
        )
    _compare_payload(
        "machine_recipe",
        dict(CANONICAL_PRETRAIN_MACHINE_RECIPE_PAYLOAD),
        _canonical_machine_payload(
            recipe_machine,
            expected=CANONICAL_PRETRAIN_MACHINE_RECIPE_PAYLOAD,
        ),
    )
    _compare_payload(
        "machine_runtime",
        dict(CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD),
        _canonical_machine_payload(
            recipe_runtime,
            expected=CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD,
        ),
    )
    _validate_supported_pretrain_machine_signature(
        name="recipe_machine_signature",
        signature=recipe_machine_signature,
    )
    _validate_supported_pretrain_machine_signature(
        name="current_machine_signature",
        signature=current_machine_signature,
    )


def cleanup_stale_tmp_files(*, output_dir: str) -> int:
    return _shared_cleanup_stale_tmp_files(output_dir=output_dir)


def ensure_disk_free(*, path: str, min_free_gb: float = MIN_DISK_FREE_GB) -> float:
    return _shared_ensure_disk_free(
        path=path,
        min_free_gb=min_free_gb,
        context="release pretrain",
    )


def build_pretrain_data_signature(
    *,
    args: PretrainRunConfig,
    bootstrap: PretrainRuntimeBootstrap,
) -> dict[str, object]:
    eval_manifest_path = manifest_policy.resolve_manifest_path(str(bootstrap.eval_data_path))
    test_manifest_path = manifest_policy.resolve_manifest_path(str(bootstrap.test_data_path))
    return {
        "data_path": os.path.abspath(str(bootstrap.data_path)),
        "eval_data_path": (
            None
            if not str(bootstrap.eval_data_path).strip()
            else os.path.abspath(str(bootstrap.eval_data_path))
        ),
        "test_data_path": (
            None
            if not str(bootstrap.test_data_path).strip()
            else os.path.abspath(str(bootstrap.test_data_path))
        ),
        "train_manifest_path": os.path.abspath(str(bootstrap.manifest_path)),
        "train_manifest_sha1": str(bootstrap.train_manifest_sha1).lower(),
        "val_manifest_path": (
            None if not eval_manifest_path else os.path.abspath(str(eval_manifest_path))
        ),
        "val_manifest_sha1": (
            None if not str(bootstrap.val_manifest_sha1).strip() else str(bootstrap.val_manifest_sha1).lower()
        ),
        "test_manifest_path": (
            None if not test_manifest_path else os.path.abspath(str(test_manifest_path))
        ),
        "test_manifest_sha1": (
            None if not str(bootstrap.test_manifest_sha1).strip() else str(bootstrap.test_manifest_sha1).lower()
        ),
        "tokenizer_path": os.path.abspath(str(bootstrap.tokenizer_path)),
        "tokenizer_bundle_sha1": compute_tokenizer_bundle_sha1(
            str(bootstrap.tokenizer_path)
        ).lower(),
        "run_kind": str(args._sophia_run_kind or "pretrain"),
    }


def apply_pretrain_machine_recipe(
    *,
    args: PretrainRunConfig,
    recipe_payload: Mapping[str, object],
) -> PretrainRunConfig:
    recipe_machine = _json_payload("machine_recipe", recipe_payload.get("machine_recipe"))
    recipe_runtime = _json_payload("machine_runtime", recipe_payload.get("machine_runtime"))
    requested_state = PretrainRuntimeState.from_config(args)
    state = PretrainRuntimeState.from_config(args)
    for field_name in (
        "batch_size",
        "accumulation_steps",
        "gradient_checkpointing",
        "gradient_checkpointing_exclude_first",
        "gradient_checkpointing_exclude_last",
        "loss_chunk_size",
    ):
        raw = recipe_machine.get(field_name)
        try:
            setattr(state, field_name, int(raw))
        except (TypeError, ValueError) as exc:
            raise SophiaUsageError(
                "[ERR] pretrain machine recipe field must be an int.\n"
                f"field={field_name}\n"
                f"value={raw!r}"
            ) from exc
    runtime = PretrainMachineRuntime.from_source(recipe_runtime)
    # The machine recipe may provide the release default backend, but a probe or
    # caller-level experiment that explicitly sets a backend must remain
    # authoritative. Other machine-runtime fields still come from the recipe.
    if str(requested_state.step_execution_backend).strip():
        runtime = replace(runtime, step_execution_backend="")
    state.apply_machine_runtime(runtime)
    projected = state.project_run_config(args)

    requested_payload = requested_state.machine_runtime().to_payload()
    effective_payload = state.machine_runtime().to_payload()
    for field_name in (
        "seq_len",
        "batch_size",
        "accumulation_steps",
        "gradient_checkpointing",
        "gradient_checkpointing_exclude_first",
        "gradient_checkpointing_exclude_last",
        "loss_chunk_size",
    ):
        requested_payload[str(field_name)] = getattr(requested_state, str(field_name))
        effective_payload[str(field_name)] = getattr(state, str(field_name))
    diff = {
        key: {
            "requested": requested_payload.get(key),
            "effective": effective_payload.get(key),
        }
        for key in sorted(set(requested_payload) | set(effective_payload))
        if requested_payload.get(key) != effective_payload.get(key)
    }
    return replace(
        projected,
        _sophia_requested_runtime_config=dict(requested_payload),
        _sophia_effective_runtime_config=dict(effective_payload),
        _sophia_recipe_override_diff=dict(diff),
    )


def _recipe_semantic_args(
    *,
    args: PretrainRunConfig,
    recipe_machine: Mapping[str, object],
) -> PretrainRunConfig:
    """Project recipe-owned machine semantics before release-gate comparison."""

    del recipe_machine
    return args


def materialize_recipe_artifacts(
    *,
    output_dir: str,
    recipe_payload: Mapping[str, object],
) -> None:
    artifacts = _json_payload("artifacts", recipe_payload.get("artifacts"))
    expected = {
        "machine_recipe": PRETRAIN_MACHINE_RECIPE_ARTIFACT,
        "machine_runtime": PRETRAIN_MACHINE_RUNTIME_ARTIFACT,
        "update_profile": PRETRAIN_UPDATE_PROFILE_ARTIFACT,
    }
    root = os.path.abspath(str(output_dir or "").strip())
    os.makedirs(root, exist_ok=True)
    for key, filename in expected.items():
        payload = _json_payload(f"artifacts.{key}", artifacts.get(key))
        write_json_atomic(os.path.join(root, filename), dict(payload))


def validate_resume_checkpoint_machine_signature(
    *,
    resume_args: Mapping[str, object] | None,
    current_machine_signature: Mapping[str, object],
) -> None:
    if not isinstance(resume_args, Mapping):
        return
    raw = resume_args.get("_sophia_machine_signature")
    if raw is None:
        return
    checkpoint_signature = machine_signature_payload(raw if isinstance(raw, Mapping) else None)
    if checkpoint_signature:
        try:
            _compare_machine_signature(
                expected=dict(checkpoint_signature),
                current=dict(current_machine_signature),
            )
        except SophiaUsageError as exc:
            raise SophiaUsageError(
                "[ERR] resume checkpoint machine signature mismatch for release pretrain.\n"
                f"checkpoint={json.dumps(dict(checkpoint_signature), ensure_ascii=False, sort_keys=True)}\n"
                f"current={json.dumps(dict(current_machine_signature), ensure_ascii=False, sort_keys=True)}"
            ) from exc

def validate_signed_pretrain_machine_recipe(
    *,
    args: PretrainRunConfig,
    bootstrap: PretrainRuntimeBootstrap,
    recipe_payload: Mapping[str, object],
    current_machine_signature: Mapping[str, object],
) -> None:
    if str(recipe_payload.get("kind", "") or "") != str(PRETRAIN_MACHINE_RECIPE_KIND):
        raise SophiaUsageError(
            "[ERR] unsupported pretrain machine recipe kind.\n"
            f"kind={recipe_payload.get('kind')!r}"
        )
    stage = str(recipe_payload.get("stage", "") or "").strip().lower()
    if stage and stage != "pretrain":
        raise SophiaUsageError(
            "[ERR] pretrain machine recipe stage mismatch.\n"
            f"stage={stage!r}"
        )
    current_data_signature = build_pretrain_data_signature(
        args=args,
        bootstrap=bootstrap,
    )
    recipe_release_semantics = _json_payload(
        "release_semantics",
        recipe_payload.get("release_semantics"),
    )
    recipe_machine_signature = _json_payload(
        "machine_signature",
        recipe_payload.get("machine_signature"),
    )
    recipe_data_signature = _json_payload(
        "data_signature",
        recipe_payload.get("data_signature"),
    )
    recipe_machine = _json_payload("machine_recipe", recipe_payload.get("machine_recipe"))
    recipe_runtime = _json_payload("machine_runtime", recipe_payload.get("machine_runtime"))
    _json_payload("artifacts", recipe_payload.get("artifacts"))
    selected_name = str(recipe_payload.get("selected_name", "") or "").strip()
    if not selected_name:
        raise SophiaUsageError("[ERR] pretrain machine recipe selected_name is required.")
    if selected_name == str(CANONICAL_PRETRAIN_MACHINE_RECIPE_NAME):
        _validate_canonical_pretrain_recipe(
            recipe_payload=recipe_payload,
            recipe_machine_signature=recipe_machine_signature,
            current_machine_signature=current_machine_signature,
            recipe_machine=recipe_machine,
            recipe_runtime=recipe_runtime,
        )
    current_release_semantics = PretrainReleaseSemantics.from_run_config(
        args=_recipe_semantic_args(args=args, recipe_machine=recipe_machine)
    ).to_payload()
    _compare_payload("release_semantics", recipe_release_semantics, current_release_semantics)
    _compare_machine_signature(
        expected=dict(recipe_machine_signature),
        current=dict(current_machine_signature),
    )
    _compare_data_signature(
        expected=recipe_data_signature,
        current=current_data_signature,
    )


def requires_release_machine_recipe(args: PretrainRunConfig) -> bool:
    return str(args._sophia_run_kind or "pretrain") == "pretrain" and bool(
        machine_signature_payload(args._sophia_machine_signature)
    )


def run_pretrain_release_preflight(
    *,
    args: PretrainRunConfig,
    bootstrap: PretrainRuntimeBootstrap,
    recipe_payload: Mapping[str, object] | None,
) -> PretrainRunConfig:
    output_dir = os.path.abspath(str(bootstrap.output_dir or "").strip())
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        deleted = cleanup_stale_tmp_files(output_dir=output_dir)
        if deleted > 0:
            print(
                f"[INFO] pretrain preflight cleaned tmp artifacts: count={int(deleted)}",
                flush=True,
            )
        ensure_disk_free(path=output_dir)

    current_machine_signature = machine_signature_payload(args._sophia_machine_signature)
    if not requires_release_machine_recipe(args):
        return args
    if recipe_payload is None:
        raise SophiaUsageError(
            "[ERR] release pretrain requires --machine_recipe_json."
        )

    validate_signed_pretrain_machine_recipe(
        args=args,
        bootstrap=bootstrap,
        recipe_payload=recipe_payload,
        current_machine_signature=current_machine_signature,
    )
    validate_resume_checkpoint_machine_signature(
        resume_args=(
            None
            if bootstrap.resume_checkpoint is None
            or not isinstance(bootstrap.resume_checkpoint.args, Mapping)
            else bootstrap.resume_checkpoint.args
        ),
        current_machine_signature=current_machine_signature,
    )
    materialize_recipe_artifacts(
        output_dir=output_dir,
        recipe_payload=recipe_payload,
    )
    return apply_pretrain_machine_recipe(
        args=args,
        recipe_payload=recipe_payload,
    )


__all__ = [
    "MIN_DISK_FREE_GB",
    "PRETRAIN_MACHINE_RECIPE_KIND",
    "apply_pretrain_machine_recipe",
    "build_pretrain_data_signature",
    "cleanup_stale_tmp_files",
    "ensure_disk_free",
    "materialize_recipe_artifacts",
    "requires_release_machine_recipe",
    "run_pretrain_release_preflight",
    "validate_signed_pretrain_machine_recipe",
]
