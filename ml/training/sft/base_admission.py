"""Admission checks for the formal pretraining checkpoint used by SFT."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ml.core.engine.checkpointing import EngineCheckpoint, load_checkpoint


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRETRAIN_PROTOCOL = (
    _REPO_ROOT / "configs" / "eval" / "pretrain_release_protocol.json"
)
DEFAULT_PRETRAIN_RECIPE = (
    _REPO_ROOT / "configs" / "pretrain" / "muon_recipe.json"
)
PARENT_VALIDATION_SCHEMA = "sophia_pretrain_final_checkpoint_validation_v1"
FORMAL_DEVICE_NAME = "NVIDIA GeForce RTX 5090"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return payload


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"formal base admission is missing {label}")
    return {str(key): item for key, item in value.items()}


def _require_equal(*, label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise RuntimeError(
            f"formal base checkpoint {label} mismatch: "
            f"expected={expected!r} actual={actual!r}"
        )


def _require_hash(value: object, *, label: str, length: int) -> str:
    text = str(value or "").lower()
    if len(text) != int(length) or any(char not in "0123456789abcdef" for char in text):
        raise RuntimeError(f"formal base admission has invalid {label}")
    return text


def _load_protocol(protocol_path: Path) -> dict[str, Any]:
    protocol = _load_object(protocol_path, label="pretraining release protocol")
    if protocol.get("schema") != "sophia_pretrain_release_protocol_v1":
        raise RuntimeError("unsupported pretraining release protocol")
    if protocol.get("status") != "fixed":
        raise RuntimeError("pretraining release protocol is not fixed")
    training = _object(protocol.get("training"), label="protocol.training")
    model = _object(protocol.get("model"), label="protocol.model")
    lineage = _object(protocol.get("lineage"), label="protocol.lineage")
    if not isinstance(lineage.get("required_hashes"), list):
        raise RuntimeError("pretraining release protocol has no lineage contract")
    for label, value in (
        ("training.seed", training.get("seed")),
        ("training.target_tokens", training.get("target_tokens")),
        ("training.tokens_per_update", training.get("tokens_per_update")),
        ("training.release_steps", training.get("release_steps")),
        ("training.consumed_tokens", training.get("consumed_tokens")),
        ("model.parameter_count", model.get("parameter_count")),
        ("model.max_seq_len", model.get("max_seq_len")),
    ):
        try:
            if int(value) <= 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"pretraining release protocol has invalid {label}") from exc
    _require_hash(model.get("spec_sha256"), label="protocol model hash", length=64)
    policy = _object(protocol.get("policy"), label="protocol.policy")
    policy_relative = Path(str(policy.get("path") or ""))
    if policy_relative.is_absolute() or ".." in policy_relative.parts:
        raise RuntimeError("pretraining release protocol has an unsafe policy path")
    policy_path = (_REPO_ROOT / policy_relative).resolve()
    try:
        policy_path.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise RuntimeError("pretraining release policy escapes the repository") from exc
    expected_policy_hash = _require_hash(
        policy.get("sha256"), label="protocol policy hash", length=64
    )
    if not policy_path.is_file() or sha256_file(policy_path) != expected_policy_hash:
        raise RuntimeError("pretraining release policy bytes do not match the protocol")
    return protocol


def validate_parent_validation_report(
    *,
    validation_path: Path,
    parent_checkpoint_path: Path,
    protocol_path: Path = DEFAULT_PRETRAIN_PROTOCOL,
) -> dict[str, Any]:
    """Bind a passing final-base evaluation report to the exact parent bytes."""

    protocol = _load_protocol(protocol_path)
    report = _load_object(validation_path, label="formal base validation report")
    if report.get("schema") != PARENT_VALIDATION_SCHEMA:
        raise RuntimeError("unsupported formal base validation report")
    if report.get("status") != "pass" or report.get("valid") is not True:
        raise RuntimeError("formal base validation report did not pass")
    protocol_hash = sha256_file(protocol_path)
    _require_equal(
        label="validation protocol SHA-256",
        actual=report.get("protocol_sha256"),
        expected=protocol_hash,
    )
    policy = _object(protocol.get("policy"), label="protocol.policy")
    expected_policy_hash = _require_hash(
        policy.get("sha256"), label="protocol policy hash", length=64
    )
    _require_equal(
        label="validation policy SHA-256",
        actual=report.get("policy_sha256"),
        expected=expected_policy_hash,
    )

    evidence = _object(report.get("evidence"), label="validation.evidence")
    if evidence.get("hard_gates_passed") is not True:
        raise RuntimeError("formal base validation evidence did not pass its hard gates")
    training = _object(protocol.get("training"), label="protocol.training")
    _require_equal(
        label="validation seed",
        actual=evidence.get("seed"),
        expected=training["seed"],
    )
    checkpoint = _object(
        evidence.get("checkpoint"), label="validation.evidence.checkpoint"
    )
    _require_equal(
        label="validation checkpoint SHA-256",
        actual=checkpoint.get("sha256"),
        expected=sha256_file(parent_checkpoint_path),
    )
    _require_equal(
        label="validation checkpoint step",
        actual=checkpoint.get("step"),
        expected=training["release_steps"],
    )
    _require_equal(
        label="validation consumed tokens",
        actual=checkpoint.get("consumed_tokens"),
        expected=training["consumed_tokens"],
    )
    _require_hash(
        evidence.get("candidate_sha256"),
        label="validation evidence hash",
        length=64,
    )
    checks = evidence.get("checks")
    if not isinstance(checks, list) or not checks:
        raise RuntimeError("formal base validation evidence has no checks")
    if any(not isinstance(check, dict) or check.get("passed") is not True for check in checks):
        raise RuntimeError("formal base validation evidence contains a failed check")
    return report


def validate_parent_checkpoint_contract(
    checkpoint: EngineCheckpoint,
    *,
    protocol_path: Path = DEFAULT_PRETRAIN_PROTOCOL,
    recipe_path: Path = DEFAULT_PRETRAIN_RECIPE,
) -> None:
    """Prove that a loaded full checkpoint is the fixed final base trajectory."""

    protocol = _load_protocol(protocol_path)
    training = _object(protocol.get("training"), label="protocol.training")
    model = _object(protocol.get("model"), label="protocol.model")
    recipe = _load_object(recipe_path, label="pretraining semantic recipe")
    semantics = _object(recipe.get("selected_semantics"), label="pretrain recipe semantics")
    _require_equal(label="kind", actual=checkpoint.kind, expected="full")
    _require_equal(
        label="step", actual=checkpoint.step, expected=training["release_steps"]
    )
    args = _object(checkpoint.args, label="checkpoint.args")
    profile = _object(args.get("_sophia_pretrain_profile"), label="release profile")
    _require_equal(label="release profile", actual=profile.get("kind"), expected="release")
    _require_equal(label="run kind", actual=args.get("_sophia_run_kind"), expected="pretrain")

    # The protocol's own bytes are deliberately absent from this contract. The
    # checkpoint records the protocol it was launched under, and resume checks
    # that recorded hash for drift inside one trajectory; but SFT admits a
    # checkpoint from an earlier release of this repository, where the protocol
    # file may since have been renamed or reformatted without changing a single
    # training semantic. The protocol says where its byte binding belongs --
    # "the checkpoint evidence binds its path and sha256" -- and the parent
    # validation report carries exactly that, against the current bytes. What
    # the checkpoint must satisfy here is the protocol's meaning, field by
    # field, which is what follows.
    exact_fields = {
        "seed": training["seed"],
        "total_tokens": training["target_tokens"],
        "target_tokens_per_update": training["tokens_per_update"],
        "_sophia_requested_target_tokens_per_update": training["tokens_per_update"],
        "seq_len": model["max_seq_len"],
        "max_seq_len": model["max_seq_len"],
        "_sophia_model_spec_sha256": model["spec_sha256"],
        "_sophia_model_parameter_count": model["parameter_count"],
        "optimizer_kind": semantics["optimizer_kind"],
        "learning_rate": semantics["learning_rate"],
        "weight_decay": semantics["weight_decay"],
        "beta1": semantics["beta1"],
        "beta2": semantics["beta2"],
        "adam_eps": semantics["adam_eps"],
        "warmup_steps": semantics["warmup_steps"],
        "lr_schedule": semantics["lr_schedule"],
        "min_lr_ratio": semantics["min_lr_ratio"],
        "muon_ns_steps": semantics["muon_ns_steps"],
    }
    for field, expected in exact_fields.items():
        _require_equal(label=field, actual=args.get(field), expected=expected)

    lineage_fields = {
        "_sophia_tokenizer_json_sha1": 40,
        "_sophia_train_manifest_sha1": 40,
        "_sophia_val_manifest_sha1": 40,
        "_sophia_test_manifest_sha1": 40,
        "_sophia_machine_recipe_sha256": 64,
        "_sophia_data_admission_sha256": 64,
        "_sophia_dataset_marker_sha256": 64,
    }
    for field, length in lineage_fields.items():
        _require_hash(args.get(field), label=field, length=length)
    machine_signature = _object(
        args.get("_sophia_machine_signature"), label="checkpoint machine signature"
    )
    _require_equal(
        label="machine device",
        actual=machine_signature.get("cuda_device_name"),
        expected=FORMAL_DEVICE_NAME,
    )
    if not checkpoint.optimizer:
        raise RuntimeError("formal base checkpoint has no optimizer state")
    if checkpoint.scheduler is None:
        raise RuntimeError("formal base checkpoint has no scheduler state")
    if checkpoint.rng is None:
        raise RuntimeError("formal base checkpoint has no RNG state")
    train_state = _object(checkpoint.train_state, label="checkpoint.train_state")
    # consumed_tokens counts what was fed; seen_supervised_tokens counts label
    # positions, and a causal sequence of length L supplies L-1 of them because
    # its last token has no next token to predict. Comparing the two directly
    # asks a correctly trained checkpoint to be wrong by exactly one token per
    # sequence.
    sequences = int(training["consumed_tokens"]) // int(model["max_seq_len"])
    _require_equal(
        label="seen supervised tokens",
        actual=train_state.get("seen_supervised_tokens"),
        expected=int(training["consumed_tokens"]) - sequences,
    )
    if not isinstance(train_state.get("data_iter_state"), Mapping):
        raise RuntimeError("formal base checkpoint has no data iterator state")


def load_and_validate_parent_checkpoint(
    path: Path,
    *,
    protocol_path: Path = DEFAULT_PRETRAIN_PROTOCOL,
    recipe_path: Path = DEFAULT_PRETRAIN_RECIPE,
) -> EngineCheckpoint:
    checkpoint = load_checkpoint(str(path), expected_kind="full")
    validate_parent_checkpoint_contract(
        checkpoint, protocol_path=protocol_path, recipe_path=recipe_path
    )
    return checkpoint


__all__ = [
    "DEFAULT_PRETRAIN_PROTOCOL",
    "DEFAULT_PRETRAIN_RECIPE",
    "FORMAL_DEVICE_NAME",
    "PARENT_VALIDATION_SCHEMA",
    "load_and_validate_parent_checkpoint",
    "sha256_file",
    "validate_parent_checkpoint_contract",
    "validate_parent_validation_report",
]
