from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.core.engine.checkpointing import EngineCheckpoint
from ml.training.sft.base_admission import (
    DEFAULT_PRETRAIN_PROTOCOL,
    DEFAULT_PRETRAIN_RECIPE,
    PARENT_VALIDATION_SCHEMA,
    validate_parent_checkpoint_contract,
    validate_parent_validation_report,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol() -> dict[str, object]:
    return json.loads(DEFAULT_PRETRAIN_PROTOCOL.read_text(encoding="utf-8"))


def _checkpoint() -> EngineCheckpoint:
    protocol = _protocol()
    training = protocol["training"]
    model = protocol["model"]
    recipe = json.loads(DEFAULT_PRETRAIN_RECIPE.read_text(encoding="utf-8"))
    semantics = recipe["selected_semantics"]
    args = {
        "_sophia_pretrain_profile": {"kind": "release"},
        "_sophia_run_kind": "pretrain",
        "seed": training["seed"],
        "total_tokens": training["target_tokens"],
        "target_tokens_per_update": training["tokens_per_update"],
        "_sophia_requested_target_tokens_per_update": training["tokens_per_update"],
        "seq_len": model["max_seq_len"],
        "max_seq_len": model["max_seq_len"],
        "_sophia_model_spec_sha256": model["spec_sha256"],
        "_sophia_model_parameter_count": model["parameter_count"],
        "_sophia_protocol_sha256": _sha256(DEFAULT_PRETRAIN_PROTOCOL),
        "_sophia_tokenizer_json_sha1": "a" * 40,
        "_sophia_train_manifest_sha1": "b" * 40,
        "_sophia_val_manifest_sha1": "c" * 40,
        "_sophia_test_manifest_sha1": "d" * 40,
        "_sophia_machine_recipe_sha256": "e" * 64,
        "_sophia_data_admission_sha256": "f" * 64,
        "_sophia_dataset_marker_sha256": "1" * 64,
        "_sophia_machine_signature": {
            "cuda_device_name": "NVIDIA GeForce RTX 5090"
        },
    }
    for field in (
        "optimizer_kind",
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "adam_eps",
        "warmup_steps",
        "lr_schedule",
        "min_lr_ratio",
        "muon_ns_steps",
    ):
        args[field] = semantics[field]
    return EngineCheckpoint(
        step=training["release_steps"],
        model={"weight": object()},
        optimizer={"state": {"ready": True}},
        scheduler={"last_epoch": training["release_steps"]},
        args=args,
        rng={"torch": "state"},
        ema=None,
        train_state={
            "seen_supervised_tokens": int(training["consumed_tokens"])
            - int(training["consumed_tokens"]) // int(model["max_seq_len"]),
            "data_iter_state": {"cursor": training["consumed_tokens"]},
        },
    )


def _validation(path: Path, parent: Path) -> Path:
    protocol = _protocol()
    training = protocol["training"]
    payload = {
        "schema": PARENT_VALIDATION_SCHEMA,
        "status": "pass",
        "valid": True,
        "policy_sha256": protocol["policy"]["sha256"],
        "protocol_sha256": _sha256(DEFAULT_PRETRAIN_PROTOCOL),
        "evidence": {
            "candidate_sha256": "a" * 64,
            "seed": training["seed"],
            "hard_gates_passed": True,
            "checkpoint": {
                "path": str(parent),
                "sha256": _sha256(parent),
                "step": training["release_steps"],
                "consumed_tokens": training["consumed_tokens"],
            },
            "checks": [{"name": "formal", "passed": True}],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_formal_base_checkpoint_contract_accepts_only_final_release() -> None:
    validate_parent_checkpoint_contract(_checkpoint())


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("step", 153, "step mismatch"),
        ("run_kind", "pretrain_machine_validate", "run kind mismatch"),
        ("seen_tokens", 1, "seen supervised tokens mismatch"),
    ],
)
def test_formal_base_checkpoint_contract_rejects_nonformal_trajectory(
    field: str, value: object, match: str
) -> None:
    checkpoint = _checkpoint()
    if field == "step":
        checkpoint = EngineCheckpoint(**{**checkpoint.__dict__, "step": value})
    elif field == "run_kind":
        checkpoint.args["_sophia_run_kind"] = value
    else:
        checkpoint.train_state["seen_supervised_tokens"] = value
    with pytest.raises(RuntimeError, match=match):
        validate_parent_checkpoint_contract(checkpoint)


def test_parent_validation_report_binds_exact_checkpoint_bytes(tmp_path: Path) -> None:
    parent = tmp_path / "base.pt"
    parent.write_bytes(b"formal base")
    validation = _validation(tmp_path / "base_validation.json", parent)

    report = validate_parent_validation_report(
        validation_path=validation,
        parent_checkpoint_path=parent,
    )
    assert report["valid"] is True

    parent.write_bytes(b"changed base")
    with pytest.raises(RuntimeError, match="validation checkpoint SHA-256 mismatch"):
        validate_parent_validation_report(
            validation_path=validation,
            parent_checkpoint_path=parent,
        )


def _protocol_file(path: Path, **overrides: object) -> Path:
    """Write a copy of the formal protocol with some semantics overridden."""

    protocol = _protocol()
    for dotted, value in overrides.items():
        section, _, field = dotted.partition("__")
        protocol[section][field] = value
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path


def test_formal_base_checkpoint_contract_admits_superseded_protocol_bytes(
    tmp_path: Path,
) -> None:
    """A rename or reformat of the protocol must not disown a finished run.

    The checkpoint records the protocol bytes it was launched under. Those
    bytes are frozen forever, while the file keeps evolving, so admission
    checks what the protocol means rather than what it weighs.
    """

    checkpoint = _checkpoint()
    checkpoint.args["_sophia_protocol_sha256"] = "0" * 64
    validate_parent_checkpoint_contract(checkpoint)

    del checkpoint.args["_sophia_protocol_sha256"]
    validate_parent_checkpoint_contract(checkpoint)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"training__release_steps": 15258}, "step mismatch"),
        ({"training__seed": 43}, "seed mismatch"),
        ({"training__target_tokens": 10_000_000_000}, "total_tokens mismatch"),
        ({"training__tokens_per_update": 655360}, "target_tokens_per_update mismatch"),
        ({"training__consumed_tokens": 1}, "seen supervised tokens mismatch"),
        ({"model__spec_sha256": "9" * 64}, "_sophia_model_spec_sha256 mismatch"),
        ({"model__parameter_count": 42}, "_sophia_model_parameter_count mismatch"),
        ({"model__max_seq_len": 8192}, "seq_len mismatch"),
    ],
)
def test_formal_base_checkpoint_contract_rejects_a_loosened_protocol(
    tmp_path: Path, override: dict[str, object], match: str
) -> None:
    """Moving the goalposts must not admit the checkpoint that missed them.

    This is the check the protocol byte hash was believed to be performing.
    It never did: the fixture computed that hash from whichever protocol was
    on disk, so it agreed with itself no matter what the protocol said.
    """

    loosened = _protocol_file(tmp_path / "loosened_protocol.json", **override)
    with pytest.raises(RuntimeError, match=match):
        validate_parent_checkpoint_contract(_checkpoint(), protocol_path=loosened)
