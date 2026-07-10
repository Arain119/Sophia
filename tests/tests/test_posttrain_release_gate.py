from __future__ import annotations

import json

import pytest

from ml.errors import SophiaUsageError
from ml.core.engine.checkpointing import EngineCheckpoint
from ml.training.posttrain.release_gate import (
    POSTTRAIN_OUTPUT_RECIPE_COPY,
    POSTTRAIN_OUTPUT_RECIPE_SUMMARY,
    build_data_signature,
    build_export_signature,
    run_posttrain_release_preflight,
    validate_resume_checkpoint_contract,
    validate_signed_machine_recipe,
)
from ml.training.posttrain.release_config import build_runtime_sft_release_semantics
from ml.training.posttrain.types import PosttrainStageArgs


def _write_export_dir(path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "chat_template.jinja").write_text("", encoding="utf-8")


def _write_data_file(path) -> None:
    path.write_text('{"messages":[{"role":"user","content":"hi"}]}\n', encoding="utf-8")


def _sft_args(tmp_path, *, allow_missing_machine_recipe: int = 0) -> PosttrainStageArgs:
    export_dir = tmp_path / "export"
    _write_export_dir(export_dir)
    train_data = tmp_path / "train.jsonl"
    eval_data = tmp_path / "eval.jsonl"
    test_data = tmp_path / "test.jsonl"
    for path in (train_data, eval_data, test_data):
        _write_data_file(path)
    return PosttrainStageArgs(
        export_dir=str(export_dir),
        train_data=str(train_data),
        eval_data=str(eval_data),
        test_data=str(test_data),
        output_dir=str(tmp_path / "out"),
        allow_missing_machine_recipe=int(allow_missing_machine_recipe),
        curriculum_stage="sft_core",
        batch_size=1,
        accumulation_steps=1,
        learning_rate=5e-6,
        weight_decay=0.1,
        eval_interval=10,
        save_interval=20,
    )


def test_validate_signed_machine_recipe_uses_recipe_machine_settings_for_sft_semantics(
    tmp_path,
) -> None:
    args = _sft_args(tmp_path)
    machine_signature = {
        "schema": "posttrain_machine_adaptive_v1",
        "device_type": "cuda",
        "torch": "2.8.0",
        "cuda": "12.8",
        "platform": "linux",
        "machine": "x86_64",
    }
    recipe_payload = {
        "kind": "posttrain_machine_recipe",
        "stage": "sft",
        "release_semantics": {
            "curriculum_stage": "sft_core",
            "target_examples_per_update": 4,
            "learning_rate": 5e-6,
            "weight_decay": 0.1,
            "eval_interval": 10,
            "save_interval": 20,
            "resolved_max_seq_len": 4096,
            "max_seq_len_source": "curriculum:sft_core",
        },
        "machine_recipe": {
            "batch_size": 2,
            "accumulation_steps": 2,
            "gradient_checkpointing": 1,
        },
        "machine_signature": dict(machine_signature),
        "export_signature": build_export_signature(export_dir=args.export_dir),
        "data_signature": build_data_signature(
            train_data=args.train_data,
            eval_data=args.eval_data,
            test_data=args.test_data,
        ),
    }

    validate_signed_machine_recipe(
        args=args,
        stage="sft",
        resolved_max_seq_len=4096,
        recipe_payload=recipe_payload,
        current_machine_signature=machine_signature,
    )


def test_runtime_sft_semantics_rejects_incomplete_machine_recipe(tmp_path) -> None:
    with pytest.raises(ValueError, match="accumulation_steps"):
        build_runtime_sft_release_semantics(
            args=_sft_args(tmp_path),
            resolved_max_seq_len=4096,
            machine_recipe={"batch_size": 2},
        )


def test_validate_signed_machine_recipe_rejects_stage_mismatch(tmp_path) -> None:
    args = _sft_args(tmp_path)

    with pytest.raises(SophiaUsageError, match="stage mismatch"):
        validate_signed_machine_recipe(
            args=args,
            stage="sft",
            resolved_max_seq_len=4096,
            recipe_payload={
                "kind": "posttrain_machine_recipe",
                "stage": "pretrain",
                "release_semantics": {},
                "machine_recipe": {},
                "machine_signature": {},
                "export_signature": {},
                "data_signature": {},
            },
            current_machine_signature={},
        )


def test_posttrain_release_preflight_rejects_missing_recipe_for_release_run(
    monkeypatch,
    tmp_path,
) -> None:
    args = _sft_args(tmp_path, allow_missing_machine_recipe=0)
    seen: list[tuple[str, bool]] = []

    class _FakeSignature:
        def to_payload(self) -> dict[str, object]:
            return {"schema": "posttrain_machine_adaptive_v1"}

    monkeypatch.setattr(
        "ml.training.posttrain.release_gate.ensure_standard_stack",
        lambda *, mode, require_tensorboard=False: seen.append((mode, require_tensorboard)),
    )
    monkeypatch.setattr(
        "ml.training.posttrain.release_gate.require_cuda",
        lambda device: str(device),
    )
    monkeypatch.setattr(
        "ml.training.posttrain.release_gate.build_machine_adaptive_signature",
        lambda *, schema, device: _FakeSignature(),
    )

    with pytest.raises(SophiaUsageError, match="requires --machine_recipe_json"):
        run_posttrain_release_preflight(
            args=args,
            stage="sft",
            resolved_max_seq_len=4096,
            recipe_payload=None,
        )

    assert seen == [("train", True)]


def test_posttrain_release_preflight_allows_local_run_without_recipe(
    monkeypatch,
    tmp_path,
) -> None:
    args = _sft_args(tmp_path, allow_missing_machine_recipe=1)

    class _FakeSignature:
        def to_payload(self) -> dict[str, object]:
            return {"schema": "posttrain_machine_adaptive_v1", "device_type": "cuda"}

    monkeypatch.setattr(
        "ml.training.posttrain.release_gate.ensure_standard_stack",
        lambda *, mode, require_tensorboard=False: {"mode": mode, "require_tensorboard": require_tensorboard},
    )
    monkeypatch.setattr(
        "ml.training.posttrain.release_gate.require_cuda",
        lambda device: str(device),
    )
    monkeypatch.setattr(
        "ml.training.posttrain.release_gate.build_machine_adaptive_signature",
        lambda *, schema, device: _FakeSignature(),
    )

    payload = run_posttrain_release_preflight(
        args=args,
        stage="sft",
        resolved_max_seq_len=4096,
        recipe_payload=None,
    )

    assert payload == {
        "schema": "posttrain_machine_adaptive_v1",
        "device_type": "cuda",
    }


def test_posttrain_release_preflight_materializes_recipe_summary_and_copy(
    monkeypatch,
    tmp_path,
) -> None:
    args = _sft_args(tmp_path, allow_missing_machine_recipe=0)
    source_summary = tmp_path / "sft_recipe" / "summary.json"
    source_summary.parent.mkdir(parents=True, exist_ok=True)
    source_summary.write_text(
        json.dumps(
            {
                "kind": "sft_machine_selection",
                "selected_runs": [{"name": "release_sft"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class _FakeSignature:
        def to_payload(self) -> dict[str, object]:
            return {"schema": "posttrain_machine_adaptive_v1", "device_type": "cuda"}

    monkeypatch.setattr(
        "ml.training.posttrain.release_gate.ensure_standard_stack",
        lambda *, mode, require_tensorboard=False: {"mode": mode, "require_tensorboard": require_tensorboard},
    )
    monkeypatch.setattr(
        "ml.training.posttrain.release_gate.require_cuda",
        lambda device: str(device),
    )
    monkeypatch.setattr(
        "ml.training.posttrain.release_gate.build_machine_adaptive_signature",
        lambda *, schema, device: _FakeSignature(),
    )

    recipe_payload = {
        "kind": "posttrain_machine_recipe",
        "stage": "sft",
        "source_summary": str(source_summary),
        "release_semantics": {
            "curriculum_stage": "sft_core",
            "target_examples_per_update": 1,
            "learning_rate": 5e-6,
            "weight_decay": 0.1,
            "eval_interval": 10,
            "save_interval": 20,
            "resolved_max_seq_len": 4096,
            "max_seq_len_source": "curriculum:sft_core",
        },
        "machine_recipe": {
            "batch_size": 1,
            "accumulation_steps": 1,
            "gradient_checkpointing": 1,
        },
        "machine_signature": {"schema": "posttrain_machine_adaptive_v1", "device_type": "cuda"},
        "export_signature": build_export_signature(export_dir=args.export_dir),
        "data_signature": build_data_signature(
            train_data=args.train_data,
            eval_data=args.eval_data,
            test_data=args.test_data,
        ),
    }

    run_posttrain_release_preflight(
        args=args,
        stage="sft",
        resolved_max_seq_len=4096,
        recipe_payload=recipe_payload,
    )

    assert json.loads(
        (tmp_path / "out" / POSTTRAIN_OUTPUT_RECIPE_SUMMARY).read_text(encoding="utf-8")
    )["kind"] == "sft_machine_selection"
    assert json.loads(
        (tmp_path / "out" / POSTTRAIN_OUTPUT_RECIPE_COPY).read_text(encoding="utf-8")
    )["kind"] == "posttrain_machine_recipe"


def test_validate_resume_checkpoint_contract_rejects_stage_mismatch(tmp_path) -> None:
    args = _sft_args(tmp_path)
    resume_checkpoint = EngineCheckpoint(
        step=7,
        model={},
        optimizer={},
        scheduler=None,
        args={
            "export_dir": str(tmp_path / "export"),
            "train_data": str(tmp_path / "train.jsonl"),
            "eval_data": str(tmp_path / "eval.jsonl"),
            "test_data": str(tmp_path / "test.jsonl"),
            "curriculum_stage": "sft_core",
            "learning_rate": 5e-6,
            "weight_decay": 0.1,
            "eval_interval": 10,
            "save_interval": 20,
            "batch_size": 1,
            "accumulation_steps": 1,
            "_sophia_run_kind": "pretrain",
            "_sophia_resolved_max_seq_len": 4096,
        },
        rng=None,
        ema=None,
        train_state={},
    )

    with pytest.raises(SophiaUsageError, match="resume checkpoint stage mismatch"):
        validate_resume_checkpoint_contract(
            args=args,
            stage="sft",
            resolved_max_seq_len=4096,
            resume_checkpoint=resume_checkpoint,
        )


def test_validate_resume_checkpoint_contract_rejects_data_signature_mismatch(tmp_path) -> None:
    args = _sft_args(tmp_path)
    other_train = tmp_path / "other_train.jsonl"
    _write_data_file(other_train)
    resume_checkpoint = EngineCheckpoint(
        step=7,
        model={},
        optimizer={},
        scheduler=None,
        args={
            "export_dir": str(tmp_path / "export"),
            "train_data": str(other_train),
            "eval_data": str(tmp_path / "eval.jsonl"),
            "test_data": str(tmp_path / "test.jsonl"),
            "curriculum_stage": "sft_core",
            "learning_rate": 5e-6,
            "weight_decay": 0.1,
            "eval_interval": 10,
            "save_interval": 20,
            "batch_size": 1,
            "accumulation_steps": 1,
            "_sophia_run_kind": "sft",
            "_sophia_resolved_max_seq_len": 4096,
        },
        rng=None,
        ema=None,
        train_state={},
    )

    with pytest.raises(SophiaUsageError, match="resume checkpoint data signature mismatch"):
        validate_resume_checkpoint_contract(
            args=args,
            stage="sft",
            resolved_max_seq_len=4096,
            resume_checkpoint=resume_checkpoint,
        )


def test_validate_resume_checkpoint_contract_accepts_matching_sft_contract(tmp_path) -> None:
    args = _sft_args(tmp_path)
    resume_checkpoint = EngineCheckpoint(
        step=7,
        model={},
        optimizer={},
        scheduler=None,
        args={
            "export_dir": str(tmp_path / "export"),
            "train_data": str(tmp_path / "train.jsonl"),
            "eval_data": str(tmp_path / "eval.jsonl"),
            "test_data": str(tmp_path / "test.jsonl"),
            "curriculum_stage": "sft_core",
            "learning_rate": 5e-6,
            "weight_decay": 0.1,
            "eval_interval": 10,
            "save_interval": 20,
            "batch_size": 1,
            "accumulation_steps": 1,
            "_sophia_run_kind": "sft",
            "_sophia_resolved_max_seq_len": 4096,
        },
        rng=None,
        ema=None,
        train_state={},
    )

    validate_resume_checkpoint_contract(
        args=args,
        stage="sft",
        resolved_max_seq_len=4096,
        resume_checkpoint=resume_checkpoint,
    )
