from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ml.errors import SophiaUsageError
from ml.training.pretrain import bootstrap_io as bootstrap_io_mod
from ml.training.pretrain import model_setup as model_factory_mod
from ml.training.pretrain import output_policy as output_policy_mod
from ml.training.pretrain import release_gate as release_gate_mod
from ml.training.pretrain import semantic_defaults as semantic_defaults_mod
from ml.tasks.pretrain import pipeline as pretrain_mod
from ml.tasks.pretrain import observability as observability_mod
from ml.tasks.pretrain.observability import policy as observability_policy_mod
from ml.tasks.pretrain import pipeline_hooks as pipeline_hooks_mod
from ml.tasks.pretrain import stage_runtime as stage_runtime_mod
from ml.tasks.pretrain.planner import (
    CurriculumStage,
    StagePlan,
    RuntimeRecipe,
)
from ml.training.pretrain.run_config import (
    PretrainRunConfig,
)
from ml.tasks.pretrain.session import PretrainPipelineSnapshot
from ml.tasks.pretrain.run_spec_builder import (
    PretrainRunSpec,
    build_pretrain_run_spec,
)
from ml.training.pretrain.loop_controls import (
    PretrainDataControl,
    PretrainLoopControl,
)
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.tasks.pretrain import planner as stage_plan_mod
from ml.tasks.pretrain.session import PretrainSession
from ml.tasks.pretrain.runtime import run_runtime_preflight_impl
from ml.tasks.pretrain.session import (
    attach_pretrain_session,
    ensure_pretrain_session,
)
from ml.training.pretrain.profiles import (
    RELEASE_PROFILE,
    VALIDATION_PROFILE,
)
from ml.core.spec import ModelSpec
from ml.training.pretrain.output_policy import (
    OUTPUT_DIR_MARKER,
    prepare_output_dir_and_resume,
)
from ml.core.engine.run_dirs import (
    looks_like_run_output_dir,
    write_output_dir_marker,
)
from ml.training.pretrain.model_setup import (
    load_tokenizer,
)
from ml.data.token_shards.tokenizer_fingerprint import compute_tokenizer_bundle_sha1
from ml.integrations.adapters.hf.model import SophiaConfig, SophiaForCausalLM
from ml.training.pretrain.engine.step_execution import StepExecutionPlan
from ml.modeling.decoder_output import DecoderOutput


def _args(
    *,
    output_dir: str,
    overwrite_output_dir: int = 0,
    resume_from_checkpoint: str = "",
) -> PretrainRunConfig:
    return pretrain_mod.build_pretrain_args(
        data_path="dataset/train",
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        overwrite_output_dir=overwrite_output_dir,
    )


def _cfg(**overrides: object) -> PretrainRunConfig:
    data_path = str(overrides.pop("data_path", "dataset/pretrain_tokens"))
    tokenizer_path = str(overrides.pop("tokenizer_path", ""))
    output_dir = str(overrides.pop("output_dir", "out"))
    resume_from_checkpoint = str(overrides.pop("resume_from_checkpoint", ""))
    overwrite_output_dir = int(overrides.pop("overwrite_output_dir", 1))
    cfg = pretrain_mod.build_pretrain_args(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        overwrite_output_dir=overwrite_output_dir,
    )
    return replace(cfg, **overrides)


def test_prepare_output_dir_respects_overwrite_instead_of_auto_resume(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "ckpt_step7.pt"
    ckpt_path.write_bytes(b"checkpoint")
    write_output_dir_marker(
        repo_root=output_policy_mod.REPO_ROOT,
        output_dir=str(output_dir),
    )
    stale = output_dir / "stale.txt"
    stale.write_text("stale", encoding="utf-8")

    args = _args(output_dir=str(output_dir), overwrite_output_dir=1)
    resolution = prepare_output_dir_and_resume(args)

    assert resolution.output_dir == str(output_dir.resolve())
    assert resolution.resume_path is None
    assert not stale.exists()
    assert not ckpt_path.exists()
    assert (output_dir / OUTPUT_DIR_MARKER).exists()


def test_explicit_max_steps_caps_stage_execution_plans() -> None:
    plans = [
        StagePlan(
            stage=CurriculumStage(
                stage_index=0,
                seq_len=4096,
                start_tokens=0,
                end_tokens=10,
            ),
            recipe=RuntimeRecipe(
                seq_len=4096,
                batch_size=2,
                accumulation_steps=32,
                tokens_per_update=270336,
            ),
            start_step=0,
            end_step=100,
        ),
        StagePlan(
            stage=CurriculumStage(
                stage_index=1,
                seq_len=8192,
                start_tokens=10,
                end_tokens=20,
            ),
            recipe=RuntimeRecipe(
                seq_len=8192,
                batch_size=1,
                accumulation_steps=32,
                tokens_per_update=270336,
            ),
            start_step=100,
            end_step=250,
        ),
        StagePlan(
            stage=CurriculumStage(
                stage_index=2,
                seq_len=8192,
                start_tokens=20,
                end_tokens=30,
            ),
            recipe=RuntimeRecipe(
                seq_len=8192,
                batch_size=1,
                accumulation_steps=16,
                tokens_per_update=270336,
            ),
            start_step=250,
            end_step=403,
        ),
    ]

    capped = stage_plan_mod.apply_explicit_max_steps_cap(
        stage_execution_plans=plans,
        max_steps=280,
    )

    assert [int(plan.start_step) for plan in capped] == [0, 100, 250]
    assert [int(plan.end_step) for plan in capped] == [100, 250, 280]


def test_prepare_output_dir_rejects_resume_and_overwrite_together(tmp_path: Path) -> None:
    args = _args(
        output_dir=str(tmp_path / "run"),
        overwrite_output_dir=1,
        resume_from_checkpoint=str(tmp_path / "resume.pt"),
    )

    with pytest.raises(SophiaUsageError, match="overwrite_output_dir"):
        prepare_output_dir_and_resume(args)


def test_prepare_output_dir_refuses_to_delete_unrecognized_directory(tmp_path: Path) -> None:
    outside_dir = tmp_path / "manual-dir"
    outside_dir.mkdir()
    (outside_dir / "notes.txt").write_text("keep", encoding="utf-8")
    args = _args(output_dir=str(outside_dir), overwrite_output_dir=1)

    with pytest.raises(SophiaUsageError, match="refusing to overwrite"):
        prepare_output_dir_and_resume(args)

    assert (outside_dir / "notes.txt").exists()


def test_prepare_output_dir_refuses_markerless_export_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    export_dir = repo_root / "out" / "model-export"
    export_dir.mkdir(parents=True)
    (export_dir / "config.json").write_text("{}", encoding="utf-8")
    (export_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(output_policy_mod, "REPO_ROOT", str(repo_root))
    args = _args(output_dir=str(export_dir), overwrite_output_dir=1)

    with pytest.raises(SophiaUsageError, match="refusing to overwrite"):
        prepare_output_dir_and_resume(args)

    assert (export_dir / "config.json").exists()
    assert (export_dir / "tokenizer.json").exists()


def test_prepare_output_dir_allows_reusing_existing_marked_run_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "recognized-run"
    output_dir.mkdir()
    write_output_dir_marker(
        repo_root=output_policy_mod.REPO_ROOT,
        output_dir=str(output_dir),
    )
    marker = output_dir / OUTPUT_DIR_MARKER
    (output_dir / "stale.txt").write_text("stale", encoding="utf-8")

    args = _args(output_dir=str(output_dir), overwrite_output_dir=1)
    resolution = prepare_output_dir_and_resume(args)

    assert resolution.output_dir == str(output_dir.resolve())
    assert resolution.resume_path is None
    assert not (output_dir / "stale.txt").exists()
    assert marker.exists()


def test_looks_like_run_output_dir_returns_false_for_unreadable_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    marker = output_dir / OUTPUT_DIR_MARKER
    marker.write_text("{}", encoding="utf-8")

    def _deny_open(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("builtins.open", _deny_open)

    assert (
        looks_like_run_output_dir(
            repo_root=output_policy_mod.REPO_ROOT,
            path=str(output_dir),
        )
        is False
    )


def test_clear_cuda_collects_gc_and_cuda_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: calls.append("ipc"))

    pipeline_hooks_mod.clear_cuda()

    assert calls == ["gc", "sync", "empty", "ipc", "gc"]


def test_runtime_preflight_always_clears_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TinyModel(torch.nn.Module):
        def __init__(self, _cfg: object) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, *, input_ids, labels, use_cache=False):
            del labels, use_cache
            loss = self.weight * input_ids.float().mean()
            return DecoderOutput(loss=loss)

    cleared: list[str] = []
    hooks = SimpleNamespace(
        runtime=SimpleNamespace(
            runtime_preflight_config=pretrain_mod.runtime_preflight_config,
            decoder_model_cls=_TinyModel,
            clear_cuda=lambda: cleared.append("cleared"),
        )
    )
    pipeline = SimpleNamespace(
        device=torch.device("cpu"),
        args=_cfg(output_dir="out"),
    )

    run_runtime_preflight_impl(pipeline, hooks=hooks)

    assert cleared == ["cleared"]


def test_prepare_output_dir_refuses_implicit_auto_resume_on_derived_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    out_root = repo_root / "out"
    output_dir = out_root / f"sophia_{int(ModelSpec.default().dim)}_export"
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "ckpt_step7.pt"
    ckpt_path.write_bytes(b"checkpoint")

    monkeypatch.setattr(output_policy_mod, "REPO_ROOT", str(repo_root))

    args = _args(output_dir="", overwrite_output_dir=0, resume_from_checkpoint="")

    with pytest.raises(SophiaUsageError, match="auto-derived output_dir already contains a checkpoint"):
        prepare_output_dir_and_resume(args)

    assert str(output_dir.resolve()).endswith(f"sophia_{int(ModelSpec.default().dim)}_export")


def test_load_manifest_or_die_prefers_dataset_local_matching_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset" / "pretrain"
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"
    tokenizer_dir = dataset_root / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer.json").write_text('{"kind":"dummy"}', encoding="utf-8")
    (tokenizer_dir / "tokenizer_config.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (tokenizer_dir / "chat_template.jinja").write_text("{{ bos_token }}\n", encoding="utf-8")
    tokenizer_sha1 = compute_tokenizer_bundle_sha1(str(tokenizer_dir))

    for split_dir in (train_dir, val_dir, test_dir):
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "dtype": "int32",
                    "eos_token_id": 3,
                    "tokenizer_sha1": tokenizer_sha1,
                    "total_tokens": 1,
                    "shards": [{"path": "shard_00000.bin", "tokens": 1}],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (split_dir / "shard_00000.bin").write_bytes((3).to_bytes(4, "little", signed=True))

    (dataset_root / "dataset.json").write_text(
        json.dumps(
            {
                "kind": "sophia_token_shards_dataset",
                "version": 1,
                "tokenizer": {"dir": "tokenizer"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        data_path=str(train_dir),
        tokenizer_path="",
    )
    loaded = bootstrap_io_mod.load_manifest_or_die(args=args)
    assert str(loaded.path) == str((train_dir / "manifest.json").resolve())
    assert str(loaded.tokenizer_path) == str(tokenizer_dir.resolve())
    assert str(loaded.manifest.tokenizer_sha1) == tokenizer_sha1


def test_pretrain_parse_args_respects_output_and_resume_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ml.cli.train import build_parser

    argv = [
        "pretrain.py",
        "--data_path",
        "dataset/train",
        "--output_dir",
        "out/run_a",
        "--resume_from_checkpoint",
        "out/run_a/checkpoints/ckpt_step7.pt",
        "--overwrite_output_dir",
        "1",
        "--machine_recipe_json",
        "out/recipes/pretrain_recipe.json",
    ]
    monkeypatch.setattr("sys.argv", argv)

    parsed = build_parser().parse_args()
    args = pretrain_mod.build_pretrain_args(
        data_path=str(parsed.data_path),
        tokenizer_path=str(parsed.tokenizer_path),
        output_dir=str(parsed.output_dir),
        resume_from_checkpoint=str(parsed.resume_from_checkpoint),
        overwrite_output_dir=int(parsed.overwrite_output_dir),
        machine_recipe_json=str(parsed.machine_recipe_json),
        profile=RELEASE_PROFILE,
    )

    assert str(args.data_path) == "dataset/train"
    assert str(args.output_dir) == "out/run_a"
    assert str(args.resume_from_checkpoint) == "out/run_a/checkpoints/ckpt_step7.pt"
    assert str(args.machine_recipe_json) == "out/recipes/pretrain_recipe.json"
    assert int(args.overwrite_output_dir) == 1


def test_build_pretrain_args_accepts_validation_profile() -> None:
    profile = VALIDATION_PROFILE
    args = pretrain_mod.build_pretrain_args(
        data_path="dataset/pretrain_tokens",
        output_dir="out/validate",
        overwrite_output_dir=1,
        profile=profile,
    )

    assert isinstance(args, PretrainRunConfig)
    assert int(args.total_tokens) == int(profile.curriculum.default_total_tokens)
    assert int(args.max_seq_len) == int(profile.curriculum.max_seq_len)
    assert int(args.seq_len) == int(profile.curriculum.train_seq_len)
    assert int(args.muon_ns_steps) == 4
    assert int(args.target_tokens_per_update) == int(
        semantic_defaults_mod.PINNED_SEMANTIC_TARGET_TOKENS_PER_UPDATE
    )
    assert float(args.learning_rate) == pytest.approx(
        semantic_defaults_mod.PINNED_SEMANTIC_LEARNING_RATE
    )
    assert float(args.weight_decay) == pytest.approx(
        semantic_defaults_mod.PINNED_SEMANTIC_WEIGHT_DECAY
    )
    assert str(args.output_dir) == "out/validate"
    assert int(args.overwrite_output_dir) == 1


def test_pretrain_run_config_payload_serializes_attached_profile() -> None:
    profile = VALIDATION_PROFILE
    args = pretrain_mod.build_pretrain_args(
        data_path="dataset/pretrain_tokens",
        profile=profile,
    )

    payload = args.to_payload()

    assert isinstance(payload, dict)
    assert isinstance(payload["_sophia_pretrain_profile"], dict)
    assert str(payload["_sophia_pretrain_profile"]["name"]) == str(profile.name)
    assert str(payload["_sophia_pretrain_profile"]["kind"]) == str(profile.kind)


def test_build_pretrain_run_spec_projects_run_contract() -> None:
    profile = VALIDATION_PROFILE
    args = pretrain_mod.build_pretrain_args(
        data_path="dataset/pretrain_tokens",
        output_dir="out/validate",
        overwrite_output_dir=1,
        profile=profile,
    )
    args._sophia_run_kind = "pretrain_check"
    args.device = "cuda:1"

    run_spec = build_pretrain_run_spec(args)

    assert isinstance(run_spec, PretrainRunSpec)
    assert str(run_spec.run_kind) == "pretrain_check"
    assert run_spec.run.model == profile.model
    assert str(run_spec.data_path) == "dataset/pretrain_tokens"
    assert str(run_spec.requested_device) == "cuda:1"
    assert bool(run_spec.overwrite_output_dir) is True
    assert int(run_spec.seq_len) == int(profile.curriculum.train_seq_len)


def test_ensure_pretrain_session_lifts_pipeline_snapshot_state() -> None:
    args = pretrain_mod.build_pretrain_args(
        data_path="dataset/pretrain_tokens",
        output_dir="out/validate",
        overwrite_output_dir=1,
        profile=VALIDATION_PROFILE,
    )
    pipeline = pretrain_mod.PretrainPipeline(args=args)
    pipeline.device = torch.device("cpu")
    pipeline.base_dtype = torch.float32
    pipeline.runtime_metadata = {"precision_stack": "test_runtime"}
    pipeline.output_dir = "/tmp/sophia_pretrain_session"
    pipeline.resume_path = None
    pipeline.manifest_path = "/tmp/train_manifest.json"
    pipeline.manifest = object()
    pipeline.tokenizer = object()
    pipeline.tokenizer_path = "/tmp/tokenizer"
    pipeline.seq_len = 128
    pipeline.vocab_size = 49152
    pipeline.max_steps = 4
    pipeline.target_tokens_per_update = 8192

    session = ensure_pretrain_session(pipeline)

    assert isinstance(session, PretrainSession)
    assert pipeline.session is session
    assert str(session.output_dir) == "/tmp/sophia_pretrain_session"
    assert session.device.type == "cpu"
    assert session.base_dtype == torch.float32
    assert session.run.runtime_metadata == {"precision_stack": "test_runtime"}
    assert int(session.seq_len) == 128
    assert int(session.vocab_size) == 49152
    assert int(session.runtime_plan.max_steps) == 4
    assert int(session.runtime_plan.target_tokens_per_update) == 8192
    assert session.spec.run.model == VALIDATION_PROFILE.model


def test_attach_pretrain_session_keeps_resolved_runtime_plan_max_steps() -> None:
    args = pretrain_mod.build_pretrain_args(
        data_path="dataset/pretrain_tokens",
        output_dir="out/validate",
        overwrite_output_dir=1,
        profile=VALIDATION_PROFILE,
    )
    pipeline = pretrain_mod.PretrainPipeline(args=args)
    pipeline.device = torch.device("cpu")
    pipeline.base_dtype = torch.float32
    pipeline.output_dir = "/tmp/sophia_pretrain_session_max_steps"

    session = ensure_pretrain_session(pipeline)
    session.runtime_state.max_steps = -1
    session.runtime_plan.max_steps = 4

    attach_pretrain_session(pipeline, session)

    assert int(pipeline.args.max_steps) == 4


def test_pretrain_pipeline_snapshot_is_immutable_snapshot() -> None:
    snapshot = PretrainPipelineSnapshot(
        output_dir=" out/run ",
        seq_len="128",
        max_steps="4",
        target_tokens_per_update="8192",
    )

    assert str(snapshot.output_dir) == " out/run "
    assert int(snapshot.seq_len) == 128
    assert int(snapshot.max_steps) == 4
    assert int(snapshot.target_tokens_per_update) == 8192

    with pytest.raises(FrozenInstanceError):
        snapshot.seq_len = 256


def test_pretrain_pipeline_property_update_replaces_snapshot() -> None:
    args = pretrain_mod.build_pretrain_args(
        data_path="dataset/pretrain_tokens",
        output_dir="out/validate",
        overwrite_output_dir=1,
        profile=VALIDATION_PROFILE,
    )
    pipeline = pretrain_mod.PretrainPipeline(args=args)
    original_snapshot = pipeline.snapshot
    pipeline.session = object()

    pipeline.seq_len = "256"

    assert pipeline.session is None
    assert pipeline.snapshot is not original_snapshot
    assert int(pipeline.snapshot.seq_len) == 256
    assert int(original_snapshot.seq_len) != 256


def test_pretrain_loop_controls_bind_runtime_state() -> None:
    args = _cfg(
        tokenizer_path="tok",
        eval_data_path="dataset/eval",
        test_data_path="dataset/test",
        batch_size=4,
        accumulation_steps=8,
        total_tokens=65536,
        eval_interval=12,
        eval_steps=7,
        log_interval=9,
        save_interval=11,
        save_total_limit=3,
        save_weights_steps=13,
        save_weights_total_limit=2,
        warmup_steps=5,
        max_grad_norm=1.25,
        grad_clip_mode="agc",
        agc_clip=0.2,
        agc_eps=1e-4,
        agc_exclude_bias_and_norm=0,
        stability_guard_enabled=1,
        stability_guard_loss_window=3,
        stability_guard_loss_max=12.5,
        stability_guard_grad_norm_max=20.0,
        stability_guard_grad_window=100,
        stability_guard_grad_spike_limit=30,
        stability_guard_min_step=100,
    )
    state = PretrainRuntimeState.from_config(args)

    data = PretrainDataControl.from_bound(args, state)
    loop = PretrainLoopControl.from_bound(args, state)

    assert str(data.tokenizer_path) == "tok"
    assert str(data.eval_data_path) == "dataset/eval"
    assert str(data.test_data_path) == "dataset/test"
    assert int(data.batch_size) == 4
    assert int(data.eval_interval) == 12
    assert int(data.eval_steps) == 7
    assert int(loop.total_tokens) == 65536
    assert int(loop.accumulation_steps) == 8
    assert float(loop.max_grad_norm) == pytest.approx(1.25)
    assert str(loop.grad_clip_mode) == "agc"
    assert float(loop.agc_clip) == pytest.approx(0.2)
    assert float(loop.agc_eps) == pytest.approx(1e-4)
    assert int(loop.agc_exclude_bias_and_norm) == 0
    assert int(loop.log_interval) == 9
    assert int(loop.save_interval) == 11
    assert int(loop.save_total_limit) == 3
    assert int(loop.save_weights_steps) == 13
    assert int(loop.save_weights_total_limit) == 2
    assert int(loop.enable_checkpoints) == 1
    assert int(loop.warmup_steps) == 5
    assert int(loop.stability_guard_enabled) == 1
    assert int(loop.stability_guard_loss_window) == 3
    assert float(loop.stability_guard_loss_max) == pytest.approx(12.5)
    assert float(loop.stability_guard_grad_norm_max) == pytest.approx(20.0)
    assert int(loop.stability_guard_grad_window) == 100
    assert int(loop.stability_guard_grad_spike_limit) == 30
    assert int(loop.stability_guard_min_step) == 100


def test_pretrain_runtime_state_does_not_mutate_cfg_until_project() -> None:
    args = _cfg(batch_size=4, accumulation_steps=8)
    state = PretrainRuntimeState.from_config(args)

    state.batch_size = 7
    state.accumulation_steps = 3

    assert int(args.batch_size) == 4
    assert int(args.accumulation_steps) == 8
    assert int(state.batch_size) == 7
    assert int(state.accumulation_steps) == 3

    projected = state.project_run_config(args)

    assert int(projected.batch_size) == 7
    assert int(projected.accumulation_steps) == 3


def test_auto_observability_policy_preserves_explicit_zero_intervals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        observability_policy_mod,
        "load_step_time_s_from_machine_artifacts",
        lambda *, output_dir: 7.5,
    )
    args = _cfg(
        output_dir=str(tmp_path),
        eval_interval=0,
        eval_steps=0,
        save_interval=0,
        log_interval=100,
    )

    resolved = observability_policy_mod.auto_observability_policy(
        args=args,
        output_dir=str(tmp_path),
        seq_len=int(args.seq_len),
        max_steps=1100,
    )

    assert int(resolved.eval_interval) == 0
    assert int(resolved.eval_steps) == 0
    assert int(resolved.save_interval) == 0
    assert int(resolved.log_interval) > 0


def test_build_pretrain_args_pins_same_semantic_lr_across_profiles() -> None:
    validation = VALIDATION_PROFILE
    release = RELEASE_PROFILE

    validation_args = pretrain_mod.build_pretrain_args(
        data_path="dataset/pretrain_tokens",
        output_dir="out/validate",
        overwrite_output_dir=1,
        profile=validation,
    )
    release_args = pretrain_mod.build_pretrain_args(
        data_path="dataset/pretrain_tokens",
        output_dir="out/release",
        overwrite_output_dir=1,
        profile=release,
    )

    assert int(validation_args.target_tokens_per_update) == int(
        semantic_defaults_mod.PINNED_SEMANTIC_TARGET_TOKENS_PER_UPDATE
    )
    assert float(validation_args.learning_rate) == pytest.approx(
        semantic_defaults_mod.PINNED_SEMANTIC_LEARNING_RATE
    )
    assert float(validation_args.adam_eps) == pytest.approx(
        semantic_defaults_mod.PINNED_SEMANTIC_ADAM_EPS
    )
    assert int(release_args.target_tokens_per_update) == int(
        semantic_defaults_mod.PINNED_RELEASE_PRETRAIN_TARGET_TOKENS_PER_UPDATE
    )
    assert float(release_args.learning_rate) == pytest.approx(
        semantic_defaults_mod.PINNED_RELEASE_PRETRAIN_LEARNING_RATE
    )
    assert float(release_args.adam_eps) == pytest.approx(
        semantic_defaults_mod.PINNED_RELEASE_PRETRAIN_ADAM_EPS
    )
    assert int(release_args.batch_size) == 3
    assert int(release_args.accumulation_steps) == 22
    assert int(release_args.target_tokens_per_microbatch) == 12288
    assert int(release_args.dataloader_num_workers) == 0
    assert int(release_args.dataloader_prefetch_factor) == 0
    assert int(release_args.dataloader_persistent_workers) == 0
    assert int(release_args.shard_preload) == 1
    assert int(release_args.shard_preload_bytes) == 4_194_304
    assert str(release_args.step_execution_backend) == "inductor"


def test_build_runner_uses_release_policy_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class _Runner:
        execution_plan: StepExecutionPlan

        def __init__(self, **kwargs) -> None:
            seen["runner"] = dict(kwargs)
            self.execution_plan = kwargs["execution_plan"]

    class _Param:
        device = torch.device("cuda")

    class _Model:
        def parameters(self):
            yield _Param()

    monkeypatch.setattr(pipeline_hooks_mod, "build_step_runner", lambda **kwargs: _Runner(**kwargs))

    args = _cfg(
        seq_len=4096,
        max_steps=64,
    )
    pretrain_mod._build_runner(args=args, model=_Model(), base_dtype=torch.bfloat16)

    assert "runner" in seen
    plan = seen["runner"]["execution_plan"]
    assert str(plan.backend) == "inductor"


def test_build_runner_allows_explicit_inductor_experiment_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Runner:
        execution_plan: StepExecutionPlan

        def __init__(self, **kwargs) -> None:
            seen["runner"] = dict(kwargs)
            self.execution_plan = kwargs["execution_plan"]

    class _Param:
        device = torch.device("cuda")

    class _Model:
        def parameters(self):
            yield _Param()

    monkeypatch.setattr(
        pipeline_hooks_mod,
        "build_step_runner",
        lambda **kwargs: _Runner(**kwargs),
    )
    args = _cfg(
        output_dir="out/release",
        step_execution_backend="inductor",
        max_steps=1,
    )
    pretrain_mod._build_runner(args=args, model=_Model(), base_dtype=torch.bfloat16)

    assert "runner" in seen
    plan = seen["runner"]["execution_plan"]
    assert str(plan.backend) == "inductor"


def test_build_runner_keeps_eager_runner_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class _Runner:
        execution_plan: StepExecutionPlan

        def __init__(self, **kwargs) -> None:
            seen["runner"] = dict(kwargs)
            self.execution_plan = kwargs["execution_plan"]

    class _Param:
        device = torch.device("cpu")

    class _Model:
        def parameters(self):
            yield _Param()

    monkeypatch.setattr(pipeline_hooks_mod, "build_step_runner", lambda **kwargs: _Runner(**kwargs))

    args = _cfg(
        seq_len=4096,
        max_steps=64,
    )
    pretrain_mod._build_runner(args=args, model=_Model(), base_dtype=torch.bfloat16)

    assert str(seen["runner"]["execution_plan"].backend) == "eager"


def test_build_runner_keeps_cuda_stage_compiled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Runner:
        execution_plan: StepExecutionPlan

        def __init__(self, **kwargs) -> None:
            seen["runner"] = dict(kwargs)
            self.execution_plan = kwargs["execution_plan"]

    class _Param:
        device = torch.device("cuda")

    class _Model:
        def parameters(self):
            yield _Param()

    monkeypatch.setattr(pipeline_hooks_mod, "build_step_runner", lambda **kwargs: _Runner(**kwargs))

    args = _cfg(
        seq_len=8192,
        max_steps=64,
    )
    pretrain_mod._build_runner(args=args, model=_Model(), base_dtype=torch.bfloat16)

    assert str(seen["runner"]["execution_plan"].backend) == "eager"


def test_build_runner_keeps_large_stage_compiled_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Runner:
        execution_plan: StepExecutionPlan

        def __init__(self, **kwargs) -> None:
            seen["runner"] = dict(kwargs)
            self.execution_plan = kwargs["execution_plan"]

    class _Param:
        device = torch.device("cuda")

    class _Model:
        def parameters(self):
            yield _Param()

    monkeypatch.setattr(pipeline_hooks_mod, "build_step_runner", lambda **kwargs: _Runner(**kwargs))

    args = _cfg(
        seq_len=8192,
        max_steps=64,
    )
    pretrain_mod._build_runner(args=args, model=_Model(), base_dtype=torch.bfloat16)

    assert str(seen["runner"]["execution_plan"].backend) == "eager"


def test_build_runner_keeps_short_budget_compiled_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Runner:
        execution_plan: StepExecutionPlan

        def __init__(self, **kwargs) -> None:
            seen["runner"] = dict(kwargs)
            self.execution_plan = kwargs["execution_plan"]

    class _Param:
        device = torch.device("cuda")

    class _Model:
        def parameters(self):
            yield _Param()

    monkeypatch.setattr(pipeline_hooks_mod, "build_step_runner", lambda **kwargs: _Runner(**kwargs))

    args = _cfg(
        seq_len=128,
        max_steps=1,
    )
    pretrain_mod._build_runner(args=args, model=_Model(), base_dtype=torch.bfloat16)

    assert str(seen["runner"]["execution_plan"].backend) == "eager"


def test_stage_run_args_dict_persists_machine_runtime_backend() -> None:
    args = _cfg(
        seq_len=4096,
        max_steps=64,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=0,
        dataloader_persistent_workers=0,
        shard_preload=1,
        shard_preload_bytes=4096,
    )
    runner = type(
        "_Runner",
        (),
        {"execution_plan": StepExecutionPlan(backend="inductor", reason="test")},
    )()

    payload = stage_runtime_mod.stage_run_args_dict(
        args=args,
        tokens_per_update=32768,
        runner=runner,
    )

    runtime_payload = payload.get("_sophia_machine_runtime")
    assert isinstance(runtime_payload, dict)
    assert str(runtime_payload.get("step_execution_backend")) == "inductor"
    assert int(runtime_payload.get("dataloader_num_workers")) == 0
    assert int(runtime_payload.get("dataloader_prefetch_factor")) == 0
    assert int(runtime_payload.get("dataloader_persistent_workers")) == 0
    assert int(runtime_payload.get("shard_preload")) == 1
    assert int(runtime_payload.get("shard_preload_bytes")) == 4096


def test_machine_recipe_preserves_explicit_step_execution_backend() -> None:
    recipe_payload = {
        "machine_recipe": {
            "accumulation_steps": 64,
            "batch_size": 1,
            "gradient_checkpointing": 1,
            "gradient_checkpointing_exclude_first": 0,
            "gradient_checkpointing_exclude_last": 0,
            "loss_chunk_size": 0,
        },
        "machine_runtime": {
            "step_execution_backend": "inductor",
            "dataloader_num_workers": 0,
            "dataloader_prefetch_factor": 0,
            "dataloader_persistent_workers": 0,
            "shard_preload": 1,
            "shard_preload_bytes": 4096,
        },
    }
    args = _cfg(
        step_execution_backend="eager",
        batch_size=1,
        accumulation_steps=1,
    )

    projected = release_gate_mod.apply_pretrain_machine_recipe(
        args=args,
        recipe_payload=recipe_payload,
    )

    assert str(projected.step_execution_backend) == "eager"
    assert int(projected.dataloader_num_workers) == 0
    assert int(projected.shard_preload_bytes) == 4096
    diff = projected._sophia_recipe_override_diff
    assert isinstance(diff, dict)
    assert "step_execution_backend" not in diff


def test_int_arg_preserve_zero_keeps_zero_prefetch_factor() -> None:
    args = argparse.Namespace(dataloader_prefetch_factor=0)

    assert (
        int(
            observability_mod.int_arg_preserve_zero(
                args, "dataloader_prefetch_factor", default=2
            )
        )
        == 0
    )


def test_apply_stage_recipe_rescales_lr_with_tokens_per_update() -> None:
    model = SophiaForCausalLM(
        SophiaConfig(
            vocab_size=128,
            dim=64,
            n_layers=2,
            n_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            rope_head_dim=16,
            rope_theta=10000.0,
            original_seq_len=4096,
            rope_factor=16.0,
            beta_fast=32,
            beta_slow=1,
            norm_eps=1e-6,
            max_seq_len=4096,
            max_batch_size=2,
            ffn_hidden=128,
        ),
        runtime_max_seq_len=4096,
    )
    args = _cfg(
        seq_len=2048,
        batch_size=1,
        accumulation_steps=2,
        learning_rate=1.9091883092036785e-4,
    )
    optimizer = pretrain_mod.torch.optim.SGD(model.parameters(), lr=float(args.learning_rate))
    scheduler = pretrain_mod.torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _step: 1.0,
    )
    recipe = RuntimeRecipe(
        seq_len=4096,
        batch_size=1,
        accumulation_steps=2,
        tokens_per_update=8192,
    )

    args = stage_runtime_mod.apply_stage_recipe(
        args=args,
        model=model,
        recipe=recipe,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    expected_lr = float(args.learning_rate) * math.sqrt(2.0)
    assert float(args.learning_rate) == pytest.approx(
        semantic_defaults_mod.PINNED_SEMANTIC_LEARNING_RATE
    )
    assert int(args.seq_len) == 4096
    assert int(args.batch_size) == 1
    assert int(args.accumulation_steps) == 2
    assert float(optimizer.param_groups[0]["lr"]) == pytest.approx(expected_lr)
    assert float(scheduler.base_lrs[0]) == pytest.approx(expected_lr)


def test_load_tokenizer_applies_requested_model_max_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tok_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    class DummyTokenizer:
        def __init__(self) -> None:
            self.padding_side = "left"
            self.truncation_side = "right"
            self.pad_token_id = None
            self.eos_token_id = 3
            self.eos_token = "<eos>"
            self.model_max_length = 4096
            self.init_kwargs: dict[str, int] = {}

    dummy = DummyTokenizer()

    def _load_local_tokenizer(
        *_args,
        **_kwargs,
    ):
        assert _kwargs["padding_side"] == "right"
        assert _kwargs["truncation_side"] == "left"
        assert _kwargs["model_max_length"] == 4096
        return dummy

    monkeypatch.setattr(model_factory_mod, "load_local_tokenizer", _load_local_tokenizer)

    tok = load_tokenizer(str(tok_dir), model_max_length=4096)
    assert tok is dummy


def test_runtime_max_batch_size_prefers_explicit_batch_size() -> None:
    args = PretrainRunConfig(data_path="dataset/pretrain_tokens", batch_size=12, auto_batch_size_max=48)
    assert int(model_factory_mod._positive_int_or_default(args.batch_size, default=4)) == 12


def test_runtime_max_batch_size_ignores_auto_batch_cap_by_default() -> None:
    args = PretrainRunConfig(data_path="dataset/pretrain_tokens", batch_size=0, auto_batch_size_max=48)
    assert int(model_factory_mod._positive_int_or_default(args.batch_size, default=4)) == 4


def test_build_decoder_config_sets_runtime_max_batch_size() -> None:
    class _Tokenizer:
        vocab_size = 49152
        bos_token_id = 1
        eos_token_id = 2
        unk_token_id = 3
        pad_token_id = 0

    args = PretrainRunConfig(
        data_path="dataset/pretrain_tokens",
        max_seq_len=4096,
        batch_size=6,
        auto_batch_size_max=32,
    )
    cfg, _ = model_factory_mod.build_decoder_config(args=args, tokenizer=_Tokenizer())
    assert int(cfg.max_batch_size) == 6


def test_build_decoder_config_uses_attached_validation_profile() -> None:
    class _Tokenizer:
        vocab_size = 49152
        bos_token_id = 1
        eos_token_id = 2
        unk_token_id = 3
        pad_token_id = 0

    profile = VALIDATION_PROFILE
    args = pretrain_mod.build_pretrain_args(
        data_path="dataset/pretrain_tokens",
        profile=profile,
    )
    args.batch_size = 2
    cfg, _ = model_factory_mod.build_decoder_config(args=args, tokenizer=_Tokenizer())

    assert int(cfg.dim) == int(profile.model.dim)
    assert int(cfg.n_layers) == int(profile.model.n_layers)
    assert int(cfg.max_seq_len) == int(profile.curriculum.max_seq_len)


def test_load_or_init_model_uses_seq_len_for_runtime_cache_budget() -> None:
    class _Tokenizer:
        vocab_size = 49152
        bos_token_id = 1
        eos_token_id = 2
        unk_token_id = 3
        pad_token_id = 0

    profile = VALIDATION_PROFILE
    profile = replace(
        profile,
        model=profile.model.with_overrides(max_seq_len=64),
        curriculum=replace(
            profile.curriculum,
            max_seq_len=64,
            train_seq_len=32,
            train_seq_stages=(32,),
            stage_token_weights=(32,),
        ),
    )
    args = PretrainRunConfig(
        data_path="dataset/pretrain_tokens",
        max_seq_len=64,
        seq_len=32,
        batch_size=2,
        auto_batch_size_max=32,
        _sophia_pretrain_profile=profile,
    )
    cfg, _ = model_factory_mod.build_decoder_config(args=args, tokenizer=_Tokenizer())
    model = model_factory_mod.load_or_init_model(
        args=args,
        decoder_config=cfg,
        device=model_factory_mod.torch.device("cpu"),
        base_dtype=model_factory_mod.torch.float32,
    )

    assert int(model.config.max_seq_len) == 64
    assert int(model.model.args.max_seq_len) == 32
