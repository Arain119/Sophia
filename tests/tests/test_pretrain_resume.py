from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ml.errors import SophiaUsageError
from ml.core.engine import checkpointing as checkpoint_mod
from ml.core.engine.checkpointing import EngineCheckpoint
from ml.core.engine.context import RunContext
from ml.core.engine.machine_adaptive import (
    build_machine_adaptive_signature,
    machine_signature_payload,
)
from ml.tasks.pretrain import pipeline as pretrain_mod
from ml.tasks.pretrain.run_spec_builder import build_pretrain_run_spec
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.resources import (
    LoadedPretrainManifest,
    LoadedPretrainTokenizer,
)
from ml.tasks.pretrain.session import (
    PretrainPipelineSnapshot,
    PretrainRuntimeState,
    PretrainSession,
)
from ml.tasks.pretrain import transition
from ml.tasks.pretrain.planner import (
    CurriculumStage,
    StagePlan,
    RuntimeRecipe,
)
from ml.training.pretrain import bootstrap_io as bootstrap_io_mod
from ml.training.pretrain import loop_runtime as loop_runtime_mod
from ml.training.pretrain import output_policy as output_policy_mod
from ml.training.pretrain import runtime_bootstrap as runtime_bootstrap_mod
from ml.training.pretrain.runtime_bootstrap import PretrainRuntimeBootstrap
from ml.training.pretrain import resume_loader as resume_loader_mod
from ml.training.pretrain import resume_restore as resume_restore_mod
from ml.training.pretrain.release_config import (
    CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD,
)
from ml.tasks.pretrain import runtime as runtime_prepare_mod
from ml.tasks.pretrain import observability as observability_mod
from ml.tasks.pretrain import stage_runtime as stage_runtime_mod
from ml.training.pretrain.train_state import PretrainTrainState


class _FakeOptimizer:
    def __init__(self, groups: int) -> None:
        self.groups = int(groups)
        self.loaded = False

    def load_state_dict(self, state_dict: dict) -> None:
        if int(state_dict.get("expected_groups", 0) or 0) != int(self.groups):
            raise RuntimeError("loaded state dict has a different number of parameter groups")
        self.loaded = True


class _RuntimeRecipeControlMixin:
    def supports_loss_chunk_size(self) -> bool:
        return False

    def supports_checkpoint_excludes(self) -> bool:
        return True

    def runtime_recipe_knobs(self) -> tuple[int, int, int]:
        return (0, 0, 0)

    def apply_runtime_recipe_knobs(
        self,
        *,
        loss_chunk_size: int,
        gradient_checkpointing_exclude_first: int,
        gradient_checkpointing_exclude_last: int,
    ) -> tuple[int, int, int]:
        del loss_chunk_size
        return (
            0,
            int(gradient_checkpointing_exclude_first),
            int(gradient_checkpointing_exclude_last),
        )

    def ensure_runtime_max_seq_len(self, max_seq_len: int) -> None:
        del max_seq_len

    def runtime_max_seq_len(self) -> int:
        return 0

    def runtime_prefix_replay_len(self) -> int:
        return 0

    def sync_runtime_batch_capacity(self, max_batch_size: int) -> int:
        return int(max_batch_size)


class _RuntimeRecipeLinear(_RuntimeRecipeControlMixin, torch.nn.Linear):
    pass


def _pretrain_cfg(**overrides: object) -> PretrainRunConfig:
    data_path = str(overrides.pop("data_path", "dataset/train"))
    tokenizer_path = str(overrides.pop("tokenizer_path", ""))
    output_dir = str(overrides.pop("output_dir", "out"))
    resume_from_checkpoint = str(overrides.pop("resume_from_checkpoint", ""))
    overwrite_output_dir = int(overrides.pop("overwrite_output_dir", 1))
    args = pretrain_mod.build_pretrain_args(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        overwrite_output_dir=overwrite_output_dir,
    )
    return replace(args, **overrides)


def _pipeline(args: PretrainRunConfig | None = None) -> pretrain_mod.PretrainPipeline:
    return pretrain_mod.PretrainPipeline(args=args or _pretrain_cfg())


def _output_resolution(
    args: PretrainRunConfig,
    *,
    output_dir: str = "out",
    resume_path: str | None = "resume.pt",
) -> output_policy_mod.OutputDirResolution:
    return output_policy_mod.OutputDirResolution(
        args=args,
        output_dir=output_dir,
        resume_path=resume_path,
    )


def _runtime_bootstrap(
    args: PretrainRunConfig,
    *,
    resume_path: str | None,
) -> PretrainRuntimeBootstrap:
    snapshot = PretrainPipelineSnapshot(
        output_dir=str(args.output_dir),
        resume_path=resume_path,
        manifest_path=str(args.data_path),
        manifest=None,
        tokenizer=None,
        tokenizer_path=str(args.tokenizer_path),
        seq_len=int(args.seq_len),
    )
    return PretrainRuntimeBootstrap.from_snapshot(
        snapshot=snapshot,
        cfg=args,
        resume_checkpoint=None,
    )


def test_resume_applies_recipe_settings_before_optimizer_load(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TinyModel(_RuntimeRecipeControlMixin, torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)
            self.config = argparse.Namespace(max_batch_size=4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    model = _TinyModel()
    ckpt = EngineCheckpoint(
        step=7,
        model=model.state_dict(),
        optimizer={"expected_groups": 3},
        scheduler=None,
        args={
            "layerwise_lr_decay": 0.9,
            "batch_size": 7,
            pretrain_mod._MACHINE_SIGNATURE_ARG_KEY: build_machine_adaptive_signature(
                schema="pretrain_machine_adaptive_v1",
                device=torch.device("cpu"),
            ),
        },
        rng=None,
        ema=None,
        train_state=None,
    )
    created_groups: list[int] = []

    def fake_load_checkpoint(_path: str):
        return ckpt

    def fake_create_optimizer(
        _model: torch.nn.Module,
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
        layerwise_lr_decay: float,
        muon_ns_steps: int,
        muon_target_rms: float | None,
    ) -> _FakeOptimizer:
        del lr, weight_decay, betas, eps, muon_ns_steps, muon_target_rms
        groups = 3 if float(layerwise_lr_decay) < 1.0 else 2
        created_groups.append(groups)
        return _FakeOptimizer(groups)

    monkeypatch.setattr(resume_loader_mod, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(resume_loader_mod, "create_optimizer", fake_create_optimizer)
    monkeypatch.setattr(resume_loader_mod, "move_optimizer_state_to_device", lambda *_args: None)
    monkeypatch.setattr(resume_loader_mod, "restore_rng_state", lambda *_args: None)
    monkeypatch.setattr(resume_loader_mod, "apply_gradient_checkpointing", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(resume_loader_mod, "sync_runtime_batch_capacity", lambda **kwargs: setattr(kwargs["model"].config, "max_batch_size", int(kwargs["args"].batch_size)))

    args = _pretrain_cfg(
        learning_rate=1e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        layerwise_lr_decay=1.0,
        muon_ns_steps=0,
        batch_size=0,
        data_path="",
        gradient_checkpointing=0,
        gradient_checkpointing_exclude_first=0,
        gradient_checkpointing_exclude_last=0,
        loss_chunk_size=0,
    )
    runtime_state = PretrainRuntimeState.from_config(args)

    resume = resume_loader_mod.maybe_resume_from_checkpoint(
        args=args,
        model=model,
        resume_path="dummy.pt",
        device=torch.device("cpu"),
        machine_signature_arg_key=pretrain_mod._MACHINE_SIGNATURE_ARG_KEY,
        runtime_state=runtime_state,
    )

    assert created_groups == [3]
    assert isinstance(resume.optimizer, _FakeOptimizer)
    assert resume.optimizer.loaded is True
    assert resume.resolved_args is not None
    assert float(resume.resolved_args.layerwise_lr_decay) == 0.9
    assert int(resume.resolved_args.batch_size) == 7
    assert int(runtime_state.batch_size) == 7
    assert int(model.config.max_batch_size) == 7


def test_resume_keeps_current_clip_semantics_over_checkpoint_args() -> None:
    current = _pretrain_cfg(
        grad_clip_mode="hybrid_auto",
        max_grad_norm=1.0,
        agc_clip=0.01,
        agc_eps=1e-3,
        agc_exclude_bias_and_norm=1,
        learning_rate=1e-4,
    )
    resumed = resume_restore_mod.apply_resume_recipe_settings(
        args=current,
        resume_args={
            "grad_clip_mode": "agc",
            "max_grad_norm": 2.0,
            "agc_clip": 0.2,
            "agc_eps": 1e-2,
            "agc_exclude_bias_and_norm": 0,
            "learning_rate": 3e-4,
        },
    )

    assert str(resumed.grad_clip_mode) == "hybrid_auto"
    assert float(resumed.max_grad_norm) == pytest.approx(1.0)
    assert float(resumed.agc_clip) == pytest.approx(0.01)
    assert float(resumed.agc_eps) == pytest.approx(1e-3)
    assert int(resumed.agc_exclude_bias_and_norm) == 1
    assert float(resumed.learning_rate) == pytest.approx(3e-4)


def test_resume_rejects_machine_signature_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TinyModel(_RuntimeRecipeControlMixin, torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)
            self.config = argparse.Namespace(max_batch_size=4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    model = _TinyModel()
    current_signature = build_machine_adaptive_signature(
        schema="pretrain_machine_adaptive_v1",
        device=torch.device("cpu"),
    )
    checkpoint_signature = machine_signature_payload(current_signature)
    checkpoint_signature["machine"] = "other-host"
    ckpt = EngineCheckpoint(
        step=7,
        model=model.state_dict(),
        optimizer={"expected_groups": 3},
        scheduler=None,
        args={
            "layerwise_lr_decay": 0.9,
            "batch_size": 7,
            "accumulation_steps": 3,
            "gradient_checkpointing": 1,
            "dataloader_num_workers": 8,
            pretrain_mod._MACHINE_SIGNATURE_ARG_KEY: checkpoint_signature,
        },
        rng=None,
        ema=None,
        train_state=None,
    )
    created_groups: list[int] = []

    def fake_load_checkpoint(_path: str):
        return ckpt

    def fake_create_optimizer(
        _model: torch.nn.Module,
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
        layerwise_lr_decay: float,
        muon_ns_steps: int,
        muon_target_rms: float | None,
    ) -> _FakeOptimizer:
        del lr, weight_decay, betas, eps, muon_ns_steps, muon_target_rms
        groups = 3 if float(layerwise_lr_decay) < 1.0 else 2
        created_groups.append(groups)
        return _FakeOptimizer(groups)

    monkeypatch.setattr(resume_loader_mod, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(resume_loader_mod, "create_optimizer", fake_create_optimizer)
    monkeypatch.setattr(resume_loader_mod, "move_optimizer_state_to_device", lambda *_args: None)
    monkeypatch.setattr(resume_loader_mod, "restore_rng_state", lambda *_args: None)
    monkeypatch.setattr(resume_loader_mod, "apply_gradient_checkpointing", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(resume_loader_mod, "sync_runtime_batch_capacity", lambda **_kwargs: None)

    args = _pretrain_cfg(
        learning_rate=1e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        layerwise_lr_decay=1.0,
        muon_ns_steps=0,
        batch_size=0,
        accumulation_steps=0,
        data_path="",
        dataloader_num_workers=-1,
        dataloader_prefetch_factor=-1,
        dataloader_persistent_workers=-1,
        shard_preload=-1,
        shard_preload_bytes=-1,
        gradient_checkpointing=0,
        gradient_checkpointing_exclude_first=0,
        gradient_checkpointing_exclude_last=0,
        loss_chunk_size=0,
    )

    with pytest.raises(SophiaUsageError, match="machine signature mismatch"):
        resume_loader_mod.maybe_resume_from_checkpoint(
            args=args,
            model=model,
            resume_path="dummy.pt",
            device=torch.device("cpu"),
            machine_signature_arg_key=pretrain_mod._MACHINE_SIGNATURE_ARG_KEY,
        )

    assert created_groups == []


def test_cuda_machine_signature_read_failure_is_not_silenced(monkeypatch) -> None:
    def _raise(_index: int):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(torch.cuda, "get_device_properties", _raise)

    with pytest.raises(RuntimeError, match="failed to read CUDA machine properties"):
        build_machine_adaptive_signature(
            schema="pretrain_machine_adaptive_v1",
            device=torch.device("cuda:0"),
        )


def test_resume_rebuilds_curriculum_from_restored_total_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TinyModel(_RuntimeRecipeControlMixin, torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)
            self.config = argparse.Namespace(max_batch_size=4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    class _Tokenizer:
        vocab_size = 49152
        bos_token_id = 1
        eos_token_id = 2
        unk_token_id = 3
        pad_token_id = 0

    manifest = type("Manifest", (), {"dtype": "int32", "total_tokens": 10_000_000_000})()
    ckpt = EngineCheckpoint(
        step=7,
        model=_TinyModel().state_dict(),
        optimizer={"expected_groups": 2},
        scheduler=None,
            args={
                "total_tokens": 1_000_000_000,
                "batch_size": 2,
                "accumulation_steps": 1,
                pretrain_mod._MACHINE_SIGNATURE_ARG_KEY: build_machine_adaptive_signature(
                    schema="pretrain_machine_adaptive_v1",
                    device=torch.device("cpu"),
                ),
            },
        rng=None,
        ema=None,
        train_state=None,
    )
    pipeline = _pipeline()
    pipeline.device = torch.device("cpu")
    pipeline.base_dtype = torch.float32
    pipeline.args = _pretrain_cfg(
        total_tokens=10_000_000_000,
        max_seq_len=4096,
        seq_len=4096,
        batch_size=2,
        accumulation_steps=1,
        tokenizer_path="",
        data_path="dataset/train",
        learning_rate=1e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        layerwise_lr_decay=1.0,
        muon_ns_steps=0,
        muon_target_rms=None,
        gradient_checkpointing=0,
        gradient_checkpointing_exclude_first=0,
        gradient_checkpointing_exclude_last=0,
        loss_chunk_size=0,
        seed=42,
    )

    monkeypatch.setattr(
        pretrain_mod,
        "prepare_output_dir_and_resume",
        lambda _args: _output_resolution(_args),
    )
    monkeypatch.setattr(
        bootstrap_io_mod,
        "prepare_pretrain_eval_data",
        lambda **kwargs: kwargs["args"],
    )
    monkeypatch.setattr(
        bootstrap_io_mod,
        "load_manifest_or_die",
        lambda **_kwargs: LoadedPretrainManifest(path="manifest.json", manifest=manifest),
    )
    monkeypatch.setattr(
        bootstrap_io_mod,
        "load_tokenizer_and_validate_manifest",
        lambda **_kwargs: LoadedPretrainTokenizer(tokenizer=_Tokenizer(), path="tok", seq_len=4096),
    )
    monkeypatch.setattr(
        runtime_bootstrap_mod,
        "sha1_file",
        lambda _path: "sha1_resume_total_tokens",
    )
    monkeypatch.setattr(pretrain_mod, "build_decoder_config", lambda **_kwargs: (argparse.Namespace(max_batch_size=4), 49152))
    monkeypatch.setattr(pretrain_mod, "load_or_init_model", lambda **_kwargs: _TinyModel())
    monkeypatch.setattr(pretrain_mod, "sync_runtime_batch_capacity", lambda **_kwargs: None)
    monkeypatch.setattr(pretrain_mod, "load_checkpoint", lambda _path: ckpt)
    monkeypatch.setattr(pretrain_mod, "create_optimizer", lambda *_args, **_kwargs: _FakeOptimizer(2))
    monkeypatch.setattr(resume_loader_mod, "move_optimizer_state_to_device", lambda *_args: None)
    monkeypatch.setattr(resume_loader_mod, "restore_rng_state", lambda *_args: None)
    monkeypatch.setattr(
        resume_loader_mod,
        "apply_gradient_checkpointing",
        lambda *_args, **_kwargs: None,
    )

    pipeline._prepare_data_and_model()

    assert int(pipeline.args.total_tokens) == 1_000_000_000
    assert pipeline.curriculum_stages is not None
    assert int(pipeline.curriculum_stages[-1].end_tokens) == 1_000_000_000


def test_resume_restores_stage_shaping_recipe_inputs_before_stage_plan_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TinyModel(_RuntimeRecipeControlMixin, torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)
            self.config = argparse.Namespace(max_batch_size=4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    class _Tokenizer:
        vocab_size = 49152
        bos_token_id = 1
        eos_token_id = 2
        unk_token_id = 3
        pad_token_id = 0

    manifest = type("Manifest", (), {"dtype": "int32", "total_tokens": 10_000_000_000})()
    ckpt = EngineCheckpoint(
        step=7,
        model=_TinyModel().state_dict(),
        optimizer={"expected_groups": 2},
        scheduler=None,
            args={
                "total_tokens": 1_000_000_000,
                "seq_len": 8192,
                "auto_batch_size_max": 96,
                "batch_size": 2,
                "accumulation_steps": 1,
                pretrain_mod._MACHINE_SIGNATURE_ARG_KEY: build_machine_adaptive_signature(
                    schema="pretrain_machine_adaptive_v1",
                    device=torch.device("cpu"),
                ),
            },
        rng=None,
        ema=None,
        train_state=None,
    )
    pipeline = _pipeline()
    pipeline.device = torch.device("cpu")
    pipeline.base_dtype = torch.float32
    pipeline.args = _pretrain_cfg(
        total_tokens=10_000_000_000,
        max_seq_len=8192,
        seq_len=4096,
        auto_batch_size_max=8,
        batch_size=2,
        accumulation_steps=1,
        tokenizer_path="",
        data_path="dataset/train",
        learning_rate=1e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        layerwise_lr_decay=1.0,
        muon_ns_steps=0,
        muon_target_rms=None,
        gradient_checkpointing=0,
        gradient_checkpointing_exclude_first=0,
        gradient_checkpointing_exclude_last=0,
        loss_chunk_size=0,
        seed=42,
    )

    monkeypatch.setattr(
        pretrain_mod,
        "prepare_output_dir_and_resume",
        lambda _args: _output_resolution(_args),
    )
    monkeypatch.setattr(
        bootstrap_io_mod,
        "prepare_pretrain_eval_data",
        lambda **kwargs: kwargs["args"],
    )
    monkeypatch.setattr(
        bootstrap_io_mod,
        "load_manifest_or_die",
        lambda **_kwargs: LoadedPretrainManifest(path="manifest.json", manifest=manifest),
    )
    monkeypatch.setattr(
        bootstrap_io_mod,
        "load_tokenizer_and_validate_manifest",
        lambda **_kwargs: LoadedPretrainTokenizer(
            tokenizer=_Tokenizer(),
            path="tok",
            seq_len=int(_kwargs["args"].seq_len),
        ),
    )
    monkeypatch.setattr(
        runtime_bootstrap_mod,
        "sha1_file",
        lambda _path: "sha1_resume_stage_shape",
    )
    monkeypatch.setattr(pretrain_mod, "build_decoder_config", lambda **_kwargs: (argparse.Namespace(max_batch_size=4), 49152))
    monkeypatch.setattr(pretrain_mod, "load_or_init_model", lambda **_kwargs: _TinyModel())
    monkeypatch.setattr(pretrain_mod, "sync_runtime_batch_capacity", lambda **_kwargs: None)
    monkeypatch.setattr(pretrain_mod, "load_checkpoint", lambda _path: ckpt)
    monkeypatch.setattr(pretrain_mod, "create_optimizer", lambda *_args, **_kwargs: _FakeOptimizer(2))
    monkeypatch.setattr(resume_loader_mod, "move_optimizer_state_to_device", lambda *_args: None)
    monkeypatch.setattr(resume_loader_mod, "restore_rng_state", lambda *_args: None)
    monkeypatch.setattr(
        resume_loader_mod,
        "apply_gradient_checkpointing",
        lambda *_args, **_kwargs: None,
    )

    pipeline._prepare_data_and_model()
    pipeline._finalize_budget_and_runner = lambda: None

    assert int(pipeline.args.seq_len) == 8192
    assert int(pipeline.args.auto_batch_size_max) == 96


def test_stage_recipe_realization_uses_current_fixed_config_with_fixed_machine_recipe() -> None:
    plan = StagePlan(
        stage=CurriculumStage(
            stage_index=1,
            seq_len=4096,
            start_tokens=0,
            end_tokens=4096,
        ),
        recipe=RuntimeRecipe(
            seq_len=4096,
            batch_size=4,
            accumulation_steps=2,
            tokens_per_update=32768,
            gradient_checkpointing=0,
            gradient_checkpointing_exclude_first=0,
            gradient_checkpointing_exclude_last=0,
            loss_chunk_size=0,
            machine_recipe_tuned=0,
        ),
        start_step=0,
        end_step=1,
    )

    resolved = stage_runtime_mod.realize_stage_plan_machine_recipe(
        args=_pretrain_cfg(
            gradient_checkpointing=1,
            gradient_checkpointing_exclude_first=2,
            gradient_checkpointing_exclude_last=3,
            loss_chunk_size=256,
        ),
        model=_RuntimeRecipeLinear(2, 2),
        device=torch.device("cpu"),
        output_dir="out",
        vocab_size=49152,
        base_dtype=torch.float32,
        plan=plan,
    )

    assert int(resolved.recipe.machine_recipe_tuned) == 1
    assert int(resolved.recipe.gradient_checkpointing) == 1
    assert int(resolved.recipe.gradient_checkpointing_exclude_first) == 2
    assert int(resolved.recipe.gradient_checkpointing_exclude_last) == 3
    assert int(resolved.recipe.loss_chunk_size) == 256


def test_prepare_data_and_model_preserves_explicit_input_settings_on_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TinyModel(_RuntimeRecipeControlMixin, torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)
            self.config = argparse.Namespace(max_batch_size=4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    class _Tokenizer:
        vocab_size = 49152
        bos_token_id = 1
        eos_token_id = 2
        unk_token_id = 3
        pad_token_id = 0

    manifest = type("Manifest", (), {"dtype": "int32", "total_tokens": 10_000_000})()
    ckpt = EngineCheckpoint(
        step=7,
        model=_TinyModel().state_dict(),
        optimizer={"expected_groups": 2},
        scheduler=None,
        args={
            "tokenizer_path": "/tmp/resume_tokenizer",
            pretrain_mod._MACHINE_SIGNATURE_ARG_KEY: build_machine_adaptive_signature(
                schema="pretrain_machine_adaptive_v1",
                device=torch.device("cpu"),
            ),
        },
        rng=None,
        ema=None,
        train_state=None,
    )
    pipeline = _pipeline()
    pipeline.device = torch.device("cpu")
    pipeline.base_dtype = torch.float32
    pipeline.args = _pretrain_cfg(
        total_tokens=10_000_000,
        max_seq_len=4096,
        seq_len=4096,
        auto_batch_size_max=8,
        batch_size=0,
        accumulation_steps=0,
        tokenizer_path="/tmp/configured_tokenizer",
        data_path="dataset/train",
        learning_rate=1e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        layerwise_lr_decay=1.0,
        muon_ns_steps=0,
        muon_target_rms=None,
        gradient_checkpointing=0,
        gradient_checkpointing_exclude_first=0,
        gradient_checkpointing_exclude_last=0,
        loss_chunk_size=0,
        seed=42,
    )

    seen: dict[str, object] = {"load_checkpoint_calls": 0}

    def fake_load_checkpoint(_path: str):
        seen["load_checkpoint_calls"] = int(seen["load_checkpoint_calls"]) + 1
        return ckpt

    def fake_load_tokenizer_and_validate_manifest(*, args, manifest):
        del manifest
        seen["tokenizer_path"] = str(args.tokenizer_path)
        return LoadedPretrainTokenizer(
            tokenizer=_Tokenizer(),
            path=str(args.tokenizer_path),
            seq_len=4096,
        )

    monkeypatch.setattr(
        pretrain_mod,
        "prepare_output_dir_and_resume",
        lambda _args: _output_resolution(_args),
    )
    monkeypatch.setattr(
        bootstrap_io_mod,
        "prepare_pretrain_eval_data",
        lambda **kwargs: kwargs["args"],
    )
    monkeypatch.setattr(
        bootstrap_io_mod,
        "load_manifest_or_die",
        lambda **_kwargs: LoadedPretrainManifest(path="manifest.json", manifest=manifest),
    )
    monkeypatch.setattr(
        bootstrap_io_mod,
        "load_tokenizer_and_validate_manifest",
        fake_load_tokenizer_and_validate_manifest,
    )
    monkeypatch.setattr(runtime_bootstrap_mod, "sha1_file", lambda _path: "abc123")
    monkeypatch.setattr(pretrain_mod, "build_decoder_config", lambda **_kwargs: (argparse.Namespace(max_batch_size=4), 49152))
    monkeypatch.setattr(pretrain_mod, "load_or_init_model", lambda **_kwargs: _TinyModel())
    monkeypatch.setattr(pretrain_mod, "sync_runtime_batch_capacity", lambda **_kwargs: None)
    monkeypatch.setattr(pretrain_mod, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(pretrain_mod, "create_optimizer", lambda *_args, **_kwargs: _FakeOptimizer(2))
    monkeypatch.setattr(resume_loader_mod, "move_optimizer_state_to_device", lambda *_args: None)
    monkeypatch.setattr(resume_loader_mod, "restore_rng_state", lambda *_args: None)
    monkeypatch.setattr(
        resume_loader_mod,
        "apply_gradient_checkpointing",
        lambda *_args, **_kwargs: None,
    )

    pipeline._prepare_data_and_model()

    assert int(seen["load_checkpoint_calls"]) == 1
    assert str(seen["tokenizer_path"]) == "/tmp/configured_tokenizer"
    assert str(pipeline.args._sophia_train_manifest_sha1) == "abc123"


def test_finalize_budget_and_runner_uses_persisted_baseline_microbatch_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline()
    pipeline.resume = resume_loader_mod.PretrainResumeState(
        ckpt=object(),
        optimizer=object(),
        start_step=7,
        resume_scheduler_state={},
        resume_args={},
        resume_train_state={},
    )
    pipeline.model = _RuntimeRecipeLinear(2, 2)
    pipeline.device = torch.device("cpu")
    pipeline.base_dtype = torch.float32
    pipeline.output_dir = "out"
    pipeline.manifest_path = "manifest.json"
    pipeline.tokenizer_path = "tok"
    pipeline.seq_len = 8192
    pipeline.curriculum_stages = [
        CurriculumStage(stage_index=0, seq_len=4096, start_tokens=0, end_tokens=4096),
        CurriculumStage(stage_index=1, seq_len=8192, start_tokens=4096, end_tokens=8192),
    ]
    pipeline.args = _pretrain_cfg(
        batch_size=2,
        accumulation_steps=1,
        target_tokens_per_update=8192,
        target_tokens_per_microbatch=8192,
        auto_batch_size_max=64,
        total_tokens=8192,
        max_seq_len=8192,
        seq_len=8192,
        device="cpu",
        learning_rate=1e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        layerwise_lr_decay=1.0,
        muon_ns_steps=0,
        muon_target_rms=None,
        max_grad_norm=1.0,
        warmup_steps=0,
        warmup_ratio=0.0,
        min_lr_ratio=0.0,
        lr_schedule="cosine",
        wsd_stable_ratio=0.0,
        wsd_decay_style="cosine",
        eval_steps=1,
        log_interval=1,
        eval_interval=1,
        save_interval=1,
        save_total_limit=1,
        save_best=0,
        async_checkpoint=0,
        async_metrics=0,
        ema_decay=0.0,
        ema_update_interval=1,
    )

    monkeypatch.setattr(
        observability_mod,
        "auto_observability_policy",
        lambda **kwargs: kwargs["args"],
    )
    monkeypatch.setattr(
        observability_mod,
        "auto_ema_policy",
        lambda **kwargs: kwargs["args"],
    )
    monkeypatch.setattr(pretrain_mod, "_build_runner", lambda **_kwargs: object())
    monkeypatch.setattr(observability_mod, "write_release_pretrain_profile", lambda **_kwargs: None)

    pipeline._finalize_budget_and_runner()

    assert pipeline.stage_execution_plans is not None
    first_recipe = pipeline.stage_execution_plans[0].recipe
    assert int(first_recipe.seq_len) == 4096
    assert int(first_recipe.batch_size) == 2
    assert int(first_recipe.accumulation_steps) == 1


def test_write_release_pretrain_profile_normalizes_manifest_fingerprints(
    tmp_path: Path,
) -> None:
    args = _pretrain_cfg(
        _sophia_run_kind="pretrain",
        data_path="dataset/train",
        eval_data_path="dataset/val",
        test_data_path="dataset/test",
        _sophia_train_manifest_sha1="train_sha1",
        device="cpu",
        total_tokens=1234,
        max_seq_len=4096,
        seq_len=4096,
    )
    plans = [
        StagePlan(
            stage=CurriculumStage(
                stage_index=0,
                seq_len=4096,
                start_tokens=0,
                end_tokens=1234,
            ),
            recipe=RuntimeRecipe(
                seq_len=4096,
                batch_size=4,
                accumulation_steps=1,
                tokens_per_update=4096,
            ),
            start_step=0,
            end_step=1,
        )
    ]

    observability_mod.write_release_pretrain_profile(
        args=args,
        output_dir=str(tmp_path),
        manifest_path="manifest.json",
        tokenizer_path="tokenizer",
        stage_execution_plans=plans,
    )

    with open(tmp_path / "pretrain_profile.json", encoding="utf-8") as handle:
        payload = json.load(handle)

    dataset = dict(payload["dataset"])
    assert str(dataset["train_manifest_sha1"]) == "train_sha1"


def test_prepare_data_and_model_rejects_train_manifest_fingerprint_mismatch_on_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TinyModel(_RuntimeRecipeControlMixin, torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)
            self.config = argparse.Namespace(max_batch_size=4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    class _Tokenizer:
        vocab_size = 49152
        bos_token_id = 1
        eos_token_id = 2
        unk_token_id = 3
        pad_token_id = 0

    manifest = type("Manifest", (), {"dtype": "int32", "total_tokens": 10_000_000})()
    ckpt = EngineCheckpoint(
        step=7,
        model=_TinyModel().state_dict(),
        optimizer={"expected_groups": 2},
        scheduler=None,
        args={
            "_sophia_train_manifest_sha1": "mismatched_train_sha1",
            pretrain_mod._MACHINE_SIGNATURE_ARG_KEY: build_machine_adaptive_signature(
                schema="pretrain_machine_adaptive_v1",
                device=torch.device("cpu"),
            ),
        },
        rng=None,
        ema=None,
        train_state=None,
    )
    pipeline = _pipeline()
    pipeline.device = torch.device("cpu")
    pipeline.base_dtype = torch.float32
    pipeline.args = _pretrain_cfg(
        total_tokens=10_000_000,
        max_seq_len=4096,
        seq_len=4096,
        auto_batch_size_max=8,
        batch_size=0,
        accumulation_steps=0,
        tokenizer_path="",
        data_path="dataset/train",
        learning_rate=1e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        layerwise_lr_decay=1.0,
        muon_ns_steps=0,
        muon_target_rms=None,
        gradient_checkpointing=0,
        gradient_checkpointing_exclude_first=0,
        gradient_checkpointing_exclude_last=0,
        loss_chunk_size=0,
        seed=42,
    )

    monkeypatch.setattr(
        pretrain_mod,
        "prepare_output_dir_and_resume",
        lambda _args: _output_resolution(_args),
    )
    monkeypatch.setattr(
        bootstrap_io_mod,
        "prepare_pretrain_eval_data",
        lambda **kwargs: kwargs["args"],
    )
    monkeypatch.setattr(
        bootstrap_io_mod,
        "load_manifest_or_die",
        lambda **_kwargs: LoadedPretrainManifest(path="manifest.json", manifest=manifest),
    )
    monkeypatch.setattr(runtime_bootstrap_mod, "sha1_file", lambda _path: "new_sha1")
    monkeypatch.setattr(
        bootstrap_io_mod,
        "load_tokenizer_and_validate_manifest",
        lambda **_kwargs: LoadedPretrainTokenizer(tokenizer=_Tokenizer(), path="tok", seq_len=4096),
    )
    monkeypatch.setattr(pretrain_mod, "build_decoder_config", lambda **_kwargs: (argparse.Namespace(max_batch_size=4), 49152))
    monkeypatch.setattr(pretrain_mod, "load_or_init_model", lambda **_kwargs: _TinyModel())
    monkeypatch.setattr(pretrain_mod, "sync_runtime_batch_capacity", lambda **_kwargs: None)
    monkeypatch.setattr(pretrain_mod, "load_checkpoint", lambda _path: ckpt)

    with pytest.raises(SophiaUsageError, match="train manifest fingerprint mismatch"):
        pipeline._prepare_data_and_model()


def test_prepare_data_and_model_rejects_val_manifest_fingerprint_mismatch_on_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TinyModel(_RuntimeRecipeControlMixin, torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)
            self.config = argparse.Namespace(max_batch_size=4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    class _Tokenizer:
        vocab_size = 49152
        bos_token_id = 1
        eos_token_id = 2
        unk_token_id = 3
        pad_token_id = 0

    manifest = type("Manifest", (), {"dtype": "int32", "total_tokens": 10_000_000})()
    ckpt = EngineCheckpoint(
        step=7,
        model=_TinyModel().state_dict(),
        optimizer={"expected_groups": 2},
        scheduler=None,
        args={
            "_sophia_train_manifest_sha1": "train_sha1",
            "_sophia_val_manifest_sha1": "mismatched_val_sha1",
            "_sophia_test_manifest_sha1": "test_sha1",
            pretrain_mod._MACHINE_SIGNATURE_ARG_KEY: build_machine_adaptive_signature(
                schema="pretrain_machine_adaptive_v1",
                device=torch.device("cpu"),
            ),
        },
        rng=None,
        ema=None,
        train_state=None,
    )
    pipeline = _pipeline()
    pipeline.device = torch.device("cpu")
    pipeline.base_dtype = torch.float32
    pipeline.args = _pretrain_cfg(
        total_tokens=10_000_000,
        max_seq_len=4096,
        seq_len=4096,
        auto_batch_size_max=8,
        batch_size=0,
        accumulation_steps=0,
        tokenizer_path="",
        data_path="dataset/train",
        eval_data_path="dataset/val",
        test_data_path="dataset/test",
        learning_rate=1e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        layerwise_lr_decay=1.0,
        muon_ns_steps=0,
        muon_target_rms=None,
        gradient_checkpointing=0,
        gradient_checkpointing_exclude_first=0,
        gradient_checkpointing_exclude_last=0,
        loss_chunk_size=0,
        seed=42,
    )

    monkeypatch.setattr(
        pretrain_mod,
        "prepare_output_dir_and_resume",
        lambda _args: _output_resolution(_args),
    )
    monkeypatch.setattr(
        bootstrap_io_mod,
        "prepare_pretrain_eval_data",
        lambda **kwargs: kwargs["args"],
    )
    monkeypatch.setattr(
        bootstrap_io_mod,
        "load_manifest_or_die",
        lambda **_kwargs: LoadedPretrainManifest(path="manifest.json", manifest=manifest),
    )
    monkeypatch.setattr(
        runtime_bootstrap_mod,
        "sha1_file",
        lambda path: (
            "new_val_sha1"
            if str(path).endswith("val/manifest.json")
            else ("test_sha1" if str(path).endswith("test/manifest.json") else "train_sha1")
        ),
    )
    monkeypatch.setattr(runtime_bootstrap_mod.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(
        bootstrap_io_mod,
        "load_tokenizer_and_validate_manifest",
        lambda **_kwargs: LoadedPretrainTokenizer(tokenizer=_Tokenizer(), path="tok", seq_len=4096),
    )
    monkeypatch.setattr(pretrain_mod, "build_decoder_config", lambda **_kwargs: (argparse.Namespace(max_batch_size=4), 49152))
    monkeypatch.setattr(pretrain_mod, "load_or_init_model", lambda **_kwargs: _TinyModel())
    monkeypatch.setattr(pretrain_mod, "sync_runtime_batch_capacity", lambda **_kwargs: None)
    monkeypatch.setattr(pretrain_mod, "load_checkpoint", lambda _path: ckpt)

    with pytest.raises(SophiaUsageError, match="val manifest fingerprint mismatch"):
        pipeline._prepare_data_and_model()


def test_resolve_resume_checkpoint_rejects_missing_explicit_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(SophiaUsageError, match="does not exist"):
        checkpoint_mod.resolve_resume_checkpoint(str(missing), output_dir=str(tmp_path))


def test_record_current_manifest_fingerprints_returns_replaced_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _pretrain_cfg(
        data_path="dataset/train",
        eval_data_path="dataset/val",
        test_data_path="dataset/test",
        seq_len=4096,
    )
    monkeypatch.setattr(
        runtime_bootstrap_mod,
        "sha1_file",
        lambda path: (
            "train_sha1"
            if str(path).endswith("train_manifest.json")
            else (
                "stage_sha1"
                if str(path).endswith("stage_manifest.json")
                else (
                    "val_sha1"
                    if str(path).endswith("val/manifest.json")
                    else "test_sha1"
                )
            )
        ),
    )
    monkeypatch.setattr(runtime_bootstrap_mod.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(
        runtime_bootstrap_mod.manifest_policy,
        "resolve_manifest_path",
        lambda data_path: (
            "/tmp/val/manifest.json"
            if str(data_path) == "dataset/val"
            else "/tmp/test/manifest.json"
        ),
    )

    updated = runtime_bootstrap_mod.record_current_manifest_fingerprints(
        args=args,
        manifest_path="train_manifest.json",
    )

    assert updated is not args
    assert str(args._sophia_train_manifest_sha1) == ""
    assert str(updated._sophia_train_manifest_sha1) == "train_sha1"
    assert str(updated._sophia_val_manifest_sha1) == "val_sha1"
    assert str(updated._sophia_test_manifest_sha1) == "test_sha1"


def test_manifest_fingerprint_failure_is_not_silenced(monkeypatch) -> None:
    monkeypatch.setattr(runtime_bootstrap_mod.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(
        runtime_bootstrap_mod,
        "sha1_file",
        lambda _path: (_ for _ in ()).throw(OSError("unreadable")),
    )

    with pytest.raises(SophiaUsageError, match="failed to fingerprint manifest"):
        runtime_bootstrap_mod.manifest_sha1_if_configured("manifest.json")


def test_resolve_resume_checkpoint_rejects_empty_explicit_directory(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_resume"
    empty_dir.mkdir()
    with pytest.raises(SophiaUsageError, match="contains no checkpoints"):
        checkpoint_mod.resolve_resume_checkpoint(str(empty_dir), output_dir=str(tmp_path))


class _ClosableIter:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True
def _build_pipeline_for_close_test(data_iter: object) -> pretrain_mod.PretrainPipeline:
    pipeline = _pipeline()
    pipeline.resume = resume_loader_mod.PretrainResumeState(
        ckpt=None,
        optimizer=object(),
        start_step=0,
        resume_scheduler_state={},
        resume_args=None,
        resume_train_state=None,
    )
    pipeline.model = _RuntimeRecipeLinear(2, 2)
    pipeline.tokenizer = object()
    pipeline.runner = object()
    pipeline.device = torch.device("cpu")
    pipeline.manifest = type("Manifest", (), {"dtype": "int32"})()
    pipeline.manifest_path = "manifest.json"
    pipeline.seq_len = 128
    pipeline.max_steps = 5
    pipeline.output_dir = "out"
    pipeline.resume_path = None
    pipeline.base_dtype = torch.bfloat16
    pipeline.stage_execution_plans = None
    pipeline.args = _pretrain_cfg(
        learning_rate=1e-3,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        layerwise_lr_decay=1.0,
        muon_ns_steps=5,
        warmup_steps=0,
        warmup_ratio=0.0,
        min_lr_ratio=0.0,
        lr_schedule="cosine",
        wsd_stable_ratio=0.0,
        wsd_decay_style="cosine",
        batch_size=2,
        seed=42,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=-1,
        shard_preload=0,
        shard_preload_bytes=0,
        eval_steps=1,
        accumulation_steps=1,
        max_grad_norm=1.0,
        grad_clip_mode="norm",
        agc_clip=0.0,
        agc_eps=1e-3,
        agc_exclude_bias_and_norm=1,
        log_interval=1,
        save_interval=100,
        save_total_limit=1,
        save_weights_steps=0,
        save_weights_total_limit=0,
        async_checkpoint=0,
        async_metrics=0,
        ckpt_staging_dir="",
        save_best=0,
        ema_eval=0,
        ema_use_for_export=0,
        ema_in_ckpt=0,
        ema_decay=0.0,
        ema_update_interval=1,
        safe_serialization=1,
    )
    return pipeline


def test_run_train_loop_closes_data_iter_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    data_iter = _ClosableIter()
    pipeline = _build_pipeline_for_close_test(data_iter)

    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_data_iter", lambda **_kwargs: data_iter)
    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_eval_fn", lambda **_kwargs: (lambda *_a, **_k: 0.0, 1))
    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_test_eval_fn", lambda **_kwargs: (lambda *_a, **_k: 0.0))
    monkeypatch.setattr(observability_mod, "maybe_print_train_eta", lambda **_kwargs: None)
    monkeypatch.setattr(pretrain_mod, "build_lr_scheduler", lambda **_kwargs: object())
    monkeypatch.setattr(
        loop_runtime_mod,
        "train_loop",
        lambda **_kwargs: loop_runtime_mod.TrainLoopResult(last_step=5, train_state={}),
    )
    monkeypatch.setattr(pretrain_mod, "_build_runner", lambda **_kwargs: object())

    pipeline._run_train_loop()

    assert data_iter.closed is True


def test_run_train_loop_closes_data_iter_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    data_iter = _ClosableIter()
    pipeline = _build_pipeline_for_close_test(data_iter)

    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_data_iter", lambda **_kwargs: data_iter)
    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_eval_fn", lambda **_kwargs: (lambda *_a, **_k: 0.0, 1))
    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_test_eval_fn", lambda **_kwargs: (lambda *_a, **_k: 0.0))
    monkeypatch.setattr(observability_mod, "maybe_print_train_eta", lambda **_kwargs: None)
    monkeypatch.setattr(pretrain_mod, "build_lr_scheduler", lambda **_kwargs: object())
    monkeypatch.setattr(pretrain_mod, "_build_runner", lambda **_kwargs: object())

    def _boom(**_kwargs) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(loop_runtime_mod, "train_loop", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        pipeline._run_train_loop()

    assert data_iter.closed is True


def test_run_train_loop_rejects_resume_without_data_iter_state(monkeypatch: pytest.MonkeyPatch) -> None:
    data_iter = _ClosableIter()
    pipeline = _build_pipeline_for_close_test(data_iter)
    pipeline.resume = resume_loader_mod.PretrainResumeState(
        ckpt=None,
        optimizer=object(),
        start_step=3,
        resume_scheduler_state={},
        resume_args=None,
        resume_train_state={},
    )

    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_eval_fn", lambda **_kwargs: (lambda *_a, **_k: 0.0, 1))
    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_test_eval_fn", lambda **_kwargs: (lambda *_a, **_k: 0.0))
    monkeypatch.setattr(observability_mod, "maybe_print_train_eta", lambda **_kwargs: None)
    monkeypatch.setattr(pretrain_mod, "build_lr_scheduler", lambda **_kwargs: object())
    monkeypatch.setattr(pretrain_mod, "_build_runner", lambda **_kwargs: object())

    with pytest.raises(RuntimeError, match="data iterator state"):
        pipeline._run_train_loop()


def test_stage_resume_state_requires_transition_step_for_incompatible_tail() -> None:
    manifest = type(
        "Manifest",
        (),
        {"shards": (type("Shard", (), {"tokens": 128})(),)},
    )()
    resume_train_state = {
        "data_iter_state": {
            "order": [0],
            "order_pos": 0,
            "current_pos": 96,
            "worker_id": 0,
            "num_workers": 1,
        }
    }
    next_recipe = RuntimeRecipe(
        seq_len=64,
        batch_size=1,
        accumulation_steps=1,
        tokens_per_update=64,
    )

    assert (
        transition.stage_resume_state_requires_transition_step(
            manifest=manifest,
            resume_train_state=resume_train_state,
            next_recipe=next_recipe,
        )
        is True
    )


def test_stage_resume_state_requires_transition_step_for_small_unread_shard() -> None:
    manifest = type(
        "Manifest",
        (),
        {
            "shards": (
                type("Shard", (), {"tokens": 48})(),
                type("Shard", (), {"tokens": 256})(),
            )
        },
    )()
    resume_train_state = {
        "data_iter_state": {
            "order": [0, 1],
            "order_pos": 0,
            "current_pos": None,
            "worker_id": 0,
            "num_workers": 1,
        }
    }
    next_recipe = RuntimeRecipe(
        seq_len=64,
        batch_size=1,
        accumulation_steps=1,
        tokens_per_update=64,
    )

    assert (
        transition.stage_resume_state_requires_transition_step(
            manifest=manifest,
            resume_train_state=resume_train_state,
            next_recipe=next_recipe,
        )
        is True
    )


def test_stage_resume_state_requires_transition_step_for_future_small_unread_shard() -> None:
    manifest = type(
        "Manifest",
        (),
        {
            "shards": (
                type("Shard", (), {"tokens": 256})(),
                type("Shard", (), {"tokens": 48})(),
                type("Shard", (), {"tokens": 256})(),
            )
        },
    )()
    resume_train_state = {
        "data_iter_state": {
            "order": [0, 1, 2],
            "order_pos": 0,
            "current_pos": 16,
            "worker_id": 0,
            "num_workers": 1,
        }
    }
    next_recipe = RuntimeRecipe(
        seq_len=64,
        batch_size=1,
        accumulation_steps=1,
        tokens_per_update=64,
    )

    assert (
        transition.stage_resume_state_requires_transition_step(
            manifest=manifest,
            resume_train_state=resume_train_state,
            next_recipe=next_recipe,
        )
        is True
    )


def test_prepare_train_state_for_stage_transition_preserves_active_state() -> None:
    original = {
        "seen_supervised_tokens": 123,
        "lr_decay_start_step": 9,
    }

    prepared = PretrainTrainState.resolve(original).cleared_for_stage_transition().to_payload()

    assert int(prepared["seen_supervised_tokens"]) == 123
    assert int(prepared["lr_decay_start_step"]) == 9


def test_pretrain_train_state_rejects_invalid_numeric_values() -> None:
    with pytest.raises(TypeError, match="expected integer train state value"):
        PretrainTrainState.from_payload({"seen_supervised_tokens": "invalid"})


def test_prepare_train_state_for_stage_transition_can_clear_data_iter_state() -> None:
    original = {
        "data_iter_state": {
            "order": [0, 1],
            "order_pos": 0,
            "current_pos": 32,
        },
        "seen_supervised_tokens": 123,
    }

    prepared = PretrainTrainState.resolve(original).cleared_for_stage_transition(
        clear_data_iter_state=True
    ).to_payload()

    assert "data_iter_state" in original
    assert "data_iter_state" not in prepared
    assert int(prepared["seen_supervised_tokens"]) == 123


def test_run_train_loop_drains_incompatible_stage_tail_before_next_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _build_pipeline_for_close_test(_ClosableIter())
    pipeline.max_steps = 3
    pipeline.stage_execution_plans = [
        StagePlan(
            stage=CurriculumStage(
                stage_index=0,
                seq_len=32,
                start_tokens=0,
                end_tokens=32,
            ),
            recipe=RuntimeRecipe(
                seq_len=32,
                batch_size=1,
                accumulation_steps=1,
                tokens_per_update=32,
            ),
            start_step=0,
            end_step=1,
        ),
        StagePlan(
            stage=CurriculumStage(
                stage_index=1,
                seq_len=64,
                start_tokens=32,
                end_tokens=96,
            ),
            recipe=RuntimeRecipe(
                seq_len=64,
                batch_size=1,
                accumulation_steps=1,
                tokens_per_update=64,
            ),
            start_step=1,
            end_step=3,
        ),
    ]
    pipeline.manifest = type(
        "Manifest",
        (),
        {
            "dtype": "int32",
            "shards": (
                type("Shard", (), {"tokens": 128})(),
                type("Shard", (), {"tokens": 256})(),
            ),
        },
    )()

    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_data_iter", lambda **_kwargs: _ClosableIter())
    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_eval_fn", lambda **_kwargs: (lambda *_a, **_k: 0.0, 1))
    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_test_eval_fn", lambda **_kwargs: (lambda *_a, **_k: 0.0))
    monkeypatch.setattr(observability_mod, "maybe_print_train_eta", lambda **_kwargs: None)
    monkeypatch.setattr(pretrain_mod, "build_lr_scheduler", lambda **_kwargs: object())
    monkeypatch.setattr(pretrain_mod, "_build_runner", lambda **_kwargs: object())

    calls: list[dict[str, object]] = []

    def fake_train_loop(**kwargs):
        calls.append(
            {
                "start_step": int(kwargs["start_step"]),
                "max_steps": int(kwargs["cfg"].max_steps),
                "resume_train_state": kwargs["resume_train_state"],
                "token_weighted_loss": int(kwargs["cfg"].token_weighted_loss),
            }
        )
        assert int(kwargs["cfg"].token_weighted_loss) == 1
        if len(calls) == 1:
            return loop_runtime_mod.TrainLoopResult(
                last_step=1,
                train_state={
                    "lr_decay_start_step": 7,
                    "data_iter_state": {
                        "order": [0, 1],
                        "order_pos": 0,
                        "current_pos": 96,
                        "worker_id": 0,
                        "num_workers": 1,
                    },
                },
            )
        if len(calls) == 2:
            return loop_runtime_mod.TrainLoopResult(
                last_step=2,
                train_state={
                    "lr_decay_start_step": 7,
                    "data_iter_state": {
                        "order": [0, 1],
                        "order_pos": 1,
                        "current_pos": None,
                        "worker_id": 0,
                        "num_workers": 1,
                    },
                },
            )
        if len(calls) == 3:
            resume_state = dict(kwargs["resume_train_state"])
            assert resume_state["lr_decay_start_step"] == 7
            return loop_runtime_mod.TrainLoopResult(last_step=3, train_state={})
        raise AssertionError("unexpected extra train loop invocation")

    monkeypatch.setattr(loop_runtime_mod, "train_loop", fake_train_loop)

    pipeline._run_train_loop()

    assert [int(call["start_step"]) for call in calls] == [0, 1, 2]
    assert [int(call["max_steps"]) for call in calls] == [1, 2, 3]


def test_run_train_loop_resume_stage_boundary_replays_transition_step_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _build_pipeline_for_close_test(_ClosableIter())
    pipeline.max_steps = 3
    pipeline.resume = resume_loader_mod.PretrainResumeState(
        ckpt=None,
        optimizer=object(),
        start_step=1,
        resume_scheduler_state={},
        resume_args=None,
        resume_train_state={
            "data_iter_state": {
                "order": [0, 1],
                "order_pos": 0,
                "current_pos": 96,
                "worker_id": 0,
                "num_workers": 1,
            },
            "lr_decay_start_step": 7,
        },
    )
    pipeline.stage_execution_plans = [
        StagePlan(
            stage=CurriculumStage(
                stage_index=0,
                seq_len=32,
                start_tokens=0,
                end_tokens=32,
            ),
            recipe=RuntimeRecipe(
                seq_len=32,
                batch_size=1,
                accumulation_steps=1,
                tokens_per_update=32,
            ),
            start_step=0,
            end_step=1,
        ),
        StagePlan(
            stage=CurriculumStage(
                stage_index=1,
                seq_len=64,
                start_tokens=32,
                end_tokens=96,
            ),
            recipe=RuntimeRecipe(
                seq_len=64,
                batch_size=1,
                accumulation_steps=1,
                tokens_per_update=64,
            ),
            start_step=1,
            end_step=3,
        ),
    ]
    pipeline.manifest = type(
        "Manifest",
        (),
        {
            "dtype": "int32",
            "shards": (
                type("Shard", (), {"tokens": 128})(),
                type("Shard", (), {"tokens": 256})(),
            ),
        },
    )()

    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_data_iter", lambda **_kwargs: _ClosableIter())
    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_eval_fn", lambda **_kwargs: (lambda *_a, **_k: 0.0, 1))
    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_test_eval_fn", lambda **_kwargs: (lambda *_a, **_k: 0.0))
    monkeypatch.setattr(observability_mod, "maybe_print_train_eta", lambda **_kwargs: None)
    monkeypatch.setattr(pretrain_mod, "build_lr_scheduler", lambda **_kwargs: object())
    monkeypatch.setattr(pretrain_mod, "_build_runner", lambda **_kwargs: object())

    calls: list[dict[str, object]] = []

    def fake_train_loop(**kwargs):
        calls.append(
            {
                "start_step": int(kwargs["start_step"]),
                "max_steps": int(kwargs["cfg"].max_steps),
                "resume_train_state": kwargs["resume_train_state"],
            }
        )
        if len(calls) == 1:
            return loop_runtime_mod.TrainLoopResult(
                last_step=2,
                train_state={
                    "lr_decay_start_step": 11,
                    "data_iter_state": {
                        "order": [0, 1],
                        "order_pos": 1,
                        "current_pos": None,
                        "worker_id": 0,
                        "num_workers": 1,
                    },
                },
            )
        if len(calls) == 2:
            resume_state = dict(kwargs["resume_train_state"])
            assert resume_state["lr_decay_start_step"] == 11
            return loop_runtime_mod.TrainLoopResult(last_step=3, train_state={})
        raise AssertionError("unexpected extra train loop invocation")

    monkeypatch.setattr(loop_runtime_mod, "train_loop", fake_train_loop)

    pipeline._run_train_loop()

    assert [int(call["start_step"]) for call in calls] == [1, 2]
    assert [int(call["max_steps"]) for call in calls] == [2, 3]


def test_run_train_loop_terminal_transition_step_uses_final_stage_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _build_pipeline_for_close_test(_ClosableIter())
    pipeline.max_steps = 2
    pipeline.resume = resume_loader_mod.PretrainResumeState(
        ckpt=None,
        optimizer=object(),
        start_step=1,
        resume_scheduler_state={},
        resume_args=None,
        resume_train_state={
            "data_iter_state": {
                "order": [0, 1],
                "order_pos": 0,
                "current_pos": 96,
                "worker_id": 0,
                "num_workers": 1,
            }
        },
    )
    pipeline.stage_execution_plans = [
        StagePlan(
            stage=CurriculumStage(
                stage_index=0,
                seq_len=32,
                start_tokens=0,
                end_tokens=32,
            ),
            recipe=RuntimeRecipe(
                seq_len=32,
                batch_size=1,
                accumulation_steps=1,
                tokens_per_update=32,
            ),
            start_step=0,
            end_step=1,
        ),
        StagePlan(
            stage=CurriculumStage(
                stage_index=1,
                seq_len=64,
                start_tokens=32,
                end_tokens=64,
            ),
            recipe=RuntimeRecipe(
                seq_len=64,
                batch_size=1,
                accumulation_steps=1,
                tokens_per_update=64,
            ),
            start_step=1,
            end_step=2,
        ),
    ]
    pipeline.manifest = type(
        "Manifest",
        (),
        {
            "dtype": "int32",
            "shards": (
                type("Shard", (), {"tokens": 128})(),
                type("Shard", (), {"tokens": 256})(),
            ),
        },
    )()

    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_data_iter", lambda **_kwargs: _ClosableIter())
    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_eval_fn", lambda **_kwargs: (lambda *_a, **_k: 0.0, 1))
    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_test_eval_fn", lambda **_kwargs: (lambda *_a, **_k: 0.0))
    monkeypatch.setattr(observability_mod, "maybe_print_train_eta", lambda **_kwargs: None)
    monkeypatch.setattr(pretrain_mod, "build_lr_scheduler", lambda **_kwargs: object())
    monkeypatch.setattr(pretrain_mod, "_build_runner", lambda **_kwargs: object())

    calls: list[dict[str, object]] = []

    def fake_train_loop(**kwargs):
        calls.append(
            {
                "start_step": int(kwargs["start_step"]),
                "max_steps": int(kwargs["cfg"].max_steps),
                "tokenizer": kwargs["tokenizer"],
                "extra_eval_names": [str(name) for name, *_rest in kwargs["extra_evals"]],
            }
        )
        return loop_runtime_mod.TrainLoopResult(last_step=2, train_state={})

    monkeypatch.setattr(loop_runtime_mod, "train_loop", fake_train_loop)

    pipeline._run_train_loop()

    assert len(calls) == 1
    assert int(calls[0]["start_step"]) == 1
    assert int(calls[0]["max_steps"]) == 2
    assert calls[0]["tokenizer"] is pipeline.tokenizer
    assert "test_loss" in calls[0]["extra_eval_names"]


def test_run_train_loop_resume_mid_bridge_continues_bridge_before_next_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _build_pipeline_for_close_test(_ClosableIter())
    pipeline.max_steps = 4
    pipeline.resume = resume_loader_mod.PretrainResumeState(
        ckpt=None,
        optimizer=object(),
        start_step=2,
        resume_scheduler_state={},
        resume_args=None,
        resume_train_state={
            "data_iter_state": {
                "order": [0, 1],
                "order_pos": 0,
                "current_pos": 96,
                "worker_id": 0,
                "num_workers": 1,
            },
            "lr_decay_start_step": 17,
        },
    )
    pipeline.stage_execution_plans = [
        StagePlan(
            stage=CurriculumStage(
                stage_index=0,
                seq_len=32,
                start_tokens=0,
                end_tokens=32,
            ),
            recipe=RuntimeRecipe(
                seq_len=32,
                batch_size=1,
                accumulation_steps=1,
                tokens_per_update=32,
            ),
            start_step=0,
            end_step=1,
        ),
        StagePlan(
            stage=CurriculumStage(
                stage_index=1,
                seq_len=64,
                start_tokens=32,
                end_tokens=128,
            ),
            recipe=RuntimeRecipe(
                seq_len=64,
                batch_size=1,
                accumulation_steps=1,
                tokens_per_update=64,
            ),
            start_step=1,
            end_step=4,
        ),
    ]
    pipeline.manifest = type(
        "Manifest",
        (),
        {
            "dtype": "int32",
            "shards": (
                type("Shard", (), {"tokens": 128})(),
                type("Shard", (), {"tokens": 256})(),
            ),
        },
    )()

    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_data_iter", lambda **_kwargs: _ClosableIter())
    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_eval_fn", lambda **_kwargs: (lambda *_a, **_k: 0.0, 1))
    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_test_eval_fn", lambda **_kwargs: (lambda *_a, **_k: 0.0))
    monkeypatch.setattr(observability_mod, "maybe_print_train_eta", lambda **_kwargs: None)
    monkeypatch.setattr(pretrain_mod, "build_lr_scheduler", lambda **_kwargs: object())
    monkeypatch.setattr(pretrain_mod, "_build_runner", lambda **_kwargs: object())

    calls: list[dict[str, object]] = []

    def fake_train_loop(**kwargs):
        calls.append(
            {
                "start_step": int(kwargs["start_step"]),
                "max_steps": int(kwargs["cfg"].max_steps),
                "resume_train_state": kwargs["resume_train_state"],
            }
        )
        if len(calls) == 1:
            return loop_runtime_mod.TrainLoopResult(
                last_step=3,
                train_state={
                    "lr_decay_start_step": 19,
                    "data_iter_state": {
                        "order": [0, 1],
                        "order_pos": 1,
                        "current_pos": None,
                        "worker_id": 0,
                        "num_workers": 1,
                    },
                },
            )
        if len(calls) == 2:
            resume_state = dict(kwargs["resume_train_state"])
            assert resume_state["lr_decay_start_step"] == 19
            return loop_runtime_mod.TrainLoopResult(last_step=4, train_state={})
        raise AssertionError("unexpected extra train loop invocation")

    monkeypatch.setattr(loop_runtime_mod, "train_loop", fake_train_loop)

    pipeline._run_train_loop()

    assert [int(call["start_step"]) for call in calls] == [2, 3]
    assert [int(call["max_steps"]) for call in calls] == [3, 4]


def test_build_lr_schedule_state_config_restores_saved_decay_boundary() -> None:
    cfg = stage_runtime_mod.build_lr_schedule_state_config(
        max_steps=1000,
        warmup_steps=100,
        wsd_stable_ratio=0.5,
        lr_schedule="cosine",
        resume_train_state={
            "lr_decay_start_step": 777,
        },
    )
    assert int(cfg.lr_decay_start_step) == 777
def test_checkpoint_roundtrip_sanitizes_pretrain_profile_args(tmp_path: Path) -> None:
    args = pretrain_mod.build_pretrain_args(
        data_path="dataset/train",
        tokenizer_path="artifacts/tokenizer",
        output_dir=str(tmp_path),
    )
    model = _RuntimeRecipeLinear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = checkpoint_mod.build_checkpoint_state(
        step=3,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        args=args.to_payload(),
        rng=None,
        ema=None,
        train_state={"seen_supervised_tokens": 128},
    )
    checkpoint_mod.save_checkpoint_state(
        output_dir=str(tmp_path),
        step=3,
        state=state,
        save_total_limit=2,
    )

    ckpt = checkpoint_mod.load_checkpoint(str(tmp_path / "checkpoints" / "ckpt_step3.pt"))
    assert ckpt.step == 3
    assert isinstance(ckpt.args, dict)
    profile_payload = ckpt.args.get("_sophia_pretrain_profile")
    assert isinstance(profile_payload, dict)
    assert str(profile_payload["kind"]) == "release"
    assert str(profile_payload["name"]) == "release"


def test_load_checkpoint_rejects_invalid_profile_object_args(tmp_path: Path) -> None:
    args = pretrain_mod.build_pretrain_args(
        data_path="dataset/train",
        tokenizer_path="artifacts/tokenizer",
        output_dir=str(tmp_path),
    )
    invalid_state = {
        "step": 5,
        "model": {},
        "optimizer": {},
        "scheduler": None,
        "args": dict(vars(args)),
        "rng": None,
        "ema": None,
        "train_state": {"seen_supervised_tokens": 256},
    }
    ckpt_path = tmp_path / "invalid_profile_args.pt"
    torch.save(invalid_state, ckpt_path)

    with pytest.raises(
        RuntimeError,
        match="weights-only Sophia checkpoints are supported",
    ):
        checkpoint_mod.load_checkpoint(str(ckpt_path))


def test_save_checkpoint_state_sanitizes_profile_args_for_weights_only_load(tmp_path: Path) -> None:
    args = pretrain_mod.build_pretrain_args(
        data_path="dataset/train",
        tokenizer_path="artifacts/tokenizer",
        output_dir=str(tmp_path),
    )
    checkpoint_mod.save_checkpoint_state(
        output_dir=str(tmp_path),
        step=9,
        state={
            "kind": "weights_only",
            "step": 9,
            "model": {},
            "args": args.to_payload(),
            "ema": None,
        },
        save_total_limit=2,
        prefix="weights_step",
    )

    payload = torch.load(
        str(tmp_path / "checkpoints" / "weights_step9.pt"),
        map_location="cpu",
        weights_only=True,
    )
    assert isinstance(payload, dict)
    profile_payload = payload["args"]["_sophia_pretrain_profile"]
    assert isinstance(profile_payload, dict)
    assert str(profile_payload["kind"]) == "release"


def test_save_checkpoint_state_cleans_tmp_file_on_interrupted_write(tmp_path: Path, monkeypatch) -> None:
    target_dir = tmp_path / "checkpoints"
    seen: dict[str, str] = {}

    def _interrupting_save(_state, path: str) -> None:
        Path(path).write_bytes(b"partial")
        seen["path"] = str(path)
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        checkpoint_mod.save_checkpoint_state(
            output_dir=str(tmp_path),
            step=2,
            state={"step": 2, "model": {}, "optimizer": {}, "args": {}},
            save_total_limit=2,
            torch_save_fn=_interrupting_save,
        )

    assert target_dir.exists()
    assert "path" in seen
    assert not Path(seen["path"]).exists()
    assert list(target_dir.iterdir()) == []


def test_save_checkpoint_state_keeps_only_latest_checkpoints(tmp_path: Path) -> None:
    for step in (1, 2, 3, 4, 5, 6):
        checkpoint_mod.save_checkpoint_state(
            output_dir=str(tmp_path),
            step=step,
            state={"step": step, "model": {}, "optimizer": {}, "args": {}},
            save_total_limit=2,
        )

    kept = sorted(path.name for path in (tmp_path / "checkpoints").glob("ckpt_step*.pt"))
    assert kept == ["ckpt_step5.pt", "ckpt_step6.pt"]


def test_pretrain_session_absorb_run_config_preserves_machine_runtime() -> None:
    args = _pretrain_cfg(seq_len=4096, max_steps=64)
    runtime_state = PretrainRuntimeState.from_config(args)
    runtime_state.step_execution_backend = "inductor"
    runtime_state.dataloader_num_workers = 0
    runtime_state.dataloader_prefetch_factor = 0
    runtime_state.dataloader_persistent_workers = 0
    runtime_state.shard_preload = 1
    runtime_state.shard_preload_bytes = 4096
    session = PretrainSession(
        spec=build_pretrain_run_spec(args),
        run=RunContext(
            output_dir=str(args.output_dir),
            resume_path=None,
            device=torch.device("cuda:0"),
            base_dtype=torch.bfloat16,
            tokenizer=None,
            runtime_metadata={},
            machine_signature=args._sophia_machine_signature or {},
            resolved_device="cuda:0",
        ),
        bootstrap=_runtime_bootstrap(args, resume_path=None),
        runtime_state=runtime_state,
    )

    session.absorb_run_config(replace(args, batch_size=8))

    assert str(session.runtime_state.step_execution_backend) == "inductor"
    assert int(session.runtime_state.dataloader_num_workers) == 0
    assert int(session.runtime_state.dataloader_prefetch_factor) == 0
    assert int(session.runtime_state.dataloader_persistent_workers) == 0
    assert int(session.runtime_state.shard_preload) == 1
    assert int(session.runtime_state.shard_preload_bytes) == 4096


def test_resume_model_session_preserves_machine_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _pretrain_cfg(seq_len=4096, max_steps=64)
    runtime_state = PretrainRuntimeState.from_config(args)
    runtime_state.step_execution_backend = "inductor"
    runtime_state.dataloader_num_workers = 0
    runtime_state.dataloader_prefetch_factor = 0
    runtime_state.dataloader_persistent_workers = 0
    runtime_state.shard_preload = 1
    runtime_state.shard_preload_bytes = 4096
    session = PretrainSession(
        spec=build_pretrain_run_spec(args),
        run=RunContext(
            output_dir=str(args.output_dir),
            resume_path="resume.pt",
            device=torch.device("cuda:0"),
            base_dtype=torch.bfloat16,
            tokenizer=None,
            runtime_metadata={},
            machine_signature=args._sophia_machine_signature or {},
            resolved_device="cuda:0",
        ),
        bootstrap=_runtime_bootstrap(args, resume_path="resume.pt"),
        model=_RuntimeRecipeLinear(2, 2),
        runtime_state=runtime_state,
    )

    resolved_args = replace(args, batch_size=8)

    hooks = type(
        "_Hooks",
        (),
        {
            "maybe_resume_from_checkpoint": staticmethod(
                lambda **_kwargs: type(
                    "_ResumeState",
                    (),
                    {"resolved_args": resolved_args},
                )()
            )
        },
    )()

    resumed_args = runtime_prepare_mod.resume_model_session(
        session=session,
        args=args,
        hooks=hooks,
    )

    assert int(resumed_args.batch_size) == 8
    assert str(session.runtime_state.step_execution_backend) == "inductor"
    assert int(session.runtime_state.dataloader_num_workers) == 0
    assert int(session.runtime_state.dataloader_prefetch_factor) == 0
    assert int(session.runtime_state.dataloader_persistent_workers) == 0
    assert int(session.runtime_state.shard_preload) == 1
    assert int(session.runtime_state.shard_preload_bytes) == 4096


def test_apply_resume_runtime_settings_restores_machine_runtime() -> None:
    args = _pretrain_cfg(seq_len=4096, max_steps=64)
    runtime_state = PretrainRuntimeState.from_config(args)

    resolved = resume_restore_mod.apply_resume_runtime_settings(
        args=args,
        resume_args={
            "_sophia_machine_runtime": {
                "step_execution_backend": "inductor",
                "dataloader_num_workers": 0,
                "dataloader_prefetch_factor": 0,
                "dataloader_persistent_workers": 0,
                "shard_preload": 1,
                "shard_preload_bytes": 4096,
            }
        },
        runtime_state=runtime_state,
    )

    assert isinstance(resolved, PretrainRunConfig)
    assert str(runtime_state.step_execution_backend) == "inductor"
    assert int(runtime_state.dataloader_num_workers) == 0
    assert int(runtime_state.dataloader_prefetch_factor) == 0
    assert int(runtime_state.dataloader_persistent_workers) == 0
    assert int(runtime_state.shard_preload) == 1
    assert int(runtime_state.shard_preload_bytes) == 4096


def test_apply_resume_runtime_settings_keeps_native_runtime_when_resume_args_empty() -> None:
    args = _pretrain_cfg(seq_len=4096, max_steps=64)
    runtime_state = PretrainRuntimeState.from_config(args)

    resolved = resume_restore_mod.apply_resume_runtime_settings(
        args=args,
        resume_args={},
        runtime_state=runtime_state,
    )

    assert isinstance(resolved, PretrainRunConfig)
    runtime_payload = CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD
    assert int(runtime_state.dataloader_num_workers) == int(
        runtime_payload["dataloader_num_workers"]
    )
    assert int(runtime_state.dataloader_prefetch_factor) == int(
        runtime_payload["dataloader_prefetch_factor"]
    )
    assert int(runtime_state.dataloader_persistent_workers) == int(
        runtime_payload["dataloader_persistent_workers"]
    )
    assert int(runtime_state.shard_preload) == int(runtime_payload["shard_preload"])
    assert int(runtime_state.shard_preload_bytes) == int(
        runtime_payload["shard_preload_bytes"]
    )
