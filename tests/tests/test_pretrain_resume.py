from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ml.errors import SophiaUsageError
from ml.core.engine import checkpointing as checkpoint_mod
from ml.core.engine.checkpointing import EngineCheckpoint
from ml.core.engine.context import RunContext
from ml.core.engine.machine_signature import (
    build_machine_adaptive_signature,
    machine_signature_payload,
)
from ml.tasks.pretrain import pipeline as pretrain_mod
from ml.tasks.pretrain.run_spec_builder import build_pretrain_run_spec
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.profiles import VALIDATION_PROFILE
from ml.training.pretrain.resources import (
    LoadedPretrainManifest,
    LoadedPretrainTokenizer,
)
from ml.tasks.pretrain.session import (
    PretrainRuntimeState,
    PretrainSession,
)
from ml.training.pretrain.pipeline_snapshot import PretrainPipelineSnapshot
from ml.training.pretrain import bootstrap_io as bootstrap_io_mod
from ml.training.pretrain import loop_runtime as loop_runtime_mod
from ml.training.pretrain import output_policy as output_policy_mod
from ml.training.pretrain import runtime_bootstrap as runtime_bootstrap_mod
from ml.training.pretrain.runtime_bootstrap import PretrainRuntimeBootstrap
from ml.training.pretrain import resume_loader as resume_loader_mod
from ml.training.pretrain import resume_restore as resume_restore_mod
from ml.training.pretrain.release_config import (
    DEFAULT_PRETRAIN_MACHINE_RUNTIME,
)
from ml.tasks.pretrain import runtime as runtime_prepare_mod
from ml.training.pretrain import observability as observability_mod
from ml.training.pretrain.train_state import PretrainTrainState


def _full_checkpoint_state(
    step: int,
    *,
    args: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema": checkpoint_mod.ENGINE_CHECKPOINT_SCHEMA,
        "kind": checkpoint_mod.FULL_CHECKPOINT_KIND,
        "step": int(step),
        "model": {},
        "optimizer": {},
        "scheduler": None,
        "args": {} if args is None else args,
        "rng": None,
        "ema": None,
        "train_state": None,
    }


class _FakeOptimizer:
    def __init__(self, groups: int) -> None:
        self.groups = int(groups)
        self.loaded = False

    def load_state_dict(self, state_dict: dict) -> None:
        if int(state_dict.get("expected_groups", 0) or 0) != int(self.groups):
            raise RuntimeError(
                "loaded state dict has a different number of parameter groups"
            )
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
    return replace(args, _sophia_pretrain_profile=VALIDATION_PROFILE, **overrides)


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


def test_resume_applies_recipe_settings_before_optimizer_load(
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
    ckpt = EngineCheckpoint(
        step=7,
        model=model.state_dict(),
        optimizer={"expected_groups": 2},
        scheduler=None,
        args={
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
        muon_ns_steps: int,
    ) -> _FakeOptimizer:
        del lr, weight_decay, betas, eps, muon_ns_steps
        created_groups.append(2)
        return _FakeOptimizer(2)

    monkeypatch.setattr(resume_loader_mod, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(resume_loader_mod, "create_optimizer", fake_create_optimizer)
    monkeypatch.setattr(
        resume_loader_mod, "move_optimizer_state_to_device", lambda *_args: None
    )
    monkeypatch.setattr(resume_loader_mod, "restore_rng_state", lambda *_args: None)
    monkeypatch.setattr(
        resume_loader_mod,
        "apply_gradient_checkpointing",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        resume_loader_mod,
        "sync_runtime_batch_capacity",
        lambda **kwargs: setattr(
            kwargs["model"].config, "max_batch_size", int(kwargs["args"].batch_size)
        ),
    )

    args = _pretrain_cfg(
        learning_rate=1e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
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

    assert created_groups == [2]
    assert isinstance(resume.optimizer, _FakeOptimizer)
    assert resume.optimizer.loaded is True
    assert resume.resolved_args is not None
    assert int(resume.resolved_args.batch_size) == 7
    assert int(runtime_state.batch_size) == 7
    assert int(model.config.max_batch_size) == 7


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
        muon_ns_steps: int,
    ) -> _FakeOptimizer:
        del lr, weight_decay, betas, eps, muon_ns_steps
        created_groups.append(2)
        return _FakeOptimizer(2)

    monkeypatch.setattr(resume_loader_mod, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(resume_loader_mod, "create_optimizer", fake_create_optimizer)
    monkeypatch.setattr(
        resume_loader_mod, "move_optimizer_state_to_device", lambda *_args: None
    )
    monkeypatch.setattr(resume_loader_mod, "restore_rng_state", lambda *_args: None)
    monkeypatch.setattr(
        resume_loader_mod,
        "apply_gradient_checkpointing",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        resume_loader_mod, "sync_runtime_batch_capacity", lambda **_kwargs: None
    )

    args = _pretrain_cfg(
        learning_rate=1e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
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


def test_resume_restores_checkpoint_token_budget(
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

    manifest = type(
        "Manifest", (), {"dtype": "int32", "total_tokens": 10_000_000_000}
    )()
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
        muon_ns_steps=0,
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
        lambda **_kwargs: LoadedPretrainManifest(
            path="manifest.json", manifest=manifest
        ),
    )
    monkeypatch.setattr(
        bootstrap_io_mod,
        "load_tokenizer_and_validate_manifest",
        lambda **_kwargs: LoadedPretrainTokenizer(
            tokenizer=_Tokenizer(), path="tok", seq_len=4096
        ),
    )
    monkeypatch.setattr(
        runtime_bootstrap_mod,
        "sha1_file",
        lambda _path: "sha1_resume_total_tokens",
    )
    monkeypatch.setattr(
        pretrain_mod,
        "build_decoder_config",
        lambda **_kwargs: (argparse.Namespace(max_batch_size=4), 49152),
    )
    monkeypatch.setattr(
        pretrain_mod, "initialize_model", lambda **_kwargs: _TinyModel()
    )
    monkeypatch.setattr(
        pretrain_mod, "sync_runtime_batch_capacity", lambda **_kwargs: None
    )
    monkeypatch.setattr(pretrain_mod, "load_checkpoint", lambda _path: ckpt)
    monkeypatch.setattr(
        pretrain_mod, "create_optimizer", lambda *_args, **_kwargs: _FakeOptimizer(2)
    )
    monkeypatch.setattr(
        resume_loader_mod, "move_optimizer_state_to_device", lambda *_args: None
    )
    monkeypatch.setattr(resume_loader_mod, "restore_rng_state", lambda *_args: None)
    monkeypatch.setattr(
        resume_loader_mod,
        "apply_gradient_checkpointing",
        lambda *_args, **_kwargs: None,
    )

    pipeline._prepare_data_and_model()

    assert int(pipeline.args.total_tokens) == 1_000_000_000




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
        batch_size=0,
        accumulation_steps=0,
        tokenizer_path="/tmp/configured_tokenizer",
        data_path="dataset/train",
        learning_rate=1e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        muon_ns_steps=0,
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
        lambda **_kwargs: LoadedPretrainManifest(
            path="manifest.json", manifest=manifest
        ),
    )
    monkeypatch.setattr(
        bootstrap_io_mod,
        "load_tokenizer_and_validate_manifest",
        fake_load_tokenizer_and_validate_manifest,
    )
    monkeypatch.setattr(runtime_bootstrap_mod, "sha1_file", lambda _path: "abc123")
    monkeypatch.setattr(
        pretrain_mod,
        "build_decoder_config",
        lambda **_kwargs: (argparse.Namespace(max_batch_size=4), 49152),
    )
    monkeypatch.setattr(
        pretrain_mod, "initialize_model", lambda **_kwargs: _TinyModel()
    )
    monkeypatch.setattr(
        pretrain_mod, "sync_runtime_batch_capacity", lambda **_kwargs: None
    )
    monkeypatch.setattr(pretrain_mod, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        pretrain_mod, "create_optimizer", lambda *_args, **_kwargs: _FakeOptimizer(2)
    )
    monkeypatch.setattr(
        resume_loader_mod, "move_optimizer_state_to_device", lambda *_args: None
    )
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


def test_finalize_budget_and_runner_enforces_fixed_token_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline()
    pipeline.resume = resume_loader_mod.PretrainResumeState(
        ckpt=None,
        optimizer=object(),
        start_step=0,
        resume_scheduler_state=None,
        resume_args=None,
        resume_train_state=None,
    )
    pipeline.model = _RuntimeRecipeLinear(2, 2)
    pipeline.device = torch.device("cpu")
    pipeline.base_dtype = torch.float32
    pipeline.output_dir = "out"
    pipeline.manifest_path = "manifest.json"
    pipeline.tokenizer = argparse.Namespace(model_max_length=0)
    pipeline.tokenizer_path = "tok"
    pipeline.seq_len = 4096
    pipeline.args = _pretrain_cfg(
        batch_size=2,
        accumulation_steps=1,
        target_tokens_per_update=8192,
        total_tokens=16_384,
        max_steps=-1,
        max_seq_len=4096,
        seq_len=4096,
        device="cpu",
    )
    runner = object()

    monkeypatch.setattr(pretrain_mod, "_build_runner", lambda **_kwargs: runner)
    monkeypatch.setattr(
        observability_mod, "write_release_pretrain_profile", lambda **_kwargs: None
    )

    pipeline._finalize_budget_and_runner()

    assert int(pipeline.max_steps) == 2
    assert int(pipeline.target_tokens_per_update) == 8192
    assert pipeline.runner is runner

def test_write_release_pretrain_profile_records_fixed_runtime(
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
        max_steps=1,
        max_seq_len=4096,
        seq_len=4096,
        batch_size=1,
        accumulation_steps=320,
    )

    observability_mod.write_release_pretrain_profile(
        args=args,
        output_dir=str(tmp_path),
        manifest_path="manifest.json",
        tokenizer_path="tokenizer",
    )

    with open(tmp_path / "pretrain_profile.json", encoding="utf-8") as handle:
        payload = json.load(handle)

    dataset = dict(payload["dataset"])
    runtime = dict(payload["runtime"])
    assert str(dataset["train_manifest_sha1"]) == "train_sha1"
    assert int(runtime["seq_len"]) == 4096
    assert int(runtime["batch_size"]) == 1
    assert int(runtime["accumulation_steps"]) == 320
    assert int(runtime["tokens_per_update"]) == 1_310_720

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
        batch_size=0,
        accumulation_steps=0,
        tokenizer_path="",
        data_path="dataset/train",
        learning_rate=1e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        muon_ns_steps=0,
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
        lambda **_kwargs: LoadedPretrainManifest(
            path="manifest.json", manifest=manifest
        ),
    )
    monkeypatch.setattr(runtime_bootstrap_mod, "sha1_file", lambda _path: "new_sha1")
    monkeypatch.setattr(
        bootstrap_io_mod,
        "load_tokenizer_and_validate_manifest",
        lambda **_kwargs: LoadedPretrainTokenizer(
            tokenizer=_Tokenizer(), path="tok", seq_len=4096
        ),
    )
    monkeypatch.setattr(
        pretrain_mod,
        "build_decoder_config",
        lambda **_kwargs: (argparse.Namespace(max_batch_size=4), 49152),
    )
    monkeypatch.setattr(
        pretrain_mod, "initialize_model", lambda **_kwargs: _TinyModel()
    )
    monkeypatch.setattr(
        pretrain_mod, "sync_runtime_batch_capacity", lambda **_kwargs: None
    )
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
        muon_ns_steps=0,
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
        lambda **_kwargs: LoadedPretrainManifest(
            path="manifest.json", manifest=manifest
        ),
    )
    monkeypatch.setattr(
        runtime_bootstrap_mod,
        "sha1_file",
        lambda path: (
            "new_val_sha1"
            if str(path).endswith("val/manifest.json")
            else (
                "test_sha1"
                if str(path).endswith("test/manifest.json")
                else "train_sha1"
            )
        ),
    )
    monkeypatch.setattr(runtime_bootstrap_mod.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(
        bootstrap_io_mod,
        "load_tokenizer_and_validate_manifest",
        lambda **_kwargs: LoadedPretrainTokenizer(
            tokenizer=_Tokenizer(), path="tok", seq_len=4096
        ),
    )
    monkeypatch.setattr(
        pretrain_mod,
        "build_decoder_config",
        lambda **_kwargs: (argparse.Namespace(max_batch_size=4), 49152),
    )
    monkeypatch.setattr(
        pretrain_mod, "initialize_model", lambda **_kwargs: _TinyModel()
    )
    monkeypatch.setattr(
        pretrain_mod, "sync_runtime_batch_capacity", lambda **_kwargs: None
    )
    monkeypatch.setattr(pretrain_mod, "load_checkpoint", lambda _path: ckpt)

    with pytest.raises(SophiaUsageError, match="val manifest fingerprint mismatch"):
        pipeline._prepare_data_and_model()


def test_resolve_resume_checkpoint_rejects_missing_explicit_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(SophiaUsageError, match="does not exist"):
        checkpoint_mod.resolve_resume_checkpoint(str(missing), output_dir=str(tmp_path))


def test_resolve_auto_resume_rejects_empty_output_dir(tmp_path: Path) -> None:
    with pytest.raises(SophiaUsageError, match="no checkpoint exists"):
        checkpoint_mod.resolve_resume_checkpoint("auto", output_dir=str(tmp_path))


def test_resolve_auto_resume_rejects_latest_checkpoint_without_sidecar(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "ckpt_step7.pt").write_bytes(b"checkpoint")

    with pytest.raises(SophiaUsageError, match="missing its SHA-256 sidecar"):
        checkpoint_mod.resolve_resume_checkpoint("latest", output_dir=str(tmp_path))


def test_resolve_explicit_checkpoint_rejects_missing_sidecar(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(SophiaUsageError, match="missing its SHA-256 sidecar"):
        checkpoint_mod.resolve_resume_checkpoint(str(checkpoint), output_dir=str(tmp_path))


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
                "train_sha1"
                if str(path).endswith("train_manifest.json")
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


def test_resolve_resume_checkpoint_rejects_empty_explicit_directory(
    tmp_path: Path,
) -> None:
    empty_dir = tmp_path / "empty_resume"
    empty_dir.mkdir()
    with pytest.raises(SophiaUsageError, match="contains no checkpoints"):
        checkpoint_mod.resolve_resume_checkpoint(
            str(empty_dir), output_dir=str(tmp_path)
        )


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
    pipeline.args = _pretrain_cfg(
        learning_rate=1e-3,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        muon_ns_steps=5,
        warmup_steps=0,
        warmup_ratio=0.0,
        min_lr_ratio=0.0,
        lr_schedule="cosine",
        batch_size=2,
        seed=42,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=-1,
        shard_preload=0,
        shard_preload_bytes=0,
        eval_steps=1,
        accumulation_steps=1,
        log_interval=1,
        save_interval=100,
        save_total_limit=1,
        async_checkpoint=0,
        async_metrics=0,
        ckpt_staging_dir="",
        save_best=0,
        safe_serialization=1,
    )
    return pipeline


def test_run_train_loop_closes_data_iter_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_iter = _ClosableIter()
    pipeline = _build_pipeline_for_close_test(data_iter)

    monkeypatch.setattr(
        loop_runtime_mod, "build_pretrain_data_iter", lambda **_kwargs: data_iter
    )
    monkeypatch.setattr(
        loop_runtime_mod,
        "build_pretrain_eval_fn",
        lambda **_kwargs: (lambda *_a, **_k: 0.0, 1),
    )
    monkeypatch.setattr(
        loop_runtime_mod,
        "build_pretrain_test_eval_fn",
        lambda **_kwargs: lambda *_a, **_k: 0.0,
    )
    monkeypatch.setattr(
        observability_mod, "maybe_print_train_eta", lambda **_kwargs: None
    )
    monkeypatch.setattr(pretrain_mod, "build_lr_scheduler", lambda **_kwargs: object())
    monkeypatch.setattr(
        loop_runtime_mod,
        "train_loop",
        lambda **_kwargs: loop_runtime_mod.TrainLoopResult(last_step=5, train_state={}),
    )
    monkeypatch.setattr(pretrain_mod, "_build_runner", lambda **_kwargs: object())

    pipeline._run_train_loop()

    assert data_iter.closed is True


def test_run_train_loop_closes_data_iter_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_iter = _ClosableIter()
    pipeline = _build_pipeline_for_close_test(data_iter)

    monkeypatch.setattr(
        loop_runtime_mod, "build_pretrain_data_iter", lambda **_kwargs: data_iter
    )
    monkeypatch.setattr(
        loop_runtime_mod,
        "build_pretrain_eval_fn",
        lambda **_kwargs: (lambda *_a, **_k: 0.0, 1),
    )
    monkeypatch.setattr(
        loop_runtime_mod,
        "build_pretrain_test_eval_fn",
        lambda **_kwargs: lambda *_a, **_k: 0.0,
    )
    monkeypatch.setattr(
        observability_mod, "maybe_print_train_eta", lambda **_kwargs: None
    )
    monkeypatch.setattr(pretrain_mod, "build_lr_scheduler", lambda **_kwargs: object())
    monkeypatch.setattr(pretrain_mod, "_build_runner", lambda **_kwargs: object())

    def _boom(**_kwargs) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(loop_runtime_mod, "train_loop", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        pipeline._run_train_loop()

    assert data_iter.closed is True


def test_run_train_loop_rejects_resume_without_data_iter_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(
        loop_runtime_mod,
        "build_pretrain_eval_fn",
        lambda **_kwargs: (lambda *_a, **_k: 0.0, 1),
    )
    monkeypatch.setattr(
        loop_runtime_mod,
        "build_pretrain_test_eval_fn",
        lambda **_kwargs: lambda *_a, **_k: 0.0,
    )
    monkeypatch.setattr(
        observability_mod, "maybe_print_train_eta", lambda **_kwargs: None
    )
    monkeypatch.setattr(pretrain_mod, "build_lr_scheduler", lambda **_kwargs: object())
    monkeypatch.setattr(pretrain_mod, "_build_runner", lambda **_kwargs: object())

    with pytest.raises(RuntimeError, match="data iterator state"):
        pipeline._run_train_loop()






def test_pretrain_train_state_rejects_invalid_numeric_values() -> None:
    with pytest.raises(TypeError, match="expected integer train state value"):
        PretrainTrainState.from_payload({"seen_supervised_tokens": "invalid"})







def test_run_train_loop_resumes_once_from_exact_data_iter_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_iter = _ClosableIter()
    pipeline = _build_pipeline_for_close_test(data_iter)
    pipeline.resume = resume_loader_mod.PretrainResumeState(
        ckpt=None,
        optimizer=object(),
        start_step=3,
        resume_scheduler_state={},
        resume_args=None,
        resume_train_state={
            "data_iter_state": {
                "order": [0],
                "order_pos": 0,
                "current_pos": 384,
                "worker_id": 0,
                "num_workers": 1,
            }
        },
    )
    seen: dict[str, object] = {}

    def build_data_iter(**kwargs):
        seen["iterator_resume_state"] = kwargs["resume_state"]
        return data_iter

    def train_loop(**kwargs):
        seen["start_step"] = kwargs["start_step"]
        seen["max_steps"] = kwargs["cfg"].max_steps
        seen["resume_train_state"] = kwargs["resume_train_state"]
        return loop_runtime_mod.TrainLoopResult(last_step=5, train_state={})

    monkeypatch.setattr(loop_runtime_mod, "build_pretrain_data_iter", build_data_iter)
    monkeypatch.setattr(
        loop_runtime_mod,
        "build_pretrain_eval_fn",
        lambda **_kwargs: (None, 1),
    )
    monkeypatch.setattr(
        loop_runtime_mod,
        "build_pretrain_test_eval_fn",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        observability_mod, "maybe_print_train_eta", lambda **_kwargs: None
    )
    monkeypatch.setattr(pretrain_mod, "build_lr_scheduler", lambda **_kwargs: object())
    monkeypatch.setattr(loop_runtime_mod, "train_loop", train_loop)

    pipeline._run_train_loop()

    assert int(seen["start_step"]) == 3
    assert int(seen["max_steps"]) == 5
    assert seen["iterator_resume_state"] == seen["resume_train_state"]["data_iter_state"]
    assert data_iter.closed is True

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

    ckpt = checkpoint_mod.load_checkpoint(
        str(tmp_path / "checkpoints" / "ckpt_step3.pt")
    )
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
        "schema": checkpoint_mod.ENGINE_CHECKPOINT_SCHEMA,
        "kind": checkpoint_mod.FULL_CHECKPOINT_KIND,
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
    digest = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    Path(f"{ckpt_path}.sha256").write_text(
        f"{digest}  {ckpt_path.name}\n",
        encoding="ascii",
    )

    with pytest.raises(
        RuntimeError,
        match="weights-only Sophia checkpoints are supported",
    ):
        checkpoint_mod.load_checkpoint(str(ckpt_path))


def test_load_checkpoint_requires_sha256_sidecar(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"step": 1, "model": {}, "optimizer": {}, "args": {}}, checkpoint)

    with pytest.raises(RuntimeError, match="sidecar is required"):
        checkpoint_mod.load_checkpoint(str(checkpoint))


def test_save_checkpoint_state_sanitizes_profile_args_for_weights_only_load(
    tmp_path: Path,
) -> None:
    args = pretrain_mod.build_pretrain_args(
        data_path="dataset/train",
        tokenizer_path="artifacts/tokenizer",
        output_dir=str(tmp_path),
    )
    checkpoint_mod.save_checkpoint_state(
        output_dir=str(tmp_path),
        step=9,
        state={
            "schema": checkpoint_mod.ENGINE_CHECKPOINT_SCHEMA,
            "kind": checkpoint_mod.WEIGHTS_ONLY_CHECKPOINT_KIND,
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

    with pytest.raises(RuntimeError, match="checkpoint kind mismatch"):
        checkpoint_mod.load_checkpoint(
            str(tmp_path / "checkpoints" / "weights_step9.pt"),
            expected_kind=checkpoint_mod.FULL_CHECKPOINT_KIND,
        )


def test_save_checkpoint_state_requires_explicit_schema(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsupported checkpoint schema"):
        checkpoint_mod.save_checkpoint_state(
            output_dir=str(tmp_path),
            step=1,
            state={
                "kind": "full",
                "step": 1,
                "model": {},
                "optimizer": {},
                "scheduler": None,
                "args": {},
                "rng": None,
                "ema": None,
                "train_state": None,
            },
            save_total_limit=1,
        )


def test_save_checkpoint_state_cleans_tmp_file_on_interrupted_write(
    tmp_path: Path, monkeypatch
) -> None:
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
            state=_full_checkpoint_state(2),
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
            state=_full_checkpoint_state(step),
            save_total_limit=2,
        )

    kept = sorted(
        path.name for path in (tmp_path / "checkpoints").glob("ckpt_step*.pt")
    )
    assert kept == ["ckpt_step5.pt", "ckpt_step6.pt"]
    assert sorted(
        path.name for path in (tmp_path / "checkpoints").glob("ckpt_step*.pt.sha256")
    ) == ["ckpt_step5.pt.sha256", "ckpt_step6.pt.sha256"]


def test_load_checkpoint_rejects_sha256_mismatch(tmp_path: Path) -> None:
    checkpoint_mod.save_checkpoint_state(
        output_dir=str(tmp_path),
        step=3,
        state=_full_checkpoint_state(3),
        save_total_limit=2,
    )
    checkpoint = tmp_path / "checkpoints" / "ckpt_step3.pt"
    with checkpoint.open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        checkpoint_mod.load_checkpoint(str(checkpoint))


def test_pretrain_session_absorb_run_config_preserves_machine_runtime() -> None:
    args = _pretrain_cfg(seq_len=4096, max_steps=64)
    runtime_state = PretrainRuntimeState.from_config(args)
    runtime_state.step_execution_backend = "sm120_graph"
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

    assert str(session.runtime_state.step_execution_backend) == "sm120_graph"
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
    runtime_state.step_execution_backend = "sm120_graph"
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
    assert str(session.runtime_state.step_execution_backend) == "sm120_graph"
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
                "step_execution_backend": "sm120_graph",
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
    assert str(runtime_state.step_execution_backend) == "sm120_graph"
    assert int(runtime_state.dataloader_num_workers) == 0
    assert int(runtime_state.dataloader_prefetch_factor) == 0
    assert int(runtime_state.dataloader_persistent_workers) == 0
    assert int(runtime_state.shard_preload) == 1
    assert int(runtime_state.shard_preload_bytes) == 4096


def _release_lineage_resume_args(tmp_path: Path) -> tuple[PretrainRunConfig, dict[str, object], Path]:
    dataset_root = tmp_path / "admitted_dataset"
    train_dir = dataset_root / "train"
    train_dir.mkdir(parents=True)
    manifest_path = train_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    marker_path = dataset_root / runtime_bootstrap_mod.DATASET_MARKER
    admission_sha256 = "a" * 64
    marker_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "lineage": {"admission_policy_sha256": admission_sha256},
            }
        ),
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_json.write_bytes(b"test-tokenizer")
    machine_recipe_path = tmp_path / "machine_recipe.json"
    machine_recipe_path.write_bytes(b"test-machine-recipe")
    protocol_path, protocol_sha256, protocol_payload = runtime_bootstrap_mod._load_release_protocol()
    lineage = {
        "_sophia_protocol_path": protocol_path,
        "_sophia_protocol_sha256": protocol_sha256,
        "_sophia_model_spec_sha256": str(protocol_payload["model"]["spec_sha256"]),
        "_sophia_model_parameter_count": int(protocol_payload["model"]["parameter_count"]),
        "_sophia_tokenizer_json_sha1": hashlib.sha1(tokenizer_json.read_bytes()).hexdigest(),
        "_sophia_dataset_marker_sha256": hashlib.sha256(marker_path.read_bytes()).hexdigest(),
        "_sophia_data_admission_sha256": admission_sha256,
        "_sophia_machine_recipe_path": str(machine_recipe_path.resolve()),
        "_sophia_machine_recipe_sha256": hashlib.sha256(machine_recipe_path.read_bytes()).hexdigest(),
    }
    args = pretrain_mod.build_pretrain_args(
        data_path=str(dataset_root),
        tokenizer_path=str(tokenizer_dir),
        output_dir=str(tmp_path / "run"),
        profile=pretrain_mod.RELEASE_PROFILE,
    )
    return replace(args, machine_recipe_json=str(machine_recipe_path)), lineage, manifest_path


def _prepare_release_bootstrap_for_test(
    args: PretrainRunConfig,
    *,
    manifest_path: Path,
    resume_args: dict[str, object] | None,
):
    manifest = type("Manifest", (), {"dtype": "int32", "total_tokens": 1234})()
    checkpoint = EngineCheckpoint(
        step=3,
        model={},
        optimizer={},
        scheduler=None,
        args=resume_args,
        rng=None,
        ema=None,
        train_state=None,
    )
    resume_path = None if resume_args is None else "resume.pt"
    return runtime_bootstrap_mod.prepare_pretrain_runtime_bootstrap(
        args=args,
        prepare_output_dir_and_resume=lambda value: _output_resolution(
            value, output_dir=str(args.output_dir), resume_path=resume_path
        ),
        load_checkpoint=lambda _path: checkpoint,
        prepare_pretrain_eval_data=lambda **kwargs: replace(
            kwargs["args"],
            data_path=str(manifest_path.parent),
            eval_data_path="",
            test_data_path="",
        ),
        load_manifest_or_die=lambda **_kwargs: LoadedPretrainManifest(
            path=str(manifest_path), manifest=manifest
        ),
        load_tokenizer_and_validate_manifest=lambda **_kwargs: LoadedPretrainTokenizer(
            tokenizer=object(), path=str(args.tokenizer_path), seq_len=4096
        ),
    )


def test_release_bootstrap_persists_lineage_and_matching_resume(
    tmp_path: Path,
) -> None:
    args, lineage, manifest_path = _release_lineage_resume_args(tmp_path)
    bootstrap = _prepare_release_bootstrap_for_test(
        args, manifest_path=manifest_path, resume_args=None
    )
    projected = bootstrap.project_run_config(args)
    for field_name, value in lineage.items():
        assert value not in (None, "")
        assert getattr(bootstrap, field_name.removeprefix("_sophia_")) == value
        assert getattr(projected, field_name) == value

    resumed = _prepare_release_bootstrap_for_test(
        args, manifest_path=manifest_path, resume_args=lineage
    )
    assert resumed.protocol_sha256 == lineage["_sophia_protocol_sha256"]


@pytest.mark.parametrize("recipe_value", ("", "missing-machine-recipe.json"))
def test_release_fresh_bootstrap_missing_machine_recipe_fails_closed(
    tmp_path: Path, recipe_value: str
) -> None:
    args, _, manifest_path = _release_lineage_resume_args(tmp_path)
    missing_recipe = replace(args, machine_recipe_json=recipe_value)
    with pytest.raises(SophiaUsageError):
        _prepare_release_bootstrap_for_test(
            missing_recipe, manifest_path=manifest_path, resume_args=None
        )


def test_machine_recipe_validation_bootstrap_does_not_require_recipe(
    tmp_path: Path,
) -> None:
    args, _, manifest_path = _release_lineage_resume_args(tmp_path)
    machine_validation_args = replace(
        args,
        machine_recipe_json="",
        _sophia_run_kind="pretrain_machine_validate",
    )
    bootstrap = _prepare_release_bootstrap_for_test(
        machine_validation_args, manifest_path=manifest_path, resume_args=None
    )
    assert bootstrap.machine_recipe_path == ""
    assert bootstrap.machine_recipe_sha256 == ""


@pytest.mark.parametrize(
    "field_name",
    (
        "_sophia_protocol_path",
        "_sophia_protocol_sha256",
        "_sophia_model_spec_sha256",
        "_sophia_model_parameter_count",
        "_sophia_tokenizer_json_sha1",
        "_sophia_dataset_marker_sha256",
        "_sophia_data_admission_sha256",
        "_sophia_machine_recipe_path",
        "_sophia_machine_recipe_sha256",
    ),
)
def test_release_resume_missing_lineage_field_fails_closed(
    tmp_path: Path, field_name: str
) -> None:
    args, lineage, manifest_path = _release_lineage_resume_args(tmp_path)
    missing = dict(lineage)
    missing.pop(field_name)
    with pytest.raises(SophiaUsageError):
        _prepare_release_bootstrap_for_test(
            args, manifest_path=manifest_path, resume_args=missing
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "_sophia_protocol_path",
        "_sophia_protocol_sha256",
        "_sophia_model_spec_sha256",
        "_sophia_model_parameter_count",
        "_sophia_tokenizer_json_sha1",
        "_sophia_dataset_marker_sha256",
        "_sophia_data_admission_sha256",
        "_sophia_machine_recipe_path",
        "_sophia_machine_recipe_sha256",
    ),
)
def test_release_resume_lineage_drift_fails_closed(
    tmp_path: Path, field_name: str
) -> None:
    args, lineage, manifest_path = _release_lineage_resume_args(tmp_path)
    drifted = dict(lineage)
    drifted[field_name] = (
        0
        if field_name == "_sophia_model_parameter_count"
        else (
            str(tmp_path / "other_recipe.json")
            if field_name == "_sophia_machine_recipe_path"
            else "f" * 64
        )
    )
    with pytest.raises(SophiaUsageError):
        _prepare_release_bootstrap_for_test(
            args, manifest_path=manifest_path, resume_args=drifted
        )


def test_release_resume_protocol_path_relocation_requires_opt_in(
    tmp_path: Path,
) -> None:
    args, lineage, manifest_path = _release_lineage_resume_args(tmp_path)
    relocated = dict(lineage)
    relocated["_sophia_protocol_path"] = str(tmp_path / "relocated_protocol.json")

    with pytest.raises(SophiaUsageError):
        _prepare_release_bootstrap_for_test(
            args, manifest_path=manifest_path, resume_args=relocated
        )

    opted_in = replace(args, _sophia_allow_protocol_path_relocation=1)
    resumed = _prepare_release_bootstrap_for_test(
        opted_in, manifest_path=manifest_path, resume_args=relocated
    )
    assert resumed.protocol_sha256 == lineage["_sophia_protocol_sha256"]


def test_release_resume_protocol_relocation_does_not_bypass_hash(
    tmp_path: Path,
) -> None:
    args, lineage, manifest_path = _release_lineage_resume_args(tmp_path)
    relocated = dict(lineage)
    relocated["_sophia_protocol_path"] = str(tmp_path / "relocated_protocol.json")
    relocated["_sophia_protocol_sha256"] = "f" * 64

    with pytest.raises(SophiaUsageError):
        _prepare_release_bootstrap_for_test(
            replace(args, _sophia_allow_protocol_path_relocation=1),
            manifest_path=manifest_path,
            resume_args=relocated,
        )


def test_release_resume_recipe_migration_accepts_explicit_source(
    tmp_path: Path,
) -> None:
    args, lineage, manifest_path = _release_lineage_resume_args(tmp_path)
    migrated_recipe = tmp_path / "migrated_machine_recipe.json"
    migrated_recipe.write_bytes(b"graph-fix-machine-recipe")
    migrated_args = replace(
        args,
        machine_recipe_json=str(migrated_recipe),
        _sophia_machine_recipe_migration_from_sha256=lineage[
            "_sophia_machine_recipe_sha256"
        ],
    )

    bootstrap = _prepare_release_bootstrap_for_test(
        migrated_args,
        manifest_path=manifest_path,
        resume_args=lineage,
    )

    assert bootstrap.machine_recipe_path == str(migrated_recipe.resolve())
    assert bootstrap.machine_recipe_sha256 == hashlib.sha256(
        migrated_recipe.read_bytes()
    ).hexdigest()


def test_release_resume_recipe_migration_rejects_wrong_source(
    tmp_path: Path,
) -> None:
    args, lineage, manifest_path = _release_lineage_resume_args(tmp_path)
    migrated_recipe = tmp_path / "migrated_machine_recipe.json"
    migrated_recipe.write_bytes(b"graph-fix-machine-recipe")
    migrated_args = replace(
        args,
        machine_recipe_json=str(migrated_recipe),
        _sophia_machine_recipe_migration_from_sha256="f" * 64,
    )

    with pytest.raises(SophiaUsageError):
        _prepare_release_bootstrap_for_test(
            migrated_args,
            manifest_path=manifest_path,
            resume_args=lineage,
        )


def test_release_resume_recipe_migration_does_not_bypass_other_lineage(
    tmp_path: Path,
) -> None:
    args, lineage, manifest_path = _release_lineage_resume_args(tmp_path)
    migrated_recipe = tmp_path / "migrated_machine_recipe.json"
    migrated_recipe.write_bytes(b"graph-fix-machine-recipe")
    drifted = dict(lineage)
    drifted["_sophia_protocol_sha256"] = "f" * 64
    migrated_args = replace(
        args,
        machine_recipe_json=str(migrated_recipe),
        _sophia_machine_recipe_migration_from_sha256=lineage[
            "_sophia_machine_recipe_sha256"
        ],
    )

    with pytest.raises(SophiaUsageError):
        _prepare_release_bootstrap_for_test(
            migrated_args,
            manifest_path=manifest_path,
            resume_args=drifted,
        )


def test_release_resume_uses_admission_policy_hash_not_marker_hash(
    tmp_path: Path,
) -> None:
    args, lineage, manifest_path = _release_lineage_resume_args(tmp_path)
    bootstrap = _prepare_release_bootstrap_for_test(
        args, manifest_path=manifest_path, resume_args=None
    )
    assert bootstrap.data_admission_sha256 == lineage["_sophia_data_admission_sha256"]
    assert bootstrap.data_admission_sha256 != bootstrap.dataset_marker_sha256


def test_release_protocol_model_hash_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _, manifest_path = _release_lineage_resume_args(tmp_path)
    protocol = json.loads(runtime_bootstrap_mod._RELEASE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["model"]["spec_sha256"] = "0" * 64
    drifted_path = tmp_path / "pretrain_release_protocol.json"
    drifted_path.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(runtime_bootstrap_mod, "_RELEASE_PROTOCOL_PATH", drifted_path)
    with pytest.raises(SophiaUsageError):
        _prepare_release_bootstrap_for_test(
            args, manifest_path=manifest_path, resume_args=None
        )


def test_apply_resume_runtime_settings_keeps_native_runtime_when_resume_args_empty() -> (
    None
):
    args = _pretrain_cfg(seq_len=4096, max_steps=64)
    runtime_state = PretrainRuntimeState.from_config(args)

    resolved = resume_restore_mod.apply_resume_runtime_settings(
        args=args,
        resume_args={},
        runtime_state=runtime_state,
    )

    assert isinstance(resolved, PretrainRunConfig)
    runtime_payload = DEFAULT_PRETRAIN_MACHINE_RUNTIME
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


def test_apply_resume_recipe_settings_allows_explicit_probe_overrides() -> None:
    args = _pretrain_cfg(max_steps=4, total_tokens=4)
    args._sophia_resume_recipe_override_fields = ("max_steps", "total_tokens")
    runtime_state = PretrainRuntimeState.from_config(args)

    resolved = resume_restore_mod.apply_resume_recipe_settings(
        args=args,
        resume_args={"max_steps": 15259, "total_tokens": 20_000_000_000},
        runtime_state=runtime_state,
    )

    assert int(resolved.max_steps) == 4
    assert int(resolved.total_tokens) == 4
