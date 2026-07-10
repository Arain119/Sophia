from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch

from ml.data.token_shards.shard_manifest import write_json_atomic
from ml.tasks.pretrain.observability.support import log_tag, try_load_json
from ml.tasks.pretrain.planner import StagePlan
from ml.training.pretrain.profiles import (
    PretrainProfile,
    pretrain_profile_to_payload,
    resolve_pretrain_profile,
)
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.runtime_state import PretrainRuntimeState
from ml.training.pretrain import manifest_policy
from ml.training.pretrain.artifacts import (
    PRETRAIN_MACHINE_RECIPE_ARTIFACT,
    PRETRAIN_MACHINE_RECIPE_SUMMARY,
    PRETRAIN_MACHINE_RUNTIME_ARTIFACT,
    PRETRAIN_PROFILE_ARTIFACT,
)
from ml.training.pretrain.semantic_defaults import (
    learning_rate_for_tokens_per_update,
)


@dataclass(frozen=True)
class ReleasePretrainDatasetSnapshot:
    train_data_path: str
    val_data_path: str
    test_data_path: str
    train_manifest_path: str
    train_manifest_sha1: str
    val_manifest_path: str
    val_manifest_sha1: str
    test_manifest_path: str
    test_manifest_sha1: str
    tokenizer_path: str = ""

    def to_payload(self) -> dict[str, object]:
        return {
            "train_data_path": str(self.train_data_path),
            "val_data_path": str(self.val_data_path),
            "test_data_path": str(self.test_data_path),
            "train_manifest_path": str(self.train_manifest_path),
            "train_manifest_sha1": str(self.train_manifest_sha1),
            "val_manifest_path": str(self.val_manifest_path),
            "val_manifest_sha1": str(self.val_manifest_sha1),
            "test_manifest_path": str(self.test_manifest_path),
            "test_manifest_sha1": str(self.test_manifest_sha1),
            "tokenizer_path": str(self.tokenizer_path),
        }


@dataclass(frozen=True)
class ReleasePretrainRuntimeSnapshot:
    device: str
    total_tokens: int
    max_seq_len: int
    initial_seq_len: int

    def to_payload(self) -> dict[str, object]:
        return {
            "device": str(self.device),
            "total_tokens": int(self.total_tokens),
            "max_seq_len": int(self.max_seq_len),
            "initial_seq_len": int(self.initial_seq_len),
        }


@dataclass(frozen=True)
class ReleasePretrainCurriculumStageSnapshot:
    stage_index: int
    seq_len: int
    token_budget: int
    start_tokens: int
    end_tokens: int
    start_step: int
    end_step: int
    batch_size: int
    accumulation_steps: int
    tokens_per_update: int

    @classmethod
    def from_stage_plan(
        cls,
        plan: StagePlan,
    ) -> ReleasePretrainCurriculumStageSnapshot:
        return cls(
            stage_index=int(plan.stage.stage_index),
            seq_len=int(plan.recipe.seq_len),
            token_budget=int(plan.stage.token_budget),
            start_tokens=int(plan.stage.start_tokens),
            end_tokens=int(plan.stage.end_tokens),
            start_step=int(plan.start_step),
            end_step=int(plan.end_step),
            batch_size=int(plan.recipe.batch_size),
            accumulation_steps=int(plan.recipe.accumulation_steps),
            tokens_per_update=int(plan.recipe.tokens_per_update),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "stage_index": int(self.stage_index),
            "seq_len": int(self.seq_len),
            "token_budget": int(self.token_budget),
            "start_tokens": int(self.start_tokens),
            "end_tokens": int(self.end_tokens),
            "start_step": int(self.start_step),
            "end_step": int(self.end_step),
            "batch_size": int(self.batch_size),
            "accumulation_steps": int(self.accumulation_steps),
            "tokens_per_update": int(self.tokens_per_update),
        }


@dataclass(frozen=True)
class ReleasePretrainProfileReport:
    run_kind: str
    profile: dict[str, object]
    dataset: ReleasePretrainDatasetSnapshot
    runtime: ReleasePretrainRuntimeSnapshot
    curriculum: list[ReleasePretrainCurriculumStageSnapshot] = field(default_factory=list)

    def to_payload(self) -> dict[str, object]:
        return {
            "run_kind": str(self.run_kind),
            "profile": dict(self.profile),
            "dataset": self.dataset.to_payload(),
            "runtime": self.runtime.to_payload(),
            "curriculum": [stage.to_payload() for stage in self.curriculum],
        }


@dataclass(frozen=True)
class MachineRecipeSelectedSettings:
    max_steps: int
    seq_len: int
    batch_size: int
    accumulation_steps: int
    tokens_per_update: int
    gradient_checkpointing: int
    loss_chunk_size: int
    learning_rate: float
    weight_decay: float
    lr_schedule: str
    warmup_steps: int
    warmup_ratio: float
    min_lr_ratio: float
    wsd_stable_ratio: float
    wsd_decay_style: str
    layerwise_lr_decay: float
    embedding_lr_scale: float
    muon_ns_steps: int
    muon_target_rms: float | None
    grad_clip_mode: str
    max_grad_norm: float
    agc_clip: float
    agc_eps: float
    ema_decay: float
    ema_update_interval: int
    log_interval: int
    eval_interval: int
    eval_steps: int
    save_interval: int
    step_execution_backend: str
    dataloader_num_workers: int
    dataloader_prefetch_factor: int
    dataloader_persistent_workers: int
    shard_preload: int
    shard_preload_bytes: int

    def to_payload(self) -> dict[str, object]:
        return {
            "max_steps": int(self.max_steps),
            "seq_len": int(self.seq_len),
            "batch_size": int(self.batch_size),
            "accumulation_steps": int(self.accumulation_steps),
            "tokens_per_update": int(self.tokens_per_update),
            "gradient_checkpointing": int(self.gradient_checkpointing),
            "loss_chunk_size": int(self.loss_chunk_size),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "lr_schedule": str(self.lr_schedule),
            "warmup_steps": int(self.warmup_steps),
            "warmup_ratio": float(self.warmup_ratio),
            "min_lr_ratio": float(self.min_lr_ratio),
            "wsd_stable_ratio": float(self.wsd_stable_ratio),
            "wsd_decay_style": str(self.wsd_decay_style),
            "layerwise_lr_decay": float(self.layerwise_lr_decay),
            "embedding_lr_scale": float(self.embedding_lr_scale),
            "muon_ns_steps": int(self.muon_ns_steps),
            "muon_target_rms": (
                None if self.muon_target_rms is None else float(self.muon_target_rms)
            ),
            "grad_clip_mode": str(self.grad_clip_mode),
            "max_grad_norm": float(self.max_grad_norm),
            "agc_clip": float(self.agc_clip),
            "agc_eps": float(self.agc_eps),
            "ema_decay": float(self.ema_decay),
            "ema_update_interval": int(self.ema_update_interval),
            "log_interval": int(self.log_interval),
            "eval_interval": int(self.eval_interval),
            "eval_steps": int(self.eval_steps),
            "save_interval": int(self.save_interval),
            "step_execution_backend": str(self.step_execution_backend),
            "dataloader_num_workers": int(self.dataloader_num_workers),
            "dataloader_prefetch_factor": int(self.dataloader_prefetch_factor),
            "dataloader_persistent_workers": int(self.dataloader_persistent_workers),
            "shard_preload": int(self.shard_preload),
            "shard_preload_bytes": int(self.shard_preload_bytes),
        }


@dataclass(frozen=True)
class MachineRecipeDataSnapshot:
    train_manifest: str
    train_manifest_sha1: str | None
    val_manifest: str | None
    val_manifest_sha1: str | None
    total_tokens: int

    def to_payload(self) -> dict[str, object]:
        return {
            "train_manifest": str(self.train_manifest),
            "train_manifest_sha1": self.train_manifest_sha1,
            "val_manifest": self.val_manifest,
            "val_manifest_sha1": self.val_manifest_sha1,
            "total_tokens": int(self.total_tokens),
        }


@dataclass(frozen=True)
class MachineRecipeArtifacts:
    machine_recipe: str | None
    machine_runtime: str | None

    def to_payload(self) -> dict[str, object]:
        return {
            "machine_recipe": self.machine_recipe,
            "machine_runtime": self.machine_runtime,
        }


@dataclass(frozen=True)
class MachineRecipeSummaryReport:
    time: float
    output_dir: str
    base_dtype: str
    device: str
    data: MachineRecipeDataSnapshot
    selected: MachineRecipeSelectedSettings
    artifacts: MachineRecipeArtifacts
    machine_recipe_best: object | None = None
    machine_runtime_best: object | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "time": float(self.time),
            "output_dir": str(self.output_dir),
            "precision": {"base_dtype": str(self.base_dtype)},
            "device": str(self.device),
            "data": self.data.to_payload(),
            "selected": self.selected.to_payload(),
            "artifacts": self.artifacts.to_payload(),
            "machine_recipe_best": self.machine_recipe_best,
            "machine_runtime_best": self.machine_runtime_best,
        }


def _resolved_manifest_path(data_path: str) -> str:
    if not str(data_path).strip():
        return ""
    return os.path.abspath(str(manifest_policy.resolve_manifest_path(str(data_path)) or ""))


def build_release_pretrain_profile_report_from_config(
    *,
    cfg: PretrainRunConfig,
    manifest_path: str,
    tokenizer_path: str,
    stage_execution_plans: list[StagePlan],
) -> ReleasePretrainProfileReport:
    profile: PretrainProfile = resolve_pretrain_profile(cfg)
    state = PretrainRuntimeState.from_config(cfg)
    run_kind = str(cfg._sophia_run_kind or "pretrain")
    return ReleasePretrainProfileReport(
        run_kind=run_kind,
        profile=dict(pretrain_profile_to_payload(profile)),
        dataset=ReleasePretrainDatasetSnapshot(
            train_data_path=os.path.abspath(str(cfg.data_path)),
            val_data_path=os.path.abspath(str(cfg.eval_data_path)),
            test_data_path=os.path.abspath(str(cfg.test_data_path)),
            train_manifest_path=os.path.abspath(str(manifest_path)),
            train_manifest_sha1=str(cfg._sophia_train_manifest_sha1),
            val_manifest_path=_resolved_manifest_path(str(cfg.eval_data_path)),
            val_manifest_sha1=str(cfg._sophia_val_manifest_sha1),
            test_manifest_path=_resolved_manifest_path(str(cfg.test_data_path)),
            test_manifest_sha1=str(cfg._sophia_test_manifest_sha1),
            tokenizer_path=os.path.abspath(str(tokenizer_path)),
        ),
        runtime=ReleasePretrainRuntimeSnapshot(
            device=str(cfg.device),
            total_tokens=int(cfg.total_tokens),
            max_seq_len=int(cfg.max_seq_len),
            initial_seq_len=int(state.seq_len),
        ),
        curriculum=[
            ReleasePretrainCurriculumStageSnapshot.from_stage_plan(plan)
            for plan in stage_execution_plans
        ],
    )


def _optional_output_artifact(path: str) -> str | None:
    return str(path) if os.path.exists(str(path)) else None


def _sha1_or_none(path: str) -> str | None:
    if not path or not os.path.exists(str(path)):
        return None
    from ml.data.token_shards.shard_manifest import sha1_file

    try:
        return sha1_file(str(path)).lower()
    except Exception:
        return None


def build_machine_recipe_summary_report_from_config(
    *,
    cfg: PretrainRunConfig,
    output_dir: str,
    train_manifest_path: str,
    max_steps: int,
    seq_len: int,
    base_dtype: torch.dtype,
    device: torch.device,
) -> tuple[str, MachineRecipeSummaryReport]:
    output_dir = os.path.abspath(str(output_dir))
    summary_path = os.path.join(str(output_dir), PRETRAIN_MACHINE_RECIPE_SUMMARY)
    state = PretrainRuntimeState.from_config(cfg)
    machine_runtime = state.machine_runtime()

    eval_manifest_path = manifest_policy.resolve_manifest_path(str(cfg.eval_data_path))
    recipe_path = os.path.join(str(output_dir), PRETRAIN_MACHINE_RECIPE_ARTIFACT)
    runtime_path = os.path.join(str(output_dir), PRETRAIN_MACHINE_RUNTIME_ARTIFACT)
    recipe_obj = try_load_json(path=str(recipe_path))
    runtime_obj = try_load_json(path=str(runtime_path))

    tokens_per_update = (
        int(seq_len) * int(state.batch_size) * int(state.accumulation_steps)
    )
    runtime_learning_rate = learning_rate_for_tokens_per_update(
        semantic_learning_rate=float(state.learning_rate),
        tokens_per_update=int(tokens_per_update),
        target_tokens_per_update=int(state.target_tokens_per_update),
    )
    return summary_path, MachineRecipeSummaryReport(
        time=float(time.time()),
        output_dir=str(output_dir),
        base_dtype=str(base_dtype),
        device=str(device),
        data=MachineRecipeDataSnapshot(
            train_manifest=os.path.abspath(str(train_manifest_path)),
            train_manifest_sha1=_sha1_or_none(str(train_manifest_path)),
            val_manifest=(
                os.path.abspath(str(eval_manifest_path)) if eval_manifest_path else None
            ),
            val_manifest_sha1=_sha1_or_none(str(eval_manifest_path))
            if eval_manifest_path
            else None,
            total_tokens=int(cfg.total_tokens),
        ),
        selected=MachineRecipeSelectedSettings(
            max_steps=int(max_steps),
            seq_len=int(seq_len),
            batch_size=int(state.batch_size),
            accumulation_steps=int(state.accumulation_steps),
            tokens_per_update=int(tokens_per_update),
            gradient_checkpointing=int(state.gradient_checkpointing),
            loss_chunk_size=int(state.loss_chunk_size),
            learning_rate=float(runtime_learning_rate),
            weight_decay=float(cfg.weight_decay),
            lr_schedule=str(cfg.lr_schedule),
            warmup_steps=int(cfg.warmup_steps),
            warmup_ratio=float(cfg.warmup_ratio),
            min_lr_ratio=float(cfg.min_lr_ratio),
            wsd_stable_ratio=float(cfg.wsd_stable_ratio),
            wsd_decay_style=str(cfg.wsd_decay_style),
            layerwise_lr_decay=float(cfg.layerwise_lr_decay),
            embedding_lr_scale=float(cfg.embedding_lr_scale),
            muon_ns_steps=int(cfg.muon_ns_steps),
            muon_target_rms=(
                None if cfg.muon_target_rms is None else float(cfg.muon_target_rms)
            ),
            grad_clip_mode=str(cfg.grad_clip_mode),
            max_grad_norm=float(cfg.max_grad_norm),
            agc_clip=float(cfg.agc_clip),
            agc_eps=float(cfg.agc_eps),
            ema_decay=float(state.ema_decay),
            ema_update_interval=int(state.ema_update_interval),
            log_interval=int(state.log_interval),
            eval_interval=int(state.eval_interval),
            eval_steps=int(state.eval_steps),
            save_interval=int(state.save_interval),
            step_execution_backend=str(machine_runtime.step_execution_backend or "eager"),
            dataloader_num_workers=int(machine_runtime.dataloader_num_workers),
            dataloader_prefetch_factor=int(machine_runtime.dataloader_prefetch_factor),
            dataloader_persistent_workers=int(
                machine_runtime.dataloader_persistent_workers
            ),
            shard_preload=int(machine_runtime.shard_preload),
            shard_preload_bytes=int(machine_runtime.shard_preload_bytes),
        ),
        artifacts=MachineRecipeArtifacts(
            machine_recipe=_optional_output_artifact(recipe_path),
            machine_runtime=_optional_output_artifact(runtime_path),
        ),
        machine_recipe_best=(
            recipe_obj.get("best") if isinstance(recipe_obj, dict) else None
        ),
        machine_runtime_best=(
            runtime_obj.get("best") if isinstance(runtime_obj, dict) else None
        ),
    )


def write_machine_recipe_summary(
    *,
    args: PretrainRunConfig,
    output_dir: str,
    train_manifest_path: str,
    max_steps: int,
    seq_len: int,
    base_dtype: torch.dtype,
    device: torch.device,
) -> None:
    try:
        summary_path, report = build_machine_recipe_summary_report_from_config(
            cfg=args,
            output_dir=output_dir,
            train_manifest_path=train_manifest_path,
            max_steps=max_steps,
            seq_len=seq_len,
            base_dtype=base_dtype,
            device=device,
        )
        write_json_atomic(str(summary_path), report.to_payload())
        print(f"{log_tag()} wrote {PRETRAIN_MACHINE_RECIPE_SUMMARY}", flush=True)
    except Exception as exc:
        print(f"[WARN] unable to write {PRETRAIN_MACHINE_RECIPE_SUMMARY}: {exc}", flush=True)


def write_release_pretrain_profile(
    *,
    args: PretrainRunConfig,
    output_dir: str,
    manifest_path: str,
    tokenizer_path: str,
    stage_execution_plans: list[StagePlan],
) -> None:
    try:
        report = build_release_pretrain_profile_report_from_config(
            cfg=args,
            manifest_path=manifest_path,
            tokenizer_path=tokenizer_path,
            stage_execution_plans=stage_execution_plans,
        )
        write_json_atomic(
            os.path.join(os.path.abspath(str(output_dir)), PRETRAIN_PROFILE_ARTIFACT),
            report.to_payload(),
        )
    except Exception as exc:
        print(f"[WARN] unable to write {PRETRAIN_PROFILE_ARTIFACT}: {exc}", flush=True)


__all__ = [
    "ReleasePretrainCurriculumStageSnapshot",
    "ReleasePretrainDatasetSnapshot",
    "ReleasePretrainProfileReport",
    "ReleasePretrainRuntimeSnapshot",
    "MachineRecipeArtifacts",
    "MachineRecipeDataSnapshot",
    "MachineRecipeSelectedSettings",
    "MachineRecipeSummaryReport",
    "build_release_pretrain_profile_report_from_config",
    "build_machine_recipe_summary_report_from_config",
    "write_release_pretrain_profile",
    "write_machine_recipe_summary",
]
