#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.release_gate import (
    PRETRAIN_MACHINE_RECIPE_KIND,
    build_pretrain_data_signature,
)
from ml.training.pretrain.release_config import (
    CANONICAL_PRETRAIN_MACHINE_RECIPE_NAME,
    CANONICAL_PRETRAIN_MACHINE_RECIPE_PAYLOAD,
    CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD,
    RELEASE_PRETRAIN_DEFAULTS,
    PretrainReleaseSemantics,
)
from ml.training.pretrain.artifacts import (
    PRETRAIN_MACHINE_RECIPE_ARTIFACT,
    PRETRAIN_MACHINE_RECIPE_SUMMARY,
    PRETRAIN_MACHINE_RUNTIME_ARTIFACT,
    PRETRAIN_SIGNED_MACHINE_RECIPE_ARTIFACT,
    PRETRAIN_UPDATE_PROFILE_ARTIFACT,
)

from ml.tasks.pretrain.pipeline import PretrainPipeline, build_pretrain_args
from ml.training.pretrain.profiles import (
    RELEASE_PROFILE,
)
from ml.training.pretrain.runtime_bootstrap import PretrainRuntimeBootstrap
from ml.training.pretrain import manifest_policy
from ml.tooling.core.pareto import pareto_frontier

PINNED_PRETRAIN_MACHINE_VALIDATION_TOTAL_TOKENS = 65_536


def _json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _default_quality_probe_total_tokens(
    *,
    target_tokens_per_update: int,
    default_total_tokens: int,
) -> int:
    del target_tokens_per_update
    return max(
        int(default_total_tokens),
        int(PINNED_PRETRAIN_MACHINE_VALIDATION_TOTAL_TOKENS),
    )


def _metrics_summary(path: Path) -> dict[str, Any]:
    best_val_loss: float | None = None
    last_val_loss: float | None = None
    test_loss: float | None = None
    contamination_safe_rate: float | None = None
    start_time: float | None = None
    first_eval_time: float | None = None
    end_time: float | None = None
    eval_steps = 0
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            typ = str(obj.get("type", "") or "")
            raw_time = obj.get("time")
            if isinstance(raw_time, (int, float)):
                if typ == "start" and start_time is None:
                    start_time = float(raw_time)
                if typ == "end":
                    end_time = float(raw_time)
            if typ == "eval" and isinstance(obj.get("val_loss"), (int, float)):
                val = float(obj["val_loss"])
                last_val_loss = val
                eval_steps += 1
                if isinstance(raw_time, (int, float)) and first_eval_time is None:
                    first_eval_time = float(raw_time)
                if best_val_loss is None or val < best_val_loss:
                    best_val_loss = val
            if str(obj.get("name", "") or "") == "test_loss" and isinstance(
                obj.get("value"), (int, float)
            ):
                test_loss = float(obj["value"])
            if str(obj.get("name", "") or "") == "pretrain_contamination_safe_rate" and isinstance(
                obj.get("value"), (int, float)
            ):
                contamination_safe_rate = float(obj["value"])
    out: dict[str, Any] = {"eval_events": int(eval_steps)}
    if best_val_loss is not None:
        out["best_val_loss"] = float(best_val_loss)
    if last_val_loss is not None:
        out["last_val_loss"] = float(last_val_loss)
    if test_loss is not None:
        out["test_loss"] = float(test_loss)
    if contamination_safe_rate is not None:
        out["contamination_safe_rate"] = float(contamination_safe_rate)
    if start_time is not None and end_time is not None and end_time >= start_time:
        out["train_wall_time_s"] = float(end_time - start_time)
    if (
        start_time is not None
        and first_eval_time is not None
        and first_eval_time >= start_time
    ):
        out["first_update_wall_time_s"] = float(first_eval_time - start_time)
    return out


def _run_summary(
    *,
    name: str,
    semantics: dict[str, object],
    run_dir: Path,
    wall_time_s: float,
    error: str = "",
) -> dict[str, Any]:
    machine_recipe = _json_or_none(run_dir / PRETRAIN_MACHINE_RECIPE_ARTIFACT) or {}
    machine_runtime = _json_or_none(run_dir / PRETRAIN_MACHINE_RUNTIME_ARTIFACT) or {}
    update_profile = _json_or_none(run_dir / PRETRAIN_UPDATE_PROFILE_ARTIFACT) or {}
    machine_recipe_summary = _json_or_none(run_dir / PRETRAIN_MACHINE_RECIPE_SUMMARY) or {}
    metrics = _metrics_summary(run_dir / "metrics.jsonl")

    recipe_best = machine_recipe.get("best")
    recipe_best = recipe_best if isinstance(recipe_best, dict) else {}
    recipe_candidate = recipe_best.get("candidate")
    recipe_candidate = recipe_candidate if isinstance(recipe_candidate, dict) else {}
    runtime_best = machine_runtime.get("best")
    runtime_best = runtime_best if isinstance(runtime_best, dict) else {}
    selected = machine_recipe_summary.get("selected")
    selected = selected if isinstance(selected, dict) else {}

    summary = {
        "name": str(name),
        "status": "ok" if not error else "error",
        "error": str(error),
        "output_dir": str(run_dir.resolve()),
        "wall_time_s": float(wall_time_s),
        "release_semantics": dict(semantics),
        "machine_adaptive_selected": {
            "seq_len": int(selected.get("seq_len", 0) or 0),
            "batch_size": int(selected.get("batch_size", 0) or 0),
            "accumulation_steps": int(selected.get("accumulation_steps", 0) or 0),
            "gradient_checkpointing": int(selected.get("gradient_checkpointing", 0) or 0),
            "loss_chunk_size": int(selected.get("loss_chunk_size", 0) or 0),
            "dataloader_num_workers": int(selected.get("dataloader_num_workers", 0) or 0),
            "dataloader_prefetch_factor": int(
                selected.get("dataloader_prefetch_factor", 0) or 0
            ),
            "dataloader_persistent_workers": int(
                selected.get("dataloader_persistent_workers", 0) or 0
            ),
            "shard_preload": int(selected.get("shard_preload", 0) or 0),
            "shard_preload_bytes": int(selected.get("shard_preload_bytes", 0) or 0),
        },
        "recipe": {
            "batch_size": int(recipe_best.get("batch_size", 0) or 0),
            "accumulation_steps": int(recipe_best.get("accumulation_steps", 0) or 0),
            "tokens_per_sec": float(recipe_best.get("tokens_per_sec", 0.0) or 0.0),
            "max_memory_gb": float(recipe_best.get("max_memory_gb", 0.0) or 0.0),
            "step_time_s": float(recipe_best.get("step_time_s", 0.0) or 0.0),
            "gradient_checkpointing": bool(
                recipe_candidate.get("gradient_checkpointing", False)
            ),
            "gradient_checkpointing_exclude_first": int(
                recipe_candidate.get("gradient_checkpointing_exclude_first", 0) or 0
            ),
            "gradient_checkpointing_exclude_last": int(
                recipe_candidate.get("gradient_checkpointing_exclude_last", 0) or 0
            ),
            "loss_chunk_size": int(recipe_candidate.get("loss_chunk_size", 0) or 0),
        },
        "runtime": {
            "tokens_per_sec": float(runtime_best.get("tokens_per_sec", 0.0) or 0.0),
            "step_time_s": float(runtime_best.get("step_time_s", 0.0) or 0.0),
            "shard_preload": int(runtime_best.get("shard_preload", 0) or 0),
            "shard_preload_bytes": int(runtime_best.get("shard_preload_bytes", 0) or 0),
        },
        "profile": {
            "tokens_per_sec_est": float(
                update_profile.get("tokens_per_sec_est", 0.0) or 0.0
            ),
            "tokens_per_update": int(update_profile.get("tokens_per_update", 0) or 0),
        },
        "metrics": metrics,
    }
    if not error:
        last_val_loss = metrics.get("last_val_loss", metrics.get("best_val_loss"))
        tokens_per_sec = summary["profile"].get("tokens_per_sec_est") or summary[
            "recipe"
        ].get("tokens_per_sec")
        if not (
            isinstance(last_val_loss, (int, float))
            and math.isfinite(float(last_val_loss))
            and isinstance(tokens_per_sec, (int, float))
            and math.isfinite(float(tokens_per_sec))
        ):
            summary["status"] = "incomplete"
            summary["error"] = "missing_or_non_finite_objective_metrics"
    return summary


def _recipe_payload(
    *,
    run_config: PretrainRunConfig,
    run_dir: Path,
) -> dict[str, Any]:
    run_args = _json_or_none(run_dir / "run_args.json")
    if not isinstance(run_args, dict):
        raise RuntimeError(f"missing run_args.json under {run_dir}")
    machine_signature = run_args.get("_sophia_machine_signature")
    if not isinstance(machine_signature, dict) or not machine_signature:
        raise RuntimeError(f"run_args.json is missing _sophia_machine_signature under {run_dir}")
    machine_recipe_artifact = _json_or_none(run_dir / PRETRAIN_MACHINE_RECIPE_ARTIFACT)
    machine_runtime_artifact = _json_or_none(run_dir / PRETRAIN_MACHINE_RUNTIME_ARTIFACT)
    update_profile = _json_or_none(run_dir / PRETRAIN_UPDATE_PROFILE_ARTIFACT)
    if not isinstance(machine_recipe_artifact, dict):
        raise RuntimeError(
            f"missing {PRETRAIN_MACHINE_RECIPE_ARTIFACT} under {run_dir}"
        )
    if not isinstance(machine_runtime_artifact, dict):
        raise RuntimeError(
            f"missing {PRETRAIN_MACHINE_RUNTIME_ARTIFACT} under {run_dir}"
        )
    if not isinstance(update_profile, dict):
        raise RuntimeError(f"missing {PRETRAIN_UPDATE_PROFILE_ARTIFACT} under {run_dir}")
    bootstrap = PretrainRuntimeBootstrap(
        output_dir=str(run_dir),
        resume_path=None,
        resume_checkpoint=None,
        resolved_resume_checkpoint="",
        data_path=str(run_args.get("data_path", "") or ""),
        eval_data_path=str(run_args.get("eval_data_path", "") or ""),
        test_data_path=str(run_args.get("test_data_path", "") or ""),
        manifest_path=str(
            manifest_policy.resolve_manifest_path(str(run_args.get("data_path", "") or ""))
        ),
        manifest=object(),
        total_tokens=int(run_args.get("total_tokens", 0) or 0),
        tokenizer=object(),
        tokenizer_path=str(run_args.get("tokenizer_path", "") or ""),
        seq_len=int(run_args.get("seq_len", 0) or 0),
        train_manifest_sha1=str(run_args.get("_sophia_train_manifest_sha1", "") or ""),
        val_manifest_sha1=str(run_args.get("_sophia_val_manifest_sha1", "") or ""),
        test_manifest_sha1=str(run_args.get("_sophia_test_manifest_sha1", "") or ""),
    )
    state = {
        "batch_size": int(run_args.get("batch_size", 0) or 0),
        "accumulation_steps": int(run_args.get("accumulation_steps", 0) or 0),
        "gradient_checkpointing": int(run_args.get("gradient_checkpointing", 0) or 0),
        "gradient_checkpointing_exclude_first": int(
            run_args.get("gradient_checkpointing_exclude_first", 0) or 0
        ),
        "gradient_checkpointing_exclude_last": int(
            run_args.get("gradient_checkpointing_exclude_last", 0) or 0
        ),
        "loss_chunk_size": int(run_args.get("loss_chunk_size", 0) or 0),
        "_sophia_pretrain_profile": run_config._sophia_pretrain_profile,
        "target_tokens_per_update": int(run_args.get("target_tokens_per_update", 0) or 0),
        "learning_rate": float(run_args.get("learning_rate", 0.0) or 0.0),
        "weight_decay": float(run_args.get("weight_decay", 0.0) or 0.0),
        "warmup_steps": int(run_args.get("warmup_steps", 0) or 0),
        "warmup_ratio": float(run_args.get("warmup_ratio", 0.0) or 0.0),
        "min_lr_ratio": float(run_args.get("min_lr_ratio", 0.0) or 0.0),
        "wsd_stable_ratio": float(run_args.get("wsd_stable_ratio", 0.0) or 0.0),
        "max_grad_norm": float(run_args.get("max_grad_norm", 0.0) or 0.0),
        "lr_schedule": str(run_args.get("lr_schedule", "") or ""),
        "dataloader_num_workers": _int_from_mapping(run_args, "dataloader_num_workers", default=-1),
        "dataloader_prefetch_factor": _int_from_mapping(run_args, "dataloader_prefetch_factor", default=-1),
        "dataloader_persistent_workers": _int_from_mapping(run_args, "dataloader_persistent_workers", default=-1),
        "shard_preload": _int_from_mapping(run_args, "shard_preload", default=-1),
        "shard_preload_bytes": _int_from_mapping(run_args, "shard_preload_bytes", default=-1),
        "machine_recipe_json": "",
        "_sophia_run_kind": "pretrain",
        "_sophia_machine_signature": dict(machine_signature),
    }
    recipe_args = replace(run_config, **state)
    return {
        "kind": PRETRAIN_MACHINE_RECIPE_KIND,
        "stage": "pretrain",
        "selected_name": str(CANONICAL_PRETRAIN_MACHINE_RECIPE_NAME),
        "release_semantics": PretrainReleaseSemantics.from_run_config(
            args=recipe_args
        ).to_payload(),
        "machine_signature": dict(machine_signature),
        "data_signature": build_pretrain_data_signature(
            args=recipe_args,
            bootstrap=bootstrap,
        ),
        "machine_recipe": {
            "batch_size": int(run_args.get("batch_size", 0) or 0),
            "accumulation_steps": int(run_args.get("accumulation_steps", 0) or 0),
            "gradient_checkpointing": int(run_args.get("gradient_checkpointing", 0) or 0),
            "gradient_checkpointing_exclude_first": int(
                run_args.get("gradient_checkpointing_exclude_first", 0) or 0
            ),
            "gradient_checkpointing_exclude_last": int(
                run_args.get("gradient_checkpointing_exclude_last", 0) or 0
            ),
            "loss_chunk_size": int(run_args.get("loss_chunk_size", 0) or 0),
        },
        "machine_runtime": {
            "step_execution_backend": str(
                (((machine_runtime_artifact.get("best") or {}) if isinstance(machine_runtime_artifact, dict) else {}).get("step_execution_backend", ""))
                or ((run_args.get("_sophia_machine_runtime", {}) if isinstance(run_args.get("_sophia_machine_runtime"), dict) else {}).get("step_execution_backend", ""))
                or "eager"
            ),
            "dataloader_num_workers": _int_from_mapping(run_args, "dataloader_num_workers", default=-1),
            "dataloader_prefetch_factor": _int_from_mapping(
                run_args, "dataloader_prefetch_factor", default=-1
            ),
            "dataloader_persistent_workers": _int_from_mapping(
                run_args, "dataloader_persistent_workers", default=-1
            ),
            "shard_preload": _int_from_mapping(run_args, "shard_preload", default=-1),
            "shard_preload_bytes": _int_from_mapping(run_args, "shard_preload_bytes", default=-1),
        },
        "artifacts": {
            "machine_recipe": machine_recipe_artifact,
            "machine_runtime": machine_runtime_artifact,
            "update_profile": update_profile,
        },
    }


def _flatten_for_frontier(summary: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "name": str(summary.get("name", "") or ""),
        "status": str(summary.get("status", "") or ""),
    }
    recipe = summary.get("recipe")
    recipe = recipe if isinstance(recipe, dict) else {}
    profile = summary.get("profile")
    profile = profile if isinstance(profile, dict) else {}
    metrics = summary.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    semantics = summary.get("release_semantics")
    semantics = semantics if isinstance(semantics, dict) else {}
    flat["val_loss"] = metrics.get("last_val_loss", metrics.get("best_val_loss"))
    flat["best_val_loss"] = metrics.get("best_val_loss")
    flat["test_loss"] = metrics.get("test_loss")
    flat["attention_exact_match_rate"] = metrics.get("attention_exact_match_rate")
    flat["contamination_safe_rate"] = metrics.get("contamination_safe_rate")
    flat["tokens_per_sec"] = profile.get("tokens_per_sec_est") or recipe.get(
        "tokens_per_sec"
    )
    flat["max_memory_gb"] = recipe.get("max_memory_gb")
    flat["target_tokens_per_update"] = semantics.get("target_tokens_per_update")
    flat["learning_rate"] = semantics.get("learning_rate")
    flat["adam_eps"] = semantics.get("adam_eps")
    flat["embedding_lr_scale"] = semantics.get("embedding_lr_scale")
    flat["ema_decay"] = semantics.get("ema_decay")
    return flat


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _positive_float(value: object, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out) or out <= 0.0:
        return float(default)
    return float(out)


def _int_from_sources(
    *,
    key: str,
    selected: dict[str, object],
    run_args: dict[str, object],
    default: int = 0,
) -> int:
    raw = selected.get(str(key), run_args.get(str(key), default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _int_from_mapping(
    mapping: dict[str, object],
    key: str,
    *,
    default: int = 0,
) -> int:
    raw = mapping.get(str(key), default)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _str_from_sources(
    *,
    key: str,
    selected: dict[str, object],
    run_args: dict[str, object],
    default: str = "",
) -> str:
    raw = selected.get(str(key), run_args.get(str(key), default))
    return str(raw if raw is not None else default)


def _cuda_device(device_value: str) -> torch.device | None:
    try:
        device = torch.device(str(device_value))
    except Exception:
        return None
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return device


def _reset_cuda_peak_memory(device_value: str) -> None:
    device = _cuda_device(str(device_value))
    if device is None:
        return
    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        return


def _cuda_peak_memory_gb(device_value: str) -> float:
    device = _cuda_device(str(device_value))
    if device is None:
        return 0.0
    try:
        torch.cuda.synchronize(device)
        return float(torch.cuda.max_memory_allocated(device)) / float(1024**3)
    except Exception:
        return 0.0


def _materialize_validation_machine_artifacts(
    *,
    run_dir: Path,
    semantics: dict[str, object],
    wall_time_s: float,
    peak_memory_gb: float,
) -> None:
    run_args = _json_or_none(run_dir / "run_args.json") or {}
    summary = _json_or_none(run_dir / PRETRAIN_MACHINE_RECIPE_SUMMARY) or {}
    selected_raw = summary.get("selected")
    selected = selected_raw if isinstance(selected_raw, dict) else {}
    if not selected and not run_args:
        return

    seq_len = _int_from_sources(key="seq_len", selected=selected, run_args=run_args)
    batch_size = _int_from_sources(key="batch_size", selected=selected, run_args=run_args)
    accumulation_steps = _int_from_sources(
        key="accumulation_steps",
        selected=selected,
        run_args=run_args,
    )
    tokens_per_update = _int_from_sources(
        key="tokens_per_update",
        selected=selected,
        run_args=run_args,
    )
    if tokens_per_update <= 0:
        tokens_per_update = int(seq_len) * int(batch_size) * int(accumulation_steps)

    metrics = _metrics_summary(run_dir / "metrics.jsonl")
    step_time_s = _positive_float(metrics.get("first_update_wall_time_s"))
    if step_time_s <= 0.0:
        step_time_s = _positive_float(metrics.get("train_wall_time_s"))
    if step_time_s <= 0.0:
        step_time_s = _positive_float(wall_time_s)
    tokens_per_sec = (
        float(tokens_per_update) / float(step_time_s)
        if int(tokens_per_update) > 0 and float(step_time_s) > 0.0
        else 0.0
    )

    max_memory_gb = _positive_float(peak_memory_gb)
    backend = str(
        _str_from_sources(
            key="step_execution_backend",
            selected=selected,
            run_args=run_args,
            default=str(semantics.get("backend", "") or "eager"),
        )
        or semantics.get("backend", "")
        or "eager"
    )
    gradient_checkpointing = _int_from_sources(
        key="gradient_checkpointing",
        selected=selected,
        run_args=run_args,
    )
    recipe_payload = {
        "kind": "pretrain_machine_validation_recipe",
        "source": "same_chain_validation",
        "best": {
            "batch_size": int(batch_size),
            "accumulation_steps": int(accumulation_steps),
            "tokens_per_sec": float(tokens_per_sec),
            "max_memory_gb": float(max_memory_gb),
            "step_time_s": float(step_time_s),
            "candidate": {
                "gradient_checkpointing": bool(int(gradient_checkpointing) == 1),
                "gradient_checkpointing_exclude_first": _int_from_sources(
                    key="gradient_checkpointing_exclude_first",
                    selected=selected,
                    run_args=run_args,
                ),
                "gradient_checkpointing_exclude_last": _int_from_sources(
                    key="gradient_checkpointing_exclude_last",
                    selected=selected,
                    run_args=run_args,
                ),
                "loss_chunk_size": _int_from_sources(
                    key="loss_chunk_size",
                    selected=selected,
                    run_args=run_args,
                ),
            },
        },
    }
    runtime_payload = {
        "kind": "pretrain_machine_validation_runtime",
        "source": "same_chain_validation",
        "best": {
            "step_execution_backend": str(backend),
            "tokens_per_sec": float(tokens_per_sec),
            "step_time_s": float(step_time_s),
            "shard_preload": _int_from_sources(
                key="shard_preload",
                selected=selected,
                run_args=run_args,
            ),
            "shard_preload_bytes": _int_from_sources(
                key="shard_preload_bytes",
                selected=selected,
                run_args=run_args,
            ),
            "dataloader_num_workers": _int_from_sources(
                key="dataloader_num_workers",
                selected=selected,
                run_args=run_args,
            ),
            "dataloader_prefetch_factor": _int_from_sources(
                key="dataloader_prefetch_factor",
                selected=selected,
                run_args=run_args,
            ),
            "dataloader_persistent_workers": _int_from_sources(
                key="dataloader_persistent_workers",
                selected=selected,
                run_args=run_args,
            ),
        },
    }
    update_profile_payload = {
        "kind": "pretrain_update_profile",
        "source": "same_chain_validation",
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "accumulation_steps": int(accumulation_steps),
        "tokens_per_update": int(tokens_per_update),
        "tokens_per_sec_est": float(tokens_per_sec),
        "max_memory_gb": float(max_memory_gb),
        "avg_update_breakdown_s": {
            "update_s": float(step_time_s),
        },
    }

    outputs = {
        PRETRAIN_MACHINE_RECIPE_ARTIFACT: recipe_payload,
        PRETRAIN_MACHINE_RUNTIME_ARTIFACT: runtime_payload,
        PRETRAIN_UPDATE_PROFILE_ARTIFACT: update_profile_payload,
    }
    for filename, payload in outputs.items():
        path = run_dir / filename
        if not path.is_file():
            _write_json(path, payload)


def _machine_baseline_row(
    *,
    summary: dict[str, Any],
    run_config: PretrainRunConfig,
) -> dict[str, Any]:
    adaptive = summary.get("machine_adaptive_selected")
    adaptive = adaptive if isinstance(adaptive, dict) else {}
    recipe = summary.get("recipe")
    recipe = recipe if isinstance(recipe, dict) else {}
    runtime = summary.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    seq_len = int(adaptive.get("seq_len", 0) or getattr(run_config, "seq_len", 0) or 0)
    semantics = summary.get("release_semantics")
    semantics = semantics if isinstance(semantics, dict) else {}
    return {
        "name": str(summary.get("name", "") or ""),
        "seq_len": int(seq_len),
        "backend": str(semantics.get("backend", "") or ""),
        "batch_size": int(adaptive.get("batch_size", 0) or recipe.get("batch_size", 0) or 0),
        "accumulation_steps": int(
            adaptive.get("accumulation_steps", 0) or recipe.get("accumulation_steps", 0) or 0
        ),
        "gradient_checkpointing": int(
            adaptive.get("gradient_checkpointing", 0)
            or bool(recipe.get("gradient_checkpointing", False))
        ),
        "loss_chunk_size": int(
            adaptive.get("loss_chunk_size", 0) or recipe.get("loss_chunk_size", 0) or 0
        ),
        "tokens_per_s": float(
            runtime.get("tokens_per_sec", 0.0) or recipe.get("tokens_per_sec", 0.0) or 0.0
        ),
        "step_time_s": float(
            runtime.get("step_time_s", 0.0) or recipe.get("step_time_s", 0.0) or 0.0
        ),
        "max_mem_gb": float(recipe.get("max_memory_gb", 0.0) or 0.0),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the fixed pretrain semantic recipe and report the selected machine-adaptive runtime recipe."
    )
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_tokens")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out/pretrain_machine_recipe",
        help="Validation root; the fixed release run executes under this path.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite_output_dir", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--validation_total_tokens",
        type=int,
        default=0,
        help=(
            "Optional override for the validation token budget. "
            "0 uses the pinned short quality-probe budget; this is intentionally "
            "independent of the release target_tokens_per_update to avoid wasting paid GPU time."
        ),
    )
    parser.add_argument(
        "--target_tokens_per_update",
        type=int,
        default=0,
        help=(
            "Experimental machine-validation override for the release semantic update "
            "budget. 0 uses the pinned release default. Non-zero values run in an "
            "isolated candidate directory and must not be treated as the signed "
            "release recipe without a quality review."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(CANONICAL_PRETRAIN_MACHINE_RECIPE_PAYLOAD["batch_size"]),
        help="Fixed release micro-batch size.",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=int(
            CANONICAL_PRETRAIN_MACHINE_RECIPE_PAYLOAD["accumulation_steps"]
        ),
        help="Fixed release gradient accumulation steps.",
    )
    parser.add_argument(
        "--step_execution_backend",
        type=str,
        default=str(
            CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD[
                "step_execution_backend"
            ]
        ),
        choices=["eager", "inductor"],
        help=(
            "Experimental step execution backend for real-chain validation. "
            "The signed release recipe remains canonical unless all release gates pass."
        ),
    )
    return parser


@dataclass(frozen=True)
class PretrainMachineValidationConfig:
    data_path: str
    tokenizer_path: str
    output_dir: str
    device: str
    seed: int
    overwrite_output_dir: bool
    validation_total_tokens: int
    target_tokens_per_update: int
    batch_size: int
    accumulation_steps: int
    step_execution_backend: str

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> PretrainMachineValidationConfig:
        return cls(
            data_path=str(args.data_path),
            tokenizer_path=str(args.tokenizer_path or ""),
            output_dir=str(args.output_dir),
            device=str(args.device),
            seed=int(args.seed),
            overwrite_output_dir=bool(int(args.overwrite_output_dir)),
            validation_total_tokens=int(args.validation_total_tokens),
            target_tokens_per_update=int(getattr(args, "target_tokens_per_update", 0)),
            batch_size=int(getattr(args, "batch_size", 1)),
            accumulation_steps=int(getattr(args, "accumulation_steps", 64)),
            step_execution_backend=str(
                getattr(args, "step_execution_backend", "eager") or "eager"
            ),
        )

    def resolved_validation_total_tokens(
        self,
        *,
        target_tokens_per_update: int,
        default_total_tokens: int,
    ) -> int:
        if int(self.validation_total_tokens) > 0:
            return int(self.validation_total_tokens)
        return _default_quality_probe_total_tokens(
            target_tokens_per_update=int(target_tokens_per_update),
            default_total_tokens=int(default_total_tokens),
        )

    def build_release_run_config(self, *, profile) -> PretrainRunConfig:
        base_run_config = build_pretrain_args(
            data_path=str(self.data_path),
            tokenizer_path=str(self.tokenizer_path),
            output_dir="",
            overwrite_output_dir=(1 if bool(self.overwrite_output_dir) else 0),
            profile=profile,
        )
        target_tokens_per_update = int(base_run_config.target_tokens_per_update)
        if int(self.target_tokens_per_update) > 0:
            target_tokens_per_update = int(self.target_tokens_per_update)
        return replace(
            base_run_config,
            device=str(self.device),
            seed=int(self.seed),
            async_checkpoint=0,
            async_metrics=0,
            save_total_limit=1,
            save_best=0,
            enable_checkpoints=0,
            dataloader_num_workers=int(
                CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD[
                    "dataloader_num_workers"
                ]
            ),
            dataloader_prefetch_factor=int(
                CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD[
                    "dataloader_prefetch_factor"
                ]
            ),
            dataloader_persistent_workers=int(
                CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD[
                    "dataloader_persistent_workers"
                ]
            ),
            shard_preload=int(
                CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD["shard_preload"]
            ),
            shard_preload_bytes=int(
                CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD[
                    "shard_preload_bytes"
                ]
            ),
            step_execution_backend=str(self.step_execution_backend),
            _sophia_run_kind="pretrain_machine_validate",
            target_tokens_per_update=int(target_tokens_per_update),
            target_tokens_per_microbatch=(
                int(self.batch_size) * int(profile.curriculum.train_seq_len)
            ),
            batch_size=int(self.batch_size),
            accumulation_steps=int(self.accumulation_steps),
            gradient_checkpointing=0,
            gradient_checkpointing_exclude_first=0,
            gradient_checkpointing_exclude_last=0,
            loss_chunk_size=0,
            learning_rate=float(RELEASE_PRETRAIN_DEFAULTS.learning_rate),
            total_tokens=int(
                self.resolved_validation_total_tokens(
                    target_tokens_per_update=int(target_tokens_per_update),
                    default_total_tokens=int(base_run_config.total_tokens),
                )
            ),
        )

    def run_config(
        self,
        *,
        base_run_config: PretrainRunConfig,
        run_dir: Path,
    ) -> PretrainRunConfig:
        return replace(
            base_run_config,
            output_dir=str(run_dir),
            overwrite_output_dir=(1 if bool(self.overwrite_output_dir) else 0),
        )

    def run_name(self, *, base_run_config: PretrainRunConfig) -> str:
        default_target = int(RELEASE_PRETRAIN_DEFAULTS.target_tokens_per_update)
        target = int(base_run_config.target_tokens_per_update)
        if target == default_target:
            suffix = ""
        else:
            suffix = f"_tpu{int(target)}"
        step_backend = str(base_run_config.step_execution_backend or "eager").strip().lower()
        canonical_step_backend = str(
            CANONICAL_PRETRAIN_MACHINE_RUNTIME_PAYLOAD[
                "step_execution_backend"
            ]
        )
        if step_backend and step_backend != canonical_step_backend:
            suffix = f"{suffix}_{step_backend}"
        return f"release_pretrain{suffix}"


def main() -> None:
    config = PretrainMachineValidationConfig.from_namespace(_build_parser().parse_args())
    profile = RELEASE_PROFILE
    root = Path(config.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    base_run_config = config.build_release_run_config(profile=profile)
    run_name = config.run_name(base_run_config=base_run_config)
    semantics = PretrainReleaseSemantics.from_run_config(
        args=base_run_config
    ).to_payload()
    run_dir = root / run_name
    run_config = config.run_config(
        base_run_config=base_run_config,
        run_dir=run_dir,
    )

    started_at = time.perf_counter()
    print(
        "[MACHINE][PRETRAIN] run "
        f"name={run_name} "
        f"target_tokens={int(semantics.get('target_tokens_per_update', 0) or 0)} "
        f"lr={float(semantics.get('learning_rate', 0.0) or 0.0):.3e} "
        f"semantic={json.dumps(semantics, ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )
    error = ""
    _reset_cuda_peak_memory(str(config.device))
    try:
        PretrainPipeline(args=run_config).run()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(
            f"[MACHINE][PRETRAIN] run failed | name={run_name} | {error}",
            flush=True,
        )
        print(
            "[MACHINE][PRETRAIN] traceback:\n" + traceback.format_exc(),
            flush=True,
        )
    wall_time_s = float(time.perf_counter() - started_at)
    peak_memory_gb = _cuda_peak_memory_gb(str(config.device))
    if not error:
        _materialize_validation_machine_artifacts(
            run_dir=run_dir,
            semantics=semantics,
            wall_time_s=float(wall_time_s),
            peak_memory_gb=float(peak_memory_gb),
        )
    summary = _run_summary(
        name=run_name,
        semantics=semantics,
        run_dir=run_dir,
        wall_time_s=wall_time_s,
        error=error,
    )
    _write_json(root / "runs" / f"{run_name}.json", summary)

    flat_rows = [_flatten_for_frontier(summary)]
    frontier_quality_speed = pareto_frontier(
        flat_rows,
        objectives=(("val_loss", "min"), ("tokens_per_sec", "max")),
    )
    frontier_quality_speed_mem = pareto_frontier(
        flat_rows,
        objectives=(
            ("val_loss", "min"),
            ("tokens_per_sec", "max"),
            ("max_memory_gb", "min"),
        ),
    )
    report = {
        "kind": "pretrain_machine_validation",
        "profile": str(profile.name),
        "target_tokens_per_update": int(base_run_config.target_tokens_per_update),
        "release_semantics": dict(semantics),
        "summary": {
            "run_count": 1,
            "ok_count": int(str(summary.get("status", "")) == "ok"),
            "wall_time_s": float(time.perf_counter() - started_at),
        },
        "validation_total_tokens": int(base_run_config.total_tokens),
        "machine_baseline_table": [
            _machine_baseline_row(
                summary=summary,
                run_config=base_run_config,
            )
        ],
        "runs": [summary],
        "frontiers": {
            "quality_speed": frontier_quality_speed,
            "quality_speed_memory": frontier_quality_speed_mem,
        },
    }
    _write_json(root / "report.json", report)
    if str(summary.get("status", "") or "") == "ok" and str(run_name) == "release_pretrain":
        try:
            recipe_payload = _recipe_payload(
                run_config=base_run_config,
                run_dir=run_dir,
            )
        except Exception as exc:
            print(
                f"[WARN] unable to export {PRETRAIN_SIGNED_MACHINE_RECIPE_ARTIFACT}: {type(exc).__name__}: {exc}",
                flush=True,
            )
        else:
            _write_json(root / PRETRAIN_SIGNED_MACHINE_RECIPE_ARTIFACT, recipe_payload)
    print(
        "[MACHINE][PRETRAIN] completed | "
        f"status={str(summary.get('status', '') or '')} "
        f"report={root / 'report.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
