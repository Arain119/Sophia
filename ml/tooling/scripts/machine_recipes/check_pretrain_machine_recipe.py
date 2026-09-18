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
import ml.training.pretrain.engine.loop_execution as loop_execution_mod

from ml.training.pretrain.run_config import PretrainRunConfig
from ml.training.pretrain.release_gate import (
    PRETRAIN_MACHINE_RECIPE_KIND,
    build_pretrain_data_signature,
)
from ml.training.pretrain.release_config import (
    DEFAULT_PRETRAIN_MACHINE_RECIPE,
    DEFAULT_PRETRAIN_MACHINE_RUNTIME,
    RELEASE_PRETRAIN_DEFAULTS,
    PretrainReleaseSemantics,
)
from ml.training.pretrain.implementation_fingerprint import (
    current_pretrain_implementation_sha256,
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

PRETRAIN_MACHINE_WARMUP_UPDATES = 1
PRETRAIN_MACHINE_STEADY_UPDATES = 2
PRETRAIN_MACHINE_MEASUREMENT_UPDATES = (
    PRETRAIN_MACHINE_WARMUP_UPDATES + PRETRAIN_MACHINE_STEADY_UPDATES
)
PRETRAIN_MACHINE_MEASUREMENT_TOKENS = (
    RELEASE_PRETRAIN_DEFAULTS.target_tokens_per_update
    * PRETRAIN_MACHINE_MEASUREMENT_UPDATES
)


def _json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _metrics_summary(path: Path) -> dict[str, Any]:
    best_val_loss: float | None = None
    last_val_loss: float | None = None
    test_loss: float | None = None
    contamination_safe_rate: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    train_events: list[tuple[float, int, float]] = []
    peak_train_memory_gb = 0.0
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
                if best_val_loss is None or val < best_val_loss:
                    best_val_loss = val
            if typ == "train":
                dt_s = obj.get("dt_s")
                updates = obj.get("updates")
                tok_s = obj.get("tok_s")
                if (
                    isinstance(dt_s, (int, float))
                    and math.isfinite(float(dt_s))
                    and float(dt_s) > 0.0
                    and isinstance(updates, int)
                    and int(updates) > 0
                    and isinstance(tok_s, (int, float))
                    and math.isfinite(float(tok_s))
                    and float(tok_s) > 0.0
                ):
                    train_events.append((float(dt_s), int(updates), float(tok_s)))
                    mem_gb = obj.get("mem_gb")
                    if (
                        isinstance(mem_gb, (int, float))
                        and math.isfinite(float(mem_gb))
                        and float(mem_gb) >= 0.0
                    ):
                        peak_train_memory_gb = max(
                            peak_train_memory_gb, float(mem_gb)
                        )
            if str(obj.get("name", "") or "") == "test_loss" and isinstance(
                obj.get("value"), (int, float)
            ):
                test_loss = float(obj["value"])
            if str(
                obj.get("name", "") or ""
            ) == "pretrain_contamination_safe_rate" and isinstance(
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
    out["train_events"] = len(train_events)
    out["peak_train_memory_gb"] = float(peak_train_memory_gb)
    if train_events:
        first_dt_s, first_updates, first_tok_s = train_events[0]
        out["first_update_wall_time_s"] = float(first_dt_s) / int(first_updates)
        out["first_update_tokens_per_sec"] = float(first_tok_s)
    steady_events = train_events[PRETRAIN_MACHINE_WARMUP_UPDATES:]
    if steady_events:
        steady_dt_s = sum(event[0] for event in steady_events)
        steady_updates = sum(event[1] for event in steady_events)
        out["steady_train_events"] = len(steady_events)
        out["steady_updates"] = int(steady_updates)
        out["steady_update_wall_time_s"] = float(steady_dt_s) / int(steady_updates)
        out["steady_tokens_per_sec"] = sum(
            event[0] * event[2] for event in steady_events
        ) / float(steady_dt_s)
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
    machine_recipe_summary = (
        _json_or_none(run_dir / PRETRAIN_MACHINE_RECIPE_SUMMARY) or {}
    )
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
            "gradient_checkpointing": int(
                selected.get("gradient_checkpointing", 0) or 0
            ),
            "loss_chunk_size": int(selected.get("loss_chunk_size", 0) or 0),
            "dataloader_num_workers": int(
                selected.get("dataloader_num_workers", 0) or 0
            ),
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
        raise RuntimeError(
            f"run_args.json is missing _sophia_machine_signature under {run_dir}"
        )
    machine_recipe_artifact = _json_or_none(run_dir / PRETRAIN_MACHINE_RECIPE_ARTIFACT)
    machine_runtime_artifact = _json_or_none(
        run_dir / PRETRAIN_MACHINE_RUNTIME_ARTIFACT
    )
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
        raise RuntimeError(
            f"missing {PRETRAIN_UPDATE_PROFILE_ARTIFACT} under {run_dir}"
        )
    bootstrap = PretrainRuntimeBootstrap(
        output_dir=str(run_dir),
        resume_path=None,
        resume_checkpoint=None,
        resolved_resume_checkpoint="",
        data_path=str(run_args.get("data_path", "") or ""),
        eval_data_path=str(run_args.get("eval_data_path", "") or ""),
        test_data_path=str(run_args.get("test_data_path", "") or ""),
        manifest_path=str(
            manifest_policy.resolve_manifest_path(
                str(run_args.get("data_path", "") or "")
            )
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
        "target_tokens_per_update": int(
            run_args.get("target_tokens_per_update", 0) or 0
        ),
        "learning_rate": float(run_args.get("learning_rate", 0.0) or 0.0),
        "weight_decay": float(run_args.get("weight_decay", 0.0) or 0.0),
        # The measured run disables warmup because it is only three updates;
        # the signed recipe must bind the formal schedule from run_config.
        "warmup_steps": int(run_config.warmup_steps),
        "warmup_ratio": float(run_args.get("warmup_ratio", 0.0) or 0.0),
        "min_lr_ratio": float(run_args.get("min_lr_ratio", 0.0) or 0.0),
        "lr_schedule": str(run_args.get("lr_schedule", "") or ""),
        "dataloader_num_workers": _int_from_mapping(
            run_args, "dataloader_num_workers", default=-1
        ),
        "dataloader_prefetch_factor": _int_from_mapping(
            run_args, "dataloader_prefetch_factor", default=-1
        ),
        "dataloader_persistent_workers": _int_from_mapping(
            run_args, "dataloader_persistent_workers", default=-1
        ),
        "shard_preload": _int_from_mapping(run_args, "shard_preload", default=-1),
        "shard_preload_bytes": _int_from_mapping(
            run_args, "shard_preload_bytes", default=-1
        ),
        "machine_recipe_json": "",
        "_sophia_run_kind": "pretrain",
        "_sophia_machine_signature": dict(machine_signature),
    }
    recipe_args = replace(run_config, **state)
    return {
        "kind": PRETRAIN_MACHINE_RECIPE_KIND,
        "stage": "pretrain",
        "pretrain_implementation_sha256": current_pretrain_implementation_sha256(),
        "selected_name": f"measured_pretrain_sophia_1b_seq{int(recipe_args.seq_len)}",
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
            "gradient_checkpointing": int(
                run_args.get("gradient_checkpointing", 0) or 0
            ),
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
                (
                    (
                        (machine_runtime_artifact.get("best") or {})
                        if isinstance(machine_runtime_artifact, dict)
                        else {}
                    ).get("step_execution_backend", "")
                )
                or (
                    (
                        run_args.get("_sophia_machine_runtime", {})
                        if isinstance(run_args.get("_sophia_machine_runtime"), dict)
                        else {}
                    ).get("step_execution_backend", "")
                )
                or DEFAULT_PRETRAIN_MACHINE_RUNTIME["step_execution_backend"]
            ),
            "dataloader_num_workers": _int_from_mapping(
                run_args, "dataloader_num_workers", default=-1
            ),
            "dataloader_prefetch_factor": _int_from_mapping(
                run_args, "dataloader_prefetch_factor", default=-1
            ),
            "dataloader_persistent_workers": _int_from_mapping(
                run_args, "dataloader_persistent_workers", default=-1
            ),
            "shard_preload": _int_from_mapping(run_args, "shard_preload", default=-1),
            "shard_preload_bytes": _int_from_mapping(
                run_args, "shard_preload_bytes", default=-1
            ),
        },
        "artifacts": {
            "machine_recipe": machine_recipe_artifact,
            "machine_runtime": machine_runtime_artifact,
            "update_profile": update_profile,
        },
    }


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
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)


def _materialize_validation_machine_artifacts(
    *,
    run_dir: Path,
) -> None:
    run_args = _json_or_none(run_dir / "run_args.json") or {}
    summary = _json_or_none(run_dir / PRETRAIN_MACHINE_RECIPE_SUMMARY) or {}
    selected_raw = summary.get("selected")
    selected = selected_raw if isinstance(selected_raw, dict) else {}
    if not selected and not run_args:
        return

    seq_len = _int_from_sources(key="seq_len", selected=selected, run_args=run_args)
    batch_size = _int_from_sources(
        key="batch_size", selected=selected, run_args=run_args
    )
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
    train_events = int(metrics.get("train_events", 0) or 0)
    steady_train_events = int(metrics.get("steady_train_events", 0) or 0)
    steady_updates = int(metrics.get("steady_updates", 0) or 0)
    if train_events != PRETRAIN_MACHINE_MEASUREMENT_UPDATES:
        raise RuntimeError(
            "machine validation must emit exactly "
            f"{PRETRAIN_MACHINE_MEASUREMENT_UPDATES} per-update train events; "
            f"got {train_events}"
        )
    if (
        steady_train_events != PRETRAIN_MACHINE_STEADY_UPDATES
        or steady_updates != PRETRAIN_MACHINE_STEADY_UPDATES
    ):
        raise RuntimeError(
            "machine validation steady-state window must contain exactly "
            f"{PRETRAIN_MACHINE_STEADY_UPDATES} updates"
        )
    cold_start_step_time_s = _positive_float(metrics.get("first_update_wall_time_s"))
    step_time_s = _positive_float(metrics.get("steady_update_wall_time_s"))
    if cold_start_step_time_s <= 0.0 or step_time_s <= 0.0:
        raise RuntimeError("machine validation emitted invalid update timings")
    cold_start_tokens_per_sec = float(tokens_per_update) / float(
        cold_start_step_time_s
    )
    tokens_per_sec = float(tokens_per_update) / float(step_time_s)

    max_memory_gb = _positive_float(metrics.get("peak_train_memory_gb"))
    backend = str(
        _str_from_sources(
            key="step_execution_backend",
            selected=selected,
            run_args=run_args,
            default="sm120_graph",
        )
        or "sm120_graph"
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
            "cold_start_tokens_per_sec": float(cold_start_tokens_per_sec),
            "cold_start_step_time_s": float(cold_start_step_time_s),
            "steady_train_events": int(steady_train_events),
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
            "cold_start_tokens_per_sec": float(cold_start_tokens_per_sec),
            "cold_start_step_time_s": float(cold_start_step_time_s),
            "steady_train_events": int(steady_train_events),
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
        "cold_start_tokens_per_sec": float(cold_start_tokens_per_sec),
        "cold_start_update_s": float(cold_start_step_time_s),
        "measurement_updates": int(PRETRAIN_MACHINE_MEASUREMENT_UPDATES),
        "steady_updates": int(PRETRAIN_MACHINE_STEADY_UPDATES),
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
        "batch_size": int(
            adaptive.get("batch_size", 0) or recipe.get("batch_size", 0) or 0
        ),
        "accumulation_steps": int(
            adaptive.get("accumulation_steps", 0)
            or recipe.get("accumulation_steps", 0)
            or 0
        ),
        "gradient_checkpointing": int(
            adaptive.get("gradient_checkpointing", 0)
            or bool(recipe.get("gradient_checkpointing", False))
        ),
        "loss_chunk_size": int(
            adaptive.get("loss_chunk_size", 0) or recipe.get("loss_chunk_size", 0) or 0
        ),
        "tokens_per_s": float(
            runtime.get("tokens_per_sec", 0.0)
            or recipe.get("tokens_per_sec", 0.0)
            or 0.0
        ),
        "step_time_s": float(
            runtime.get("step_time_s", 0.0) or recipe.get("step_time_s", 0.0) or 0.0
        ),
        "max_mem_gb": float(recipe.get("max_memory_gb", 0.0) or 0.0),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure and sign the fixed RTX 5090 pretraining runtime."
    )
    parser.add_argument("--data_path", type=str, default="dataset/pretrain")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out/pretrain_machine_recipe",
        help="Validation root; the fixed release run executes under this path.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


@dataclass(frozen=True)
class PretrainMachineValidationConfig:
    data_path: str
    tokenizer_path: str
    output_dir: str
    device: str

    @classmethod
    def from_namespace(
        cls, args: argparse.Namespace
    ) -> PretrainMachineValidationConfig:
        return cls(
            data_path=str(args.data_path),
            tokenizer_path=str(args.tokenizer_path or ""),
            output_dir=str(args.output_dir),
            device=str(args.device),
        )

    def build_release_run_config(self, *, profile) -> PretrainRunConfig:
        base_run_config = build_pretrain_args(
            data_path=str(self.data_path),
            tokenizer_path=str(self.tokenizer_path),
            output_dir="",
            overwrite_output_dir=0,
            profile=profile,
        )
        return replace(
            base_run_config,
            device=str(self.device),
            seed=42,
            async_checkpoint=0,
            async_metrics=0,
            save_total_limit=1,
            save_best=0,
            enable_checkpoints=0,
            log_interval=1,
            dataloader_num_workers=int(
                DEFAULT_PRETRAIN_MACHINE_RUNTIME["dataloader_num_workers"]
            ),
            dataloader_prefetch_factor=int(
                DEFAULT_PRETRAIN_MACHINE_RUNTIME["dataloader_prefetch_factor"]
            ),
            dataloader_persistent_workers=int(
                DEFAULT_PRETRAIN_MACHINE_RUNTIME[
                    "dataloader_persistent_workers"
                ]
            ),
            shard_preload=int(
                DEFAULT_PRETRAIN_MACHINE_RUNTIME["shard_preload"]
            ),
            shard_preload_bytes=int(
                DEFAULT_PRETRAIN_MACHINE_RUNTIME["shard_preload_bytes"]
            ),
            step_execution_backend=str(
                DEFAULT_PRETRAIN_MACHINE_RUNTIME["step_execution_backend"]
            ),
            _sophia_run_kind="pretrain_machine_validate",
            batch_size=int(DEFAULT_PRETRAIN_MACHINE_RECIPE["batch_size"]),
            accumulation_steps=int(
                DEFAULT_PRETRAIN_MACHINE_RECIPE["accumulation_steps"]
            ),
            gradient_checkpointing=int(
                DEFAULT_PRETRAIN_MACHINE_RECIPE["gradient_checkpointing"]
            ),
            gradient_checkpointing_exclude_first=int(
                DEFAULT_PRETRAIN_MACHINE_RECIPE[
                    "gradient_checkpointing_exclude_first"
                ]
            ),
            gradient_checkpointing_exclude_last=int(
                DEFAULT_PRETRAIN_MACHINE_RECIPE[
                    "gradient_checkpointing_exclude_last"
                ]
            ),
            loss_chunk_size=int(DEFAULT_PRETRAIN_MACHINE_RECIPE["loss_chunk_size"]),
            lr_schedule="cosine",
            total_tokens=PRETRAIN_MACHINE_MEASUREMENT_TOKENS,
        )

    def run_config(
        self,
        *,
        base_run_config: PretrainRunConfig,
        run_dir: Path,
    ) -> PretrainRunConfig:
        return replace(
            base_run_config,
            output_dir=run_dir.as_posix(),
            overwrite_output_dir=0,
            # The three-update measurement cannot contain the formal 153-step
            # warmup; keep the formal value on base_run_config for the recipe.
            warmup_steps=0,
        )

    def run_name(self, *, base_run_config: PretrainRunConfig) -> str:
        suffix = ""
        step_backend = (
            str(base_run_config.step_execution_backend or "sm120_graph")
            .strip()
            .lower()
        )
        if step_backend:
            suffix = f"{suffix}_{step_backend}"
        return f"release_pretrain{suffix}"


def main() -> None:
    config = PretrainMachineValidationConfig.from_namespace(
        _build_parser().parse_args()
    )
    seq_len = int(RELEASE_PROFILE.training.train_seq_len)
    measured_tokens_per_update = (
        int(DEFAULT_PRETRAIN_MACHINE_RECIPE["batch_size"])
        * int(DEFAULT_PRETRAIN_MACHINE_RECIPE["accumulation_steps"])
        * seq_len
    )
    if measured_tokens_per_update != int(
        RELEASE_PRETRAIN_DEFAULTS.target_tokens_per_update
    ):
        raise ValueError(
            "batch_size * accumulation_steps * seq_len must equal the fixed "
            f"{RELEASE_PRETRAIN_DEFAULTS.target_tokens_per_update} tokens/update"
        )
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
        loop_execution_mod._maybe_save_model_export = lambda **_kwargs: False
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
    if not error:
        _materialize_validation_machine_artifacts(
            run_dir=run_dir,
        )
    summary = _run_summary(
        name=run_name,
        semantics=semantics,
        run_dir=run_dir,
        wall_time_s=wall_time_s,
        error=error,
    )
    _write_json(root / "runs" / f"{run_name}.json", summary)

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
    }
    _write_json(root / "report.json", report)
    if str(summary.get("status", "") or "") == "ok":
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
