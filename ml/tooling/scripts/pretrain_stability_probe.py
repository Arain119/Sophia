from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

import ml.training.pretrain.engine.loop_execution as loop_execution
from ml.core.engine.checkpoint_paths import resolve_resume_checkpoint
from ml.tasks.pretrain.pipeline import build_pretrain_args, run
from ml.training.pretrain.profiles import RELEASE_PROFILE
from ml.training.pretrain.implementation_fingerprint import (
    current_pretrain_implementation_sha256,
)
from ml.training.pretrain.release_config import (
    RELEASE_PRETRAIN_DEFAULTS,
    RELEASE_PRETRAIN_STABILITY_STEPS,
)

MINIMUM_PROBE_STEPS = RELEASE_PRETRAIN_STABILITY_STEPS
MLA_LOGIT_METRICS_PER_UPDATE = 7 * 16


def _skip_final_export(**_kwargs: Any) -> bool:
    return False


def _install_probe_hooks() -> None:
    loop_execution._maybe_save_model_export = _skip_final_export


def _sha256(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing evaluation binding file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_binding(
    *,
    protocol_json: str | Path | None,
    checkpoint: str | Path | None,
    checkpoint_sha256: str | None,
    eval_export_model: str | Path | None,
) -> dict[str, str | None] | None:
    values = (protocol_json, checkpoint, checkpoint_sha256, eval_export_model)
    if not any(value is not None and str(value).strip() for value in values):
        return None
    if eval_export_model is not None and str(eval_export_model).strip():
        raise ValueError("eval_export_model is not applicable to stability reports")
    if not all(value is not None and str(value).strip() for value in (protocol_json, checkpoint)):
        raise ValueError(
            "release-bound stability evaluation requires --protocol_json and --checkpoint"
        )
    protocol_path = Path(str(protocol_json)).expanduser().resolve()
    checkpoint_path = Path(str(checkpoint)).expanduser().resolve()
    protocol_digest = _sha256(protocol_path)
    checkpoint_digest = _sha256(checkpoint_path)
    if checkpoint_sha256 and str(checkpoint_sha256).lower() != checkpoint_digest:
        raise ValueError("checkpoint_sha256 does not match checkpoint")
    return {
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "eval_export_model_path": None,
        "eval_export_model_sha256": None,
        "evaluation_asset_sha256": None,
    }


def read_metrics(metrics_path: str | Path) -> list[dict[str, Any]]:
    path = Path(metrics_path)
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(dict(item))
    return rows


def summarize_metrics(
    metrics_path: str | Path,
) -> dict[str, Any]:
    rows = read_metrics(metrics_path)
    train_rows = [row for row in rows if row.get("type") == "train"]
    eval_rows = [row for row in rows if row.get("type") == "eval"]
    attention_rows = [
        row
        for row in rows
        if row.get("type") == "metric"
        and str(row.get("name", "")).startswith("attention/mla_layer_")
    ]
    attention_counts: dict[int, int] = {}
    nonfinite_attention_logit_steps: set[int] = set()
    for row in attention_rows:
        step = int(row.get("step", 0))
        attention_counts[step] = attention_counts.get(step, 0) + 1
        if not math.isfinite(float(row.get("value", math.nan))):
            nonfinite_attention_logit_steps.add(step)
    train_steps = {int(row.get("step", 0)) for row in train_rows}
    attention_logit_telemetry_complete = bool(train_steps) and (
        set(attention_counts) == train_steps
        and all(
            count == MLA_LOGIT_METRICS_PER_UPDATE
            for count in attention_counts.values()
        )
        and not nonfinite_attention_logit_steps
    )
    losses = [float(row["loss"]) for row in train_rows if "loss" in row]
    grad_norms = [
        float(row["grad_norm"])
        for row in train_rows
        if row.get("grad_norm") is not None
    ]
    tok_s = [float(row["tok_s"]) for row in train_rows if row.get("tok_s") is not None]
    loss_window = min(len(losses), int(RELEASE_PRETRAIN_DEFAULTS.warmup_steps))
    nonfinite_loss_steps = [
        int(row.get("step", 0))
        for row in train_rows
        if not math.isfinite(float(row.get("loss", math.nan)))
    ]
    nonfinite_grad_steps = [
        int(row.get("step", 0))
        for row in train_rows
        if row.get("grad_norm") is None
        or not math.isfinite(float(row.get("grad_norm", math.nan)))
    ]
    completed = bool(rows and rows[-1].get("type") == "end")
    passed = (
        bool(completed)
        and bool(train_rows)
        and not nonfinite_loss_steps
        and not nonfinite_grad_steps
    )
    return {
        "metrics_path": str(metrics_path),
        "completed": bool(completed),
        "passed": bool(passed),
        "train_steps": len(train_rows),
        "first_step": int(train_rows[0].get("step", 0)) if train_rows else 0,
        "last_step": int(train_rows[-1].get("step", 0)) if train_rows else 0,
        "max_steps": int(train_rows[-1].get("max_steps", 0)) if train_rows else 0,
        "last_loss": float(losses[-1]) if losses else None,
        "max_loss": max(losses) if losses else None,
        "first_window_mean_loss": (
            sum(losses[:loss_window]) / loss_window if loss_window else None
        ),
        "last_window_mean_loss": (
            sum(losses[-loss_window:]) / loss_window if loss_window else None
        ),
        "loss_improvement": (
            (sum(losses[:loss_window]) - sum(losses[-loss_window:])) / loss_window
            if loss_window
            else None
        ),
        "loss_window_steps": int(loss_window),
        "nonfinite_loss_steps": nonfinite_loss_steps,
        "last_grad_norm": float(grad_norms[-1]) if grad_norms else None,
        "observed_max_grad_norm": max(grad_norms) if grad_norms else None,
        "nonfinite_grad_norm_steps": nonfinite_grad_steps,
        "avg_tok_s": (sum(tok_s) / len(tok_s)) if tok_s else None,
        "avg_tok_s_after_20": (
            sum(tok_s[20:]) / len(tok_s[20:]) if len(tok_s) > 20 else None
        ),
        "eval_count": len(eval_rows),
        "last_val_loss": (
            float(eval_rows[-1]["val_loss"])
            if eval_rows and "val_loss" in eval_rows[-1]
            else None
        ),
        "attention_logit_metric_count": len(attention_rows),
        "attention_logit_telemetry_complete": attention_logit_telemetry_complete,
        "nonfinite_attention_logit_steps": sorted(nonfinite_attention_logit_steps),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(str(args.output_dir)).resolve()
    graph_recapture_probe = bool(int(args.graph_recapture_probe))
    if int(args.overwrite_output_dir) == 1:
        raise ValueError(
            "overwrite_output_dir is forbidden for the formal stability probe; "
            "use a fresh output directory"
        )
    if int(args.max_steps) < int(MINIMUM_PROBE_STEPS) and not graph_recapture_probe:
        raise ValueError(
            "formal target Muon stability probe must reach the first "
            f"post-warmup update ({MINIMUM_PROBE_STEPS} steps)"
        )
    if graph_recapture_probe and not (
        int(args.eval_interval) > 0
        and int(args.max_steps) > int(args.eval_interval)
    ):
        raise ValueError(
            "graph recapture probe requires 0 < eval_interval < max_steps"
        )
    recipe_path = Path(str(args.machine_recipe_json)).expanduser().resolve()
    if not recipe_path.is_file():
        raise FileNotFoundError(f"missing signed machine recipe: {recipe_path}")
    resolved_resume = resolve_resume_checkpoint(
        str(args.resume_from_checkpoint or ""), output_dir=str(output_dir)
    )
    resume_checkpoint_path = (
        None if resolved_resume is None else Path(resolved_resume).resolve()
    )
    resume_checkpoint_sha256 = (
        "" if resume_checkpoint_path is None else _sha256(resume_checkpoint_path)
    )

    _install_probe_hooks()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    run_args = build_pretrain_args(
        data_path=str(args.data_path),
        tokenizer_path=str(args.tokenizer_path),
        output_dir=str(output_dir),
        resume_from_checkpoint=str(args.resume_from_checkpoint or ""),
        overwrite_output_dir=int(args.overwrite_output_dir),
        machine_recipe_json=str(args.machine_recipe_json),
        profile=RELEASE_PROFILE,
    )
    release_total_tokens = int(run_args.total_tokens)
    schedule_horizon_steps = int(
        math.ceil(release_total_tokens / int(run_args.target_tokens_per_update))
    )
    if schedule_horizon_steps < int(args.max_steps):
        raise ValueError(
            "formal LR schedule horizon cannot end before the stability probe: "
            f"horizon={schedule_horizon_steps} probe_steps={int(args.max_steps)}"
        )
    run_args.max_steps = int(args.max_steps)
    run_args.total_tokens = int(args.max_steps) * int(run_args.target_tokens_per_update)
    run_args._sophia_resume_recipe_override_fields = (
        "max_steps",
        "total_tokens",
        "eval_interval",
        "eval_steps",
        "log_interval",
        "save_interval",
        "save_total_limit",
        "async_checkpoint",
        "async_metrics",
        "save_best",
        "enable_checkpoints",
    )
    run_args._sophia_lr_schedule_horizon_steps = int(schedule_horizon_steps)
    run_args.eval_interval = int(args.eval_interval)
    run_args.eval_steps = int(args.eval_steps)
    run_args.external_eval_contamination_samples = int(args.contamination_samples)
    run_args.external_eval_contamination_interval = 0
    run_args.log_interval = int(args.log_interval)
    run_args.save_interval = int(args.checkpoint_interval)
    run_args.save_total_limit = int(args.checkpoint_keep)
    run_args.async_checkpoint = 0
    run_args.async_metrics = 0
    run_args.save_best = 0
    run_args.enable_checkpoints = 1
    runtime_evidence = dict(run(run_args) or {})

    summary = summarize_metrics(output_dir / "metrics.jsonl")
    graph_recapture_covered = bool(
        graph_recapture_probe
        and int(summary.get("eval_count", 0) or 0) > 0
        and int(summary.get("last_step", 0) or 0) > int(args.eval_interval)
    )
    attention_logit_telemetry_expected = bool(
        runtime_evidence.get("mla_logit_telemetry", False)
    )
    summary.update(
        {
            "kind": (
                "pretrain_graph_recapture_probe"
                if graph_recapture_probe
                else "pretrain_stability_probe"
            ),
            "status": "pass" if bool(summary.get("passed")) else "fail",
            "output_dir": str(output_dir),
            "warmup_steps": int(run_args.warmup_steps),
            "max_steps": int(args.max_steps),
            "minimum_required_steps": int(MINIMUM_PROBE_STEPS),
            "graph_recapture_covered": graph_recapture_covered,
            "attention_logit_telemetry_expected": (
                attention_logit_telemetry_expected
            ),
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None
            ),
            "cuda_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else None
            ),
            "graph_recapture_peak_allocated_bytes": runtime_evidence.get(
                "recapture_peak_allocated_bytes"
            ),
            "graph_recapture_peak_reserved_bytes": runtime_evidence.get(
                "recapture_peak_reserved_bytes"
            ),
            "lr_schedule_horizon_steps": int(schedule_horizon_steps),
            "machine_recipe_path": str(recipe_path),
            "machine_recipe_sha256": _sha256(recipe_path),
            "pretrain_implementation_sha256": current_pretrain_implementation_sha256(),
            "resume_from_checkpoint": str(args.resume_from_checkpoint or ""),
            "resume_checkpoint_path": (
                "" if resume_checkpoint_path is None else str(resume_checkpoint_path)
            ),
            "resume_checkpoint_sha256": resume_checkpoint_sha256,
            "checkpoint_interval": int(args.checkpoint_interval),
            "checkpoint_keep": int(args.checkpoint_keep),
            "evaluation_binding": _evaluation_binding(
                protocol_json=args.protocol_json,
                checkpoint=args.checkpoint,
                checkpoint_sha256=args.checkpoint_sha256,
                eval_export_model=args.eval_export_model,
            ),
        }
    )
    if attention_logit_telemetry_expected and not bool(
        summary["attention_logit_telemetry_complete"]
    ):
        summary["passed"] = False
        summary["status"] = "fail"
        summary["failure_reason"] = "attention_logit_telemetry_not_covered"
    elif graph_recapture_probe and (
        not graph_recapture_covered
        or summary["graph_recapture_peak_allocated_bytes"] is None
        or summary["graph_recapture_peak_reserved_bytes"] is None
    ):
        summary["passed"] = False
        summary["status"] = "fail"
        summary["failure_reason"] = "eval_recapture_memory_not_covered"
    elif (
        not graph_recapture_probe
        and int(summary.get("last_step", 0) or 0) < int(MINIMUM_PROBE_STEPS)
    ):
        summary["passed"] = False
        summary["status"] = "fail"
        summary["failure_reason"] = "probe_did_not_reach_post_warmup_update"
    if str(args.output_json).strip():
        out = Path(str(args.output_json))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or summarize a pretrain stability probe."
    )
    parser.add_argument("--data_path", type=str, default="dataset/pretrain")
    parser.add_argument("--tokenizer_path", type=str, default="ml/modeling/text")
    parser.add_argument(
        "--machine_recipe_json",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir", type=str, default="out/pretrain_stability_probe"
    )
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--overwrite_output_dir", type=int, choices=[0, 1], default=0)
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--max_steps", type=int, default=MINIMUM_PROBE_STEPS)
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=RELEASE_PRETRAIN_DEFAULTS.warmup_steps,
    )
    parser.add_argument("--checkpoint_keep", type=int, default=3)
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--eval_steps", type=int, default=2)
    parser.add_argument("--contamination_samples", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--graph_recapture_probe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--summarize_metrics", type=str, default="")
    parser.add_argument("--protocol_json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint_sha256")
    parser.add_argument("--eval_export_model")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if str(args.summarize_metrics).strip():
        summary = summarize_metrics(str(args.summarize_metrics))
        # A summary the release gate can read has to carry the same verdict
        # fields the probe stamps on a live run. Without them the summarised
        # report is missing exactly the keys the gate checks, so this mode
        # could never serve the purpose it exists for.
        summary["kind"] = "pretrain_stability_probe"
        summary["minimum_required_steps"] = int(MINIMUM_PROBE_STEPS)
        if not bool(summary.get("attention_logit_telemetry_complete")):
            summary["passed"] = False
            summary["failure_reason"] = "attention_logit_telemetry_not_covered"
        elif int(summary.get("last_step", 0) or 0) < int(MINIMUM_PROBE_STEPS):
            summary["passed"] = False
            summary["failure_reason"] = "probe_did_not_reach_post_warmup_update"
        summary["status"] = "pass" if bool(summary.get("passed")) else "fail"
        summary["evaluation_binding"] = _evaluation_binding(
            protocol_json=args.protocol_json,
            checkpoint=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            eval_export_model=args.eval_export_model,
        )
        if str(args.output_json).strip():
            out = Path(str(args.output_json))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0
    result = run_probe(args)
    return 0 if bool(result.get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
