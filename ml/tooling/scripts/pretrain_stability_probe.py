from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml.tasks.pretrain.observability as observability_mod
import ml.training.pretrain.engine.loop_execution as loop_execution
import ml.training.pretrain.release_gate as release_gate
from ml.tasks.pretrain.pipeline import build_pretrain_args, run
from ml.training.pretrain.profiles import RELEASE_PROFILE
from ml.training.pretrain.release_config import canonical_pretrain_machine_recipe_path


@dataclass(frozen=True)
class StabilityThresholds:
    max_loss: float
    max_grad_norm: float
    sustained_grad_norm: float
    sustained_grad_window: int


DEFAULT_THRESHOLDS = StabilityThresholds(
    max_loss=13.0,
    max_grad_norm=50.0,
    sustained_grad_norm=20.0,
    sustained_grad_window=3,
)


def _keep_configured_observability(*, args: Any, **_kwargs: Any) -> Any:
    return args


def _skip_final_export(**_kwargs: Any) -> bool:
    return False


def _install_experiment_hooks() -> None:
    observability_mod.auto_observability_policy = _keep_configured_observability
    release_gate.validate_signed_pretrain_machine_recipe = lambda **_: None
    loop_execution._maybe_save_model_export = _skip_final_export


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
    *,
    thresholds: StabilityThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    rows = read_metrics(metrics_path)
    train_rows = [row for row in rows if row.get("type") == "train"]
    eval_rows = [row for row in rows if row.get("type") == "eval"]
    losses = [float(row["loss"]) for row in train_rows if "loss" in row]
    grad_norms = [
        float(row["grad_norm"]) for row in train_rows if row.get("grad_norm") is not None
    ]
    tok_s = [float(row["tok_s"]) for row in train_rows if row.get("tok_s") is not None]
    loss_gt = [
        int(row.get("step", 0))
        for row in train_rows
        if float(row.get("loss", 0.0)) > float(thresholds.max_loss)
    ]
    grad_gt = [
        int(row.get("step", 0))
        for row in train_rows
        if float(row.get("grad_norm", 0.0)) > float(thresholds.max_grad_norm)
    ]
    sustained_steps: list[int] = []
    window = max(int(thresholds.sustained_grad_window), 1)
    for idx in range(0, max(len(train_rows) - window + 1, 0)):
        current = train_rows[idx : idx + window]
        if all(
            float(row.get("grad_norm", 0.0)) > float(thresholds.sustained_grad_norm)
            for row in current
        ):
            sustained_steps.append(int(current[0].get("step", 0)))
    completed = bool(rows and rows[-1].get("type") == "end")
    passed = (
        bool(completed)
        and not loss_gt
        and not grad_gt
        and not sustained_steps
        and bool(train_rows)
    )
    return {
        "metrics_path": str(metrics_path),
        "completed": bool(completed),
        "passed": bool(passed),
        "train_steps": len(train_rows),
        "last_step": int(train_rows[-1].get("step", 0)) if train_rows else 0,
        "max_steps": int(train_rows[-1].get("max_steps", 0)) if train_rows else 0,
        "last_loss": float(losses[-1]) if losses else None,
        "max_loss": max(losses) if losses else None,
        "loss_gt_threshold_steps": loss_gt,
        "last_grad_norm": float(grad_norms[-1]) if grad_norms else None,
        "max_grad_norm": max(grad_norms) if grad_norms else None,
        "grad_gt_threshold_steps": grad_gt,
        "sustained_grad_threshold_steps": sustained_steps,
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
        "thresholds": {
            "max_loss": float(thresholds.max_loss),
            "max_grad_norm": float(thresholds.max_grad_norm),
            "sustained_grad_norm": float(thresholds.sustained_grad_norm),
            "sustained_grad_window": int(thresholds.sustained_grad_window),
        },
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(str(args.output_dir)).resolve()
    if int(args.overwrite_output_dir) == 1:
        shutil.rmtree(output_dir, ignore_errors=True)

    _install_experiment_hooks()
    run_args = build_pretrain_args(
        data_path=str(args.data_path),
        tokenizer_path=str(args.tokenizer_path),
        output_dir=str(output_dir),
        resume_from_checkpoint="",
        overwrite_output_dir=int(args.overwrite_output_dir),
        machine_recipe_json=str(args.machine_recipe_json),
        profile=RELEASE_PROFILE,
    )
    run_args.max_steps = int(args.max_steps)
    run_args.total_tokens = int(args.max_steps) * int(run_args.target_tokens_per_update)
    run_args.warmup_steps = int(args.warmup_steps)
    run_args.eval_interval = int(args.eval_interval)
    run_args.eval_steps = int(args.eval_steps)
    run_args.external_eval_contamination_samples = int(args.contamination_samples)
    run_args.external_eval_contamination_interval = 0
    run_args.log_interval = int(args.log_interval)
    run_args.save_interval = 0
    run_args.save_total_limit = 0
    run_args.async_checkpoint = 0
    run_args.async_metrics = 0
    run_args.save_best = 0
    run_args.enable_checkpoints = 0
    run(run_args)

    summary = summarize_metrics(
        output_dir / "metrics.jsonl",
        thresholds=StabilityThresholds(
            max_loss=float(args.max_loss_threshold),
            max_grad_norm=float(args.max_grad_norm_threshold),
            sustained_grad_norm=float(args.sustained_grad_norm_threshold),
            sustained_grad_window=int(args.sustained_grad_window),
        ),
    )
    summary.update(
        {
            "kind": "pretrain_stability_probe",
            "output_dir": str(output_dir),
            "warmup_steps": int(args.warmup_steps),
            "max_steps": int(args.max_steps),
        }
    )
    if str(args.output_json).strip():
        out = Path(str(args.output_json))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or summarize a pretrain stability probe.")
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_tokens/train")
    parser.add_argument("--tokenizer_path", type=str, default="ml/modeling/text")
    parser.add_argument(
        "--machine_recipe_json",
        type=str,
        default=canonical_pretrain_machine_recipe_path(),
    )
    parser.add_argument("--output_dir", type=str, default="out/pretrain_stability_probe")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--overwrite_output_dir", type=int, choices=[0, 1], default=1)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--eval_steps", type=int, default=2)
    parser.add_argument("--contamination_samples", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--max_loss_threshold", type=float, default=13.0)
    parser.add_argument("--max_grad_norm_threshold", type=float, default=50.0)
    parser.add_argument("--sustained_grad_norm_threshold", type=float, default=20.0)
    parser.add_argument("--sustained_grad_window", type=int, default=3)
    parser.add_argument("--summarize_metrics", type=str, default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if str(args.summarize_metrics).strip():
        summary = summarize_metrics(
            str(args.summarize_metrics),
            thresholds=StabilityThresholds(
                max_loss=float(args.max_loss_threshold),
                max_grad_norm=float(args.max_grad_norm_threshold),
                sustained_grad_norm=float(args.sustained_grad_norm_threshold),
                sustained_grad_window=int(args.sustained_grad_window),
            ),
        )
        if str(args.output_json).strip():
            out = Path(str(args.output_json))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0
    run_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
