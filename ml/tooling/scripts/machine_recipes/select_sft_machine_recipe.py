#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml.errors import SophiaUsageError
from ml.tooling.core.pareto import pareto_frontier
from ml.training.posttrain.curriculum import default_posttrain_curriculum_stage
from ml.training.posttrain.defaults import (
    DEFAULT_POSTTRAIN_CURRICULUM_PATH,
    MACHINE_SELECTION_SFT_EVAL_STEPS,
    MACHINE_SELECTION_SFT_MAX_STEPS,
    MACHINE_SELECTION_SFT_REPORT_MAX_EXAMPLES_PER_SPLIT,
    MACHINE_SELECTION_SFT_REPORT_PROGRESS_EVERY,
)
from ml.training.posttrain.release_config import (
    RELEASE_SFT_DEFAULTS,
    SftReleaseSemantics,
    build_fixed_sft_release_semantics,
)
from ml.training.posttrain.types import PosttrainStageArgs


@dataclass(frozen=True)
class MachineCandidate:
    batch_size: int
    accumulation_steps: int
    gradient_checkpointing: bool

    def machine_selection_payload(self) -> dict[str, int]:
        return {
            "batch_size": int(self.batch_size),
            "accumulation_steps": int(self.accumulation_steps),
            "gradient_checkpointing": int(bool(self.gradient_checkpointing)),
        }


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    out: list[int] = []
    seen: set[int] = set()
    for part in str(raw or "").split(","):
        token = str(part).strip()
        if not token:
            continue
        value = int(token)
        if value in seen:
            continue
        out.append(int(value))
        seen.add(int(value))
    if not out:
        raise SophiaUsageError("expected at least one int value")
    return tuple(out)


def _parse_bool_csv(raw: str) -> tuple[bool, ...]:
    out: list[bool] = []
    seen: set[int] = set()
    for part in str(raw or "").split(","):
        token = str(part).strip().lower()
        if not token:
            continue
        if token in ("1", "true", "t", "yes", "y", "on"):
            value = True
        elif token in ("0", "false", "f", "no", "n", "off"):
            value = False
        else:
            raise SophiaUsageError(f"invalid bool token: {part!r}")
        key = int(bool(value))
        if key in seen:
            continue
        out.append(bool(value))
        seen.add(key)
    if not out:
        raise SophiaUsageError("expected at least one bool value")
    return tuple(out)


def _machine_candidates(
    *,
    target_examples_per_update: int,
    micro_batch_candidates: tuple[int, ...],
    checkpoint_candidates: tuple[bool, ...],
) -> list[MachineCandidate]:
    out: list[MachineCandidate] = []
    seen: set[tuple[int, int, int]] = set()
    target = max(int(target_examples_per_update), 1)
    for batch_size in micro_batch_candidates:
        batch = max(int(batch_size), 1)
        if target % batch != 0:
            continue
        accumulation = target // batch
        for checkpoint in checkpoint_candidates:
            key = (int(batch), int(accumulation), int(bool(checkpoint)))
            if key in seen:
                continue
            out.append(
                MachineCandidate(
                    batch_size=int(batch),
                    accumulation_steps=int(accumulation),
                    gradient_checkpointing=bool(checkpoint),
                )
            )
            seen.add(key)
    out.sort(
        key=lambda item: (
            int(not bool(item.gradient_checkpointing)),
            int(item.batch_size),
            int(item.accumulation_steps),
        )
    )
    if not out:
        raise SophiaUsageError(
            "no SFT machine candidates remain after divisibility filtering; "
            f"target_examples_per_update={target}"
        )
    return out


def _json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _metrics_summary(path: Path) -> dict[str, Any]:
    start_time: float | None = None
    end_time: float | None = None
    last_train: dict[str, Any] = {}
    last_eval_loss: float | None = None
    best_eval_loss: float | None = None
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
            if str(obj.get("stage", "") or "") != "sft":
                continue
            last_train = dict(obj)
            eval_loss = obj.get("eval_loss")
            if isinstance(eval_loss, (int, float)):
                value = float(eval_loss)
                last_eval_loss = value
                if best_eval_loss is None or value < best_eval_loss:
                    best_eval_loss = value
    out: dict[str, Any] = {}
    if last_train:
        for key in (
            "loss",
            "eval_loss",
            "test_loss",
            "lr",
            "supervised_tokens",
            "train_tokens_per_s",
            "wall_tokens_per_s",
            "train_time_s",
            "wall_time_s",
        ):
            value = last_train.get(key)
            if isinstance(value, (int, float)):
                out[key] = float(value)
    if last_eval_loss is not None:
        out["last_eval_loss"] = float(last_eval_loss)
    if best_eval_loss is not None:
        out["best_eval_loss"] = float(best_eval_loss)
    if start_time is not None and end_time is not None and end_time >= start_time:
        out["train_wall_time_s"] = float(end_time - start_time)
    return out


def _report_summary(path: Path) -> dict[str, Any]:
    payload = _json_or_none(path) or {}
    splits = payload.get("splits")
    splits = splits if isinstance(splits, dict) else {}
    out: dict[str, Any] = {}
    for split_name in ("val", "test"):
        split_payload = splits.get(split_name)
        split_payload = split_payload if isinstance(split_payload, dict) else {}
        overall = split_payload.get("overall")
        overall = overall if isinstance(overall, dict) else {}
        loss_mean = overall.get("loss_mean")
        rows = overall.get("rows")
        tokens = overall.get("supervised_tokens")
        if isinstance(loss_mean, (int, float)):
            out[f"{split_name}_loss_mean"] = float(loss_mean)
        if isinstance(rows, (int, float)):
            out[f"{split_name}_rows"] = int(rows)
        if isinstance(tokens, (int, float)):
            out[f"{split_name}_supervised_tokens"] = int(tokens)
    return out


def _run_command(*, argv: list[str], cwd: Path, timeout_s: float) -> None:
    timeout = None if float(timeout_s) <= 0.0 else float(timeout_s)
    subprocess.run(argv, cwd=str(cwd), check=True, timeout=timeout)


def _machine_selected_summary(
    *,
    run_dir: Path,
    machine: MachineCandidate,
) -> dict[str, int]:
    payload = _json_or_none(run_dir / "run_args.json") or {}
    if payload:
        return PosttrainStageArgs.machine_selection_payload_from_mapping(payload)
    return machine.machine_selection_payload()


def _run_summary(
    *,
    semantics: SftReleaseSemantics,
    machine: MachineCandidate,
    run_dir: Path,
    wall_time_s: float,
    error: str = "",
) -> dict[str, Any]:
    metrics = _metrics_summary(run_dir / "metrics.jsonl")
    report = _report_summary(run_dir / "sft_capability_report.json")
    summary = {
        "name": str(semantics.name),
        "status": "ok" if not error else "error",
        "error": str(error),
        "output_dir": str(run_dir.resolve()),
        "wall_time_s": float(wall_time_s),
        "release_semantics": semantics.to_payload(include_metadata=True),
        "machine_candidate": asdict(machine),
        "machine_selected": _machine_selected_summary(run_dir=run_dir, machine=machine),
        "metrics": metrics,
        "report": report,
    }
    if not error:
        loss_value = report.get("val_loss_mean", metrics.get("last_eval_loss"))
        speed_value = metrics.get("wall_tokens_per_s")
        if not (
            isinstance(loss_value, (int, float))
            and math.isfinite(float(loss_value))
            and isinstance(speed_value, (int, float))
            and math.isfinite(float(speed_value))
        ):
            summary["status"] = "incomplete"
            summary["error"] = "missing_or_non_finite_objective_metrics"
    return summary


def _flatten_for_frontier(summary: dict[str, Any]) -> dict[str, Any]:
    report = summary.get("report")
    report = report if isinstance(report, dict) else {}
    metrics = summary.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    semantics = summary.get("release_semantics")
    semantics = semantics if isinstance(semantics, dict) else {}
    machine = summary.get("machine_selected")
    machine = machine if isinstance(machine, dict) else {}
    return {
        "name": str(summary.get("name", "") or ""),
        "status": str(summary.get("status", "") or ""),
        "val_loss_mean": report.get("val_loss_mean", metrics.get("last_eval_loss")),
        "test_loss_mean": report.get("test_loss_mean"),
        "best_eval_loss": metrics.get("best_eval_loss"),
        "wall_tokens_per_s": metrics.get("wall_tokens_per_s"),
        "train_tokens_per_s": metrics.get("train_tokens_per_s"),
        "target_examples_per_update": semantics.get("target_examples_per_update"),
        "batch_size": machine.get("batch_size"),
        "accumulation_steps": machine.get("accumulation_steps"),
        "gradient_checkpointing": machine.get("gradient_checkpointing"),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _machine_baseline_row(summary: dict[str, Any]) -> dict[str, Any]:
    report = summary.get("report")
    report = report if isinstance(report, dict) else {}
    metrics = summary.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    machine = summary.get("machine_selected")
    machine = machine if isinstance(machine, dict) else {}
    return {
        "name": str(summary.get("name", "") or ""),
        "batch_size": int(machine.get("batch_size", 0) or 0),
        "accumulation_steps": int(machine.get("accumulation_steps", 0) or 0),
        "gradient_checkpointing": int(machine.get("gradient_checkpointing", 0) or 0),
        "loss": report.get("val_loss_mean", metrics.get("last_eval_loss")),
        "val_test": report.get("test_loss_mean"),
        "speed": metrics.get("wall_tokens_per_s"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a machine recipe for the fixed SFT semantic recipe and report the observed Pareto frontier."
    )
    parser.add_argument("--export_dir", required=True, type=str)
    parser.add_argument("--train_data", required=True, type=str)
    parser.add_argument("--eval_data", required=True, type=str)
    parser.add_argument("--test_data", required=True, type=str)
    parser.add_argument("--output_dir", type=str, default="out/sft_machine_recipe")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite_output_dir", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--curriculum_path",
        type=str,
        default=str(DEFAULT_POSTTRAIN_CURRICULUM_PATH),
    )
    parser.add_argument(
        "--curriculum_stage",
        type=str,
        default=default_posttrain_curriculum_stage(stage_kind="sft"),
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=MACHINE_SELECTION_SFT_MAX_STEPS,
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=MACHINE_SELECTION_SFT_EVAL_STEPS,
    )
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--safe_serialization", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--report_max_examples_per_split",
        type=int,
        default=MACHINE_SELECTION_SFT_REPORT_MAX_EXAMPLES_PER_SPLIT,
    )
    parser.add_argument(
        "--report_progress_every",
        type=int,
        default=MACHINE_SELECTION_SFT_REPORT_PROGRESS_EVERY,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=RELEASE_SFT_DEFAULTS.learning_rate,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=RELEASE_SFT_DEFAULTS.weight_decay,
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--micro_batch_candidates", type=str, default="1,2,4,8")
    parser.add_argument("--checkpoint_candidates", type=str, default="false,true")
    parser.add_argument(
        "--machine_timeout_s",
        type=float,
        default=180.0,
        help="Per-machine-candidate timeout. 0 disables the timeout.",
    )
    return parser


@dataclass(frozen=True)
class SftMachineSelectionConfig:
    export_dir: str
    train_data: str
    eval_data: str
    test_data: str
    output_dir: str
    device: str
    seed: int
    overwrite_output_dir: bool
    curriculum_path: str
    curriculum_stage: str
    max_steps: int
    eval_steps: int
    eval_interval: int
    save_total_limit: int
    safe_serialization: int
    report_max_examples_per_split: int
    report_progress_every: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    min_lr_ratio: float
    max_grad_norm: float
    micro_batch_candidates: tuple[int, ...]
    checkpoint_candidates: tuple[bool, ...]
    machine_timeout_s: float

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> SftMachineSelectionConfig:
        return cls(
            export_dir=str(args.export_dir),
            train_data=str(args.train_data),
            eval_data=str(args.eval_data),
            test_data=str(args.test_data),
            output_dir=str(args.output_dir),
            device=str(args.device),
            seed=int(args.seed),
            overwrite_output_dir=bool(int(args.overwrite_output_dir)),
            curriculum_path=str(args.curriculum_path),
            curriculum_stage=str(args.curriculum_stage),
            max_steps=int(args.max_steps),
            eval_steps=int(args.eval_steps),
            eval_interval=int(args.eval_interval),
            save_total_limit=int(args.save_total_limit),
            safe_serialization=int(args.safe_serialization),
            report_max_examples_per_split=int(args.report_max_examples_per_split),
            report_progress_every=int(args.report_progress_every),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            warmup_ratio=float(args.warmup_ratio),
            min_lr_ratio=float(args.min_lr_ratio),
            max_grad_norm=float(args.max_grad_norm),
            micro_batch_candidates=_parse_int_csv(str(args.micro_batch_candidates)),
            checkpoint_candidates=_parse_bool_csv(str(args.checkpoint_candidates)),
            machine_timeout_s=float(args.machine_timeout_s),
        )

    @property
    def release_semantics(self) -> SftReleaseSemantics:
        return build_fixed_sft_release_semantics(
            curriculum_path=str(self.curriculum_path),
            curriculum_stage=str(self.curriculum_stage),
            learning_rate=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
            notes=(
                f"target_examples_per_update={int(RELEASE_SFT_DEFAULTS.target_examples_per_update)} "
                f"learning_rate={float(self.learning_rate):.3e} "
                f"weight_decay={float(self.weight_decay):.3f}"
            ),
        )

    def resolve_machine_candidates(self) -> list[MachineCandidate]:
        return _machine_candidates(
            target_examples_per_update=int(
                self.release_semantics.target_examples_per_update
            ),
            micro_batch_candidates=tuple(self.micro_batch_candidates),
            checkpoint_candidates=tuple(self.checkpoint_candidates),
        )

    def argv_for_machine(
        self,
        *,
        machine: MachineCandidate,
        run_dir: Path,
    ) -> list[str]:
        return [
            sys.executable,
            "-m",
            "ml.cli.sft",
            "--export_dir",
            str(self.export_dir),
            "--train_data",
            str(self.train_data),
            "--eval_data",
            str(self.eval_data),
            "--test_data",
            str(self.test_data),
            "--output_dir",
            str(run_dir),
            "--overwrite_output_dir",
            str(int(self.overwrite_output_dir)),
            "--allow_missing_machine_recipe",
            "1",
            "--device",
            str(self.device),
            "--seed",
            str(int(self.seed)),
            "--curriculum_path",
            str(self.curriculum_path),
            "--curriculum_stage",
            str(self.curriculum_stage),
            "--batch_size",
            str(int(machine.batch_size)),
            "--accumulation_steps",
            str(int(machine.accumulation_steps)),
            "--max_steps",
            str(int(self.max_steps)),
            "--learning_rate",
            str(float(self.learning_rate)),
            "--weight_decay",
            str(float(self.weight_decay)),
            "--warmup_ratio",
            str(float(self.warmup_ratio)),
            "--min_lr_ratio",
            str(float(self.min_lr_ratio)),
            "--max_grad_norm",
            str(float(self.max_grad_norm)),
            "--eval_interval",
            str(max(int(self.eval_interval), 1)),
            "--eval_steps",
            str(int(self.eval_steps)),
            "--save_interval",
            str(max(int(self.max_steps), 1)),
            "--save_total_limit",
            str(int(self.save_total_limit)),
            "--safe_serialization",
            str(int(self.safe_serialization)),
            "--gradient_checkpointing",
            str(int(bool(machine.gradient_checkpointing))),
            "--report_max_examples_per_split",
            str(int(self.report_max_examples_per_split)),
            "--report_progress_every",
            str(int(self.report_progress_every)),
        ]


def main() -> None:
    selection = SftMachineSelectionConfig.from_namespace(_build_parser().parse_args())
    root = Path(selection.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[5]

    semantics = selection.release_semantics
    machine_candidates = selection.resolve_machine_candidates()

    runs: list[dict[str, Any]] = []
    best_summary: dict[str, Any] | None = None
    best_speed = float("-inf")
    selection_started_at = time.perf_counter()
    print(
        "[PARETO][SFT] semantics "
        f"name={semantics.name} "
        f"target_examples_per_update={int(semantics.target_examples_per_update)} "
        f"machine_candidates={len(machine_candidates)}",
        flush=True,
    )
    for machine_index, machine in enumerate(machine_candidates, start=1):
        run_dir = (
            root
            / semantics.name
            / (
                f"mb_{int(machine.batch_size)}"
                f"__acc_{int(machine.accumulation_steps)}"
                f"__ckpt_{int(bool(machine.gradient_checkpointing))}"
            )
        )
        argv = selection.argv_for_machine(machine=machine, run_dir=run_dir)
        print(
            "[PARETO][SFT] machine "
            f"{machine_index}/{len(machine_candidates)} | "
            f"batch={int(machine.batch_size)} "
            f"accum={int(machine.accumulation_steps)} "
            f"ckpt={int(bool(machine.gradient_checkpointing))}",
            flush=True,
        )
        t0 = time.perf_counter()
        error = ""
        try:
            _run_command(
                argv=argv,
                cwd=repo_root,
                timeout_s=float(selection.machine_timeout_s),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[PARETO][SFT] machine failed | {error}", flush=True)
        wall_time_s = float(time.perf_counter() - t0)
        summary = _run_summary(
            semantics=semantics,
            machine=machine,
            run_dir=run_dir,
            wall_time_s=wall_time_s,
            error=error,
        )
        runs.append(summary)
        _write_json(
            root
            / "runs"
            / (
                f"{semantics.name}__mb_{machine.batch_size}"
                f"__acc_{machine.accumulation_steps}"
                f"__ckpt_{int(bool(machine.gradient_checkpointing))}.json"
            ),
            summary,
        )
        if str(summary.get("status", "") or "") != "ok":
            continue
        metrics = summary.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        speed = metrics.get("wall_tokens_per_s")
        if isinstance(speed, (int, float)) and float(speed) > float(best_speed):
            best_speed = float(speed)
            best_summary = summary

    selected_runs = (
        [best_summary]
        if best_summary is not None
        else [
            {
                "name": str(semantics.name),
                "status": "error",
                "error": "no_successful_machine_candidate",
                "release_semantics": asdict(semantics),
            }
        ]
    )
    flat_rows = [_flatten_for_frontier(item) for item in runs]
    frontier_quality_speed = pareto_frontier(
        flat_rows,
        objectives=(("val_loss_mean", "min"), ("wall_tokens_per_s", "max")),
    )
    payload = {
        "kind": "sft_machine_selection",
        "started_at": float(selection_started_at),
        "wall_time_s": float(time.perf_counter() - selection_started_at),
        "release_semantics": asdict(semantics),
        "selected_runs": selected_runs,
        "all_machine_runs": runs,
        "machine_baseline_table": [
            _machine_baseline_row(item)
            for item in (
                list(selected_runs)
                + [
                    row
                    for row in runs
                    if str(row.get("status", "") or "") == "ok"
                    and row not in list(selected_runs)
                ]
            )
            if str(item.get("status", "") or "") == "ok"
        ],
        "frontier_quality_speed": frontier_quality_speed,
    }
    _write_json(root / "summary.json", payload)
    print(
        "[PARETO][SFT] completed | "
        f"successful={sum(1 for row in runs if str(row.get('status', '') or '') == 'ok')} "
        f"frontier={len(frontier_quality_speed)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
