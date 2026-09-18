"""Run one bounded GRPO correction pass on judged rollout groups."""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ml.integrations.export.artifacts import export_model_artifacts
from ml.integrations.export.model_dir import load_local_export_tokenizer, load_trainable_decoder_from_export_dir
from ml.tooling.core.json_io import write_json_atomic
from ml.tooling.scripts.train_dpo import _batch, _encode, _model_logits
from ml.training.pretrain.optimizer import create_torch_muon_optimizer
from ml.training.pretrain.engine.train_step import check_finite_update_state
from ml.training.rl.dpo import token_logprobs
from ml.training.rl.grpo import group_advantages, grpo_loss
from ml.training.sft.runner import SFTEagerStepRunner
from ml.training.runtime_tools import apply_gradient_checkpointing


@dataclass(frozen=True)
class Group:
    prompt_id: str
    sequences: tuple[dict[str, torch.Tensor], ...]
    scores: torch.Tensor


def _load_groups(
    path: Path,
    tokenizer: Any,
    max_seq_len: int,
    *,
    minimum_score_std: float,
    minimum_signal_rate: float,
) -> list[Group]:
    groups: list[Group] = []
    judged_groups = 0
    signal_groups = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"GRPO group {line_number} is not an object")
            messages = payload.get("messages")
            rows = payload.get("completions")
            if not isinstance(messages, list) or not isinstance(rows, list):
                continue
            sequences: list[dict[str, torch.Tensor]] = []
            scores: list[float] = []
            for row in rows:
                if not isinstance(row, dict) or "score" not in row:
                    continue
                raw = str(row.get("raw") or "")
                if not raw:
                    continue
                sequences.append(_encode(tokenizer, messages, raw, int(max_seq_len)))
                scores.append(float(row["score"]))
            if len(sequences) >= 2:
                reward = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
                judged_groups += 1
                signal_groups += int(
                    statistics.pstdev(scores) >= float(minimum_score_std)
                )
                try:
                    group_advantages(reward, min_std=float(minimum_score_std))
                except ValueError:
                    continue
                groups.append(
                    Group(
                        prompt_id=str(payload.get("prompt_id") or f"group_{line_number:06d}"),
                        sequences=tuple(sequences),
                        scores=reward,
                    )
                )
    if not groups:
        raise ValueError("no usable GRPO groups with reward signal")
    if judged_groups <= 0 or signal_groups / judged_groups < float(minimum_signal_rate):
        raise RuntimeError(
            "GRPO reward signal below gate: "
            f"{signal_groups}/{judged_groups} groups meet score spread "
            f"{float(minimum_score_std):g}, required rate {float(minimum_signal_rate):g}"
        )
    return groups


def _cache_reference(
    model: torch.nn.Module,
    groups: list[Group],
    *,
    pad_token_id: int,
    device: torch.device,
    minimum_score_std: float,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    model.eval()
    cached: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.inference_mode():
        for group in groups:
            batch = _batch(list(group.sequences), pad_token_id=pad_token_id, device=device)
            logits = _model_logits(model, batch, device=device)
            values = token_logprobs(logits, batch["input_ids"])
            mask = batch["labels"][:, 1:].ne(-100)
            advantages = group_advantages(
                group.scores.to(device=device), min_std=float(minimum_score_std)
            )
            cached.append((values.detach().cpu(), mask.detach().cpu(), advantages.detach().cpu()))
    return cached


def train_grpo(
    *,
    model_dir: Path,
    judged_groups_path: Path,
    output_dir: Path,
    steps: int = 64,
    learning_rate: float = 5.0e-7,
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.02,
    max_seq_len: int = 4096,
    max_groups: int = 128,
    minimum_score_std: float = 5.0,
    minimum_signal_rate: float = 0.25,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    if int(steps) <= 0 or int(max_groups) <= 0:
        raise ValueError("GRPO steps and max_groups must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GRPO training requires CUDA")
    tokenizer = load_local_export_tokenizer(str(model_dir))
    model = load_trainable_decoder_from_export_dir(
        export_dir=str(model_dir),
        device=device,
        base_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        gradient_checkpointing=True,
        gradient_checkpointing_exclude_first=0,
        gradient_checkpointing_exclude_last=0,
        apply_gradient_checkpointing_fn=apply_gradient_checkpointing,
    )
    if float(minimum_score_std) < 0.0 or not 0.0 <= float(minimum_signal_rate) <= 1.0:
        raise ValueError("minimum score gate values are out of range")
    groups = _load_groups(
        judged_groups_path,
        tokenizer,
        int(max_seq_len),
        minimum_score_std=float(minimum_score_std),
        minimum_signal_rate=float(minimum_signal_rate),
    )
    if int(max_groups) > 0:
        groups = groups[: int(max_groups)]
    # Rollouts are category-balanced, but a shuffle prevents a deterministic
    # category order from becoming an optimisation schedule.
    random.Random(42).shuffle(groups)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id)
    cached = _cache_reference(
        model,
        groups,
        pad_token_id=pad_token_id,
        device=device,
        minimum_score_std=float(minimum_score_std),
    )
    gc.collect()
    optimizer = create_torch_muon_optimizer(
        model,
        lr=float(learning_rate),
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        muon_ns_steps=5,
    )
    optimizer.initialize_state()
    runner = SFTEagerStepRunner(model=model, base_dtype=torch.bfloat16, token_weighted=False)
    model.train()
    losses: list[float] = []
    started = time.monotonic()
    for step in range(min(int(steps), len(cached))):
        optimizer.zero_grad(set_to_none=True)
        runner.begin_update(accum_steps=1)
        group = groups[step]
        old, mask, advantages = cached[step]
        batch = _batch(list(group.sequences), pad_token_id=pad_token_id, device=device)
        policy = token_logprobs(_model_logits(model, batch, device=device), batch["input_ids"])
        loss = grpo_loss(
            policy.unsqueeze(0),
            old.to(device=device).unsqueeze(0),
            old.to(device=device).unsqueeze(0),
            advantages.to(device=device),
            mask.to(device=device).unsqueeze(0),
            clip_epsilon=float(clip_epsilon),
            kl_beta=float(kl_beta),
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite GRPO loss at step {step + 1}")
        loss.backward()
        optimizer.step()
        runner.post_optimizer_step(optimizer)
        check_finite_update_state(model=model, optimizer=optimizer)
        losses.append(float(loss.detach().float().cpu()))
        if (step + 1) % 16 == 0 or step + 1 == min(int(steps), len(cached)):
            print(f"[GRPO] step={step + 1} loss={losses[-1]:.4f} prompt={group.prompt_id}", flush=True)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_model_artifacts(model=model, tokenizer=tokenizer, output_dir=str(output_dir), safe_serialization=True)
    report = {
        "schema": "sophia_grpo_run_report",
        "status": "complete",
        "model_dir": str(model_dir.resolve()),
        "judged_groups": len(groups),
        "steps": len(losses),
        "learning_rate": float(learning_rate),
        "clip_epsilon": float(clip_epsilon),
        "kl_beta": float(kl_beta),
        "minimum_score_std": float(minimum_score_std),
        "minimum_signal_rate": float(minimum_signal_rate),
        "mean_loss": sum(losses) / len(losses),
        "elapsed_seconds": time.monotonic() - started,
        "output_dir": str(output_dir.resolve()),
    }
    write_json_atomic(output_dir / "grpo_run_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--judged-groups", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5.0e-7)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-groups", type=int, default=128)
    parser.add_argument("--minimum-score-std", type=float, default=5.0)
    parser.add_argument("--minimum-signal-rate", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = train_grpo(
        model_dir=args.model_dir,
        judged_groups_path=args.judged_groups,
        output_dir=args.output_dir,
        steps=args.steps,
        learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        kl_beta=args.kl_beta,
        max_seq_len=args.max_seq_len,
        max_groups=args.max_groups,
        minimum_score_std=args.minimum_score_std,
        minimum_signal_rate=args.minimum_signal_rate,
        device_name=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
