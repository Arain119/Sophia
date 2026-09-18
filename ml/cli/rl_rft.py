"""Rejection-sampling fine-tuning: the shortest path from a flat policy to a sharp one.

The run's measured defect is that greedy decoding answers well and sampled
decoding does not. That gap is not missing knowledge -- the same weights
produce both -- it is probability mass sitting on tokens the argmax never
visits. Cross-entropy on more corpus cannot move it, because the corpus is
what put it there and a third epoch changed the loss by 0.02.

What moves it is one step of policy iteration. Sample the policy, keep what
the reward likes, fit the policy to that: the new target is the old policy
reweighted by the reward, which is exactly the distribution the sampled decode
is supposed to be drawing from. It needs no reference model, no ratio, no KL
coefficient, and it cannot diverge -- the worst case is that it learns nothing
because every sample was equally good.

Only the best completion in each group is kept. Keeping everything above a
threshold trains on the mean of the acceptable set, which is the mode the
policy already has; keeping only the argmax is what sharpens.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from ml.tooling.scripts.train_dpo import _batch, _encode
from ml.training.pretrain.model_setup import load_tokenizer
from ml.training.pretrain.optimizer import create_torch_muon_optimizer
from ml.training.pretrain.engine.train_step import check_finite_update_state
from ml.training.rl.engine import read_jsonl
from ml.training.runtime_tools import apply_gradient_checkpointing
from ml.training.scheduler import build_lr_scheduler
from ml.training.sft.runner import SFTEagerStepRunner
from ml.training.sft.trainer import _model_from_spec


def rebuild_completion(sample: dict[str, Any]) -> str:
    """The assistant turn exactly as the policy emitted it.

    A think block stays in when the winning sample had one and is absent when
    the winner was sampled with an empty block: which of the two wins is how
    the policy learns where thinking helps and where it only adds a place to
    go wrong.
    """
    think = str(sample.get("think") or "").strip()
    answer = str(sample.get("answer") or "").strip()
    if think:
        return "<think>" + think + "</think>" + answer
    return answer


def select(
    groups: list[dict[str, Any]],
    *,
    floor: float,
    anchor_floor: float,
    anchor_margin: float,
    fuel_ceiling: float,
    max_repetition: float,
    require_eos: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fuel prompts are kept when the best is good and the group is bad: that
    gap is the whole gradient. Anchor prompts are kept only when the best
    clearly beats a group that was already fine, so the round revisits what
    the release must keep without teaching it anything it does not do."""
    kept: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    margins: list[float] = []
    think_share: Counter[str] = Counter()
    for group in groups:
        usable = [
            s
            for s in group["samples"]
            if s.get("judge_ok")
            and s.get("reward") is not None
            and str(s.get("answer") or "").strip()
            and not s.get("flags")
        ]
        if not usable:
            reasons["no_clean_sample"] += 1
            continue
        best = max(usable, key=lambda s: float(s["reward"]))
        scored = [float(s["judge_score"]) for s in group["samples"] if s.get("judge_ok")]
        mean = statistics.fmean(scored)
        if group.get("role") == "anchor":
            if float(best["judge_score"]) < float(anchor_floor) or float(best["judge_score"]) - mean < float(anchor_margin):
                reasons["anchor_no_margin"] += 1
                continue
        else:
            if float(best["judge_score"]) < float(floor):
                reasons["best_below_floor"] += 1
                continue
            if mean > float(fuel_ceiling):
                reasons["fuel_group_already_good"] += 1
                continue
        if float(best["repetition"]) > float(max_repetition):
            reasons["best_repetitive"] += 1
            continue
        if require_eos and not best.get("hit_eos"):
            reasons["best_unfinished"] += 1
            continue
        others = [float(s["reward"]) for s in usable if s is not best]
        margins.append(float(best["reward"]) - (statistics.fmean(others) if others else 0.0))
        kept.append(
            {
                "prompt_id": group["prompt_id"],
                "messages": group["messages"],
                "category": group.get("category"),
                "role": group.get("role"),
                "move": group.get("move"),
                "think_mode": best.get("think_mode"),
                "completion": rebuild_completion(best),
                "judge_score": float(best["judge_score"]),
                "reward": float(best["reward"]),
            }
        )
        reasons["kept"] += 1
        think_share[str(group.get("category"))] += 1
        if best.get("think"):
            think_share[str(group.get("category")) + ":think"] += 1
    stats = {
        "groups": len(groups),
        "kept": len(kept),
        "keep_rate": round(len(kept) / max(len(groups), 1), 4),
        "reasons": dict(reasons),
        "mean_kept_judge": round(
            statistics.fmean([r["judge_score"] for r in kept]), 2
        )
        if kept
        else None,
        "mean_best_minus_rest_reward": round(statistics.fmean(margins), 4)
        if margins
        else None,
        "by_category": dict(Counter(str(r["category"]) for r in kept)),
        "by_role": dict(Counter(str(r["role"]) for r in kept)),
        "think_share": {
            c: round(think_share[c + ":think"] / think_share[c], 3)
            for c in sorted(k for k in think_share if ":" not in k)
        },
    }
    return kept, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--parent", required=True, help="checkpoint to start from")
    parser.add_argument("--output", required=True, help="checkpoint to write")
    parser.add_argument("--model-spec", default="configs/model/sophia.json")
    parser.add_argument("--tokenizer-path", default="ml/modeling/text")
    parser.add_argument("--floor", type=float, default=85.0)
    parser.add_argument("--anchor-floor", type=float, default=90.0)
    parser.add_argument("--anchor-margin", type=float, default=8.0)
    parser.add_argument("--fuel-ceiling", type=float, default=70.0)
    parser.add_argument("--max-repetition", type=float, default=0.15)
    parser.add_argument("--require-eos", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    groups = read_jsonl(args.rollouts)
    kept, stats = select(
        groups,
        floor=float(args.floor),
        anchor_floor=float(args.anchor_floor),
        anchor_margin=float(args.anchor_margin),
        fuel_ceiling=float(args.fuel_ceiling),
        max_repetition=float(args.max_repetition),
        require_eos=bool(args.require_eos),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    if not kept:
        raise SystemExit("RFT selected nothing; lower --floor or fix the rollout")

    tokenizer = load_tokenizer(str(args.tokenizer_path))
    encoded: list[dict[str, torch.Tensor]] = []
    dropped = 0
    for row in kept:
        try:
            encoded.append(
                _encode(tokenizer, row["messages"], row["completion"], int(args.max_seq_len))
            )
        except ValueError:
            dropped += 1
    print(
        json.dumps(
            {"encoded": len(encoded), "dropped_on_truncation": dropped},
            ensure_ascii=False,
        ),
        flush=True,
    )

    # Length-sorted micro-batches: padding to the longest row in a batch is
    # the whole cost of a ragged batch, and sorting removes most of it.
    rng = random.Random(args.seed)
    order = sorted(range(len(encoded)), key=lambda i: int(encoded[i]["input_ids"].numel()))
    micro = [
        [encoded[i] for i in order[start : start + int(args.micro_batch)]]
        for start in range(0, len(order), int(args.micro_batch))
    ]
    steps_per_epoch = max(len(micro) // int(args.accumulation), 1)
    total_steps = steps_per_epoch * int(args.epochs)
    print(
        json.dumps(
            {
                "micro_batches": len(micro),
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "effective_batch": int(args.micro_batch) * int(args.accumulation),
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return

    device = torch.device("cuda")
    model = _model_from_spec(
        Path(args.model_spec), tokenizer=tokenizer, batch_size=int(args.micro_batch)
    )
    payload = torch.load(args.parent, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    parent_step = payload.get("step")
    del payload
    model.to(device=device, dtype=torch.bfloat16)
    apply_gradient_checkpointing(model, enabled=True)

    optimizer = create_torch_muon_optimizer(
        model,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
        eps=1.0e-8,
        muon_ns_steps=5,
    )
    optimizer.initialize_state()
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        max_steps=max(total_steps, 1),
        warmup_steps=max(int(total_steps * float(args.warmup_ratio)), 1),
        warmup_ratio=0.0,
        min_lr_ratio=float(args.min_lr_ratio),
        schedule="cosine",
        resume_state=None,
    )
    runner = SFTEagerStepRunner(model=model, base_dtype=torch.bfloat16, token_weighted=True)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id)

    metrics_path = Path(args.metrics) if args.metrics else None
    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_handle = metrics_path.open("a", encoding="utf-8", newline="\n")
    else:
        metrics_handle = None

    model.train()
    started = time.time()
    step = 0
    for epoch in range(int(args.epochs)):
        rng.shuffle(micro)
        for start in range(0, len(micro) - int(args.accumulation) + 1, int(args.accumulation)):
            optimizer.zero_grad(set_to_none=True)
            runner.begin_update(accum_steps=int(args.accumulation))
            window = micro[start : start + int(args.accumulation)]
            supervised = 0
            loss_sum = 0.0
            for chunk in window:
                batch = _batch(chunk, pad_token_id=pad_token_id, device=device)
                tokens = int((batch["labels"] != -100).sum().item())
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        compute_loss=True,
                        use_cache=False,
                    )
                # Token-weighted so a batch of short rows does not count the
                # same as a batch of long ones inside one accumulation window.
                # The window's total token count is only known once the window
                # is done, so accumulate loss*tokens here and divide by that
                # total below -- dividing here as well would scale the whole
                # gradient by an extra 1/accumulation.
                (output.loss * tokens).backward()
                loss_sum += float(output.loss.detach().float()) * tokens
                supervised += tokens
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.div_(max(supervised, 1))
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            runner.post_optimizer_step(optimizer)
            check_finite_update_state(model=model, optimizer=optimizer)
            scheduler.step()
            step += 1
            record = {
                "type": "train",
                "step": step,
                "epoch": epoch,
                "loss": loss_sum / max(supervised, 1),
                "grad_norm": float(grad_norm),
                "lr": float(scheduler.get_last_lr()[0]),
                "supervised_tokens": supervised,
                "elapsed_s": round(time.time() - started, 1),
            }
            if not math.isfinite(record["loss"]):
                raise RuntimeError(f"non-finite RFT loss at step {step}")
            if metrics_handle:
                metrics_handle.write(json.dumps(record) + "\n")
                metrics_handle.flush()
            if step % 10 == 0 or step == 1:
                print(json.dumps(record), flush=True)

    model.eval()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "step": step,
            "stage": "rft",
            "parent": str(args.parent),
            "parent_step": parent_step,
            "selection": stats,
            "hyperparameters": {
                "learning_rate": float(args.learning_rate),
                "epochs": int(args.epochs),
                "effective_batch": int(args.micro_batch) * int(args.accumulation),
                "floor": float(args.floor),
            },
        },
        output_path,
    )
    if metrics_handle:
        metrics_handle.close()
    print(
        json.dumps(
            {"saved": str(output_path), "steps": step, "wall_min": round((time.time() - started) / 60, 1)},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
