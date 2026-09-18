"""Packed supervised fine-tuning from the pretrained checkpoint.

    python -m ml.cli.sft_train --dataset release/corpus --parent <ckpt> --output runs/sft

Conversations are encoded with the chat template and labels masked to
assistant tokens. One conversation per row: the model has no attention mask,
so several conversations packed into one row would all see each other, and a
policy trained that way learns to invent the context it half-remembers. Rows
are batched with their nearest neighbours by length and right-padded, which
costs a few per cent of padding and keeps the card near pretraining
throughput. An update is as many batches as it takes to reach
``--tokens-per-update``; the loss is the token-weighted mean over the
supervised tokens in that window.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from ml.modeling.text.conversation_encode import encode_conversation
from ml.training.pretrain.engine.train_step import check_finite_update_state
from ml.training.pretrain.model_setup import load_tokenizer
from ml.training.pretrain.optimizer import create_torch_muon_optimizer
from ml.training.runtime_tools import apply_gradient_checkpointing
from ml.training.scheduler import build_lr_scheduler
from ml.training.sft.runner import SFTEagerStepRunner
from ml.training.sft.trainer import _model_from_spec


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encode_rows(tokenizer: Any, rows: list[dict[str, Any]], seq_len: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for row in rows:
        enc = encode_conversation(tokenizer=tokenizer, messages=row["messages"], max_seq_len=seq_len, add_generation_prompt=False)
        if int((enc.labels != -100).sum().item()) > 0:
            out.append((enc.input_ids, enc.labels))
    return out


def batches(
    items: list[tuple[torch.Tensor, torch.Tensor]], batch_tokens: int, max_rows: int, pad_id: int, rng: random.Random
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """One conversation per row, grouped with its nearest neighbours by length.

    A batch is as many rows as fit in ``batch_tokens``, so a group of short
    conversations costs one kernel launch instead of dozens and the activation
    memory stays flat across widths. Right padding is safe without an
    attention mask: attention is causal, so a real token never reads a pad
    that follows it, and the pads carry label -100.
    """
    order = sorted(range(len(items)), key=lambda i: -int(items[i][0].numel()))
    out, cursor = [], 0
    while cursor < len(order):
        width = int(items[order[cursor]][0].numel())
        rows = max(1, min(int(batch_tokens) // max(width, 1), int(max_rows), len(order) - cursor))
        members = order[cursor : cursor + rows]
        cursor += rows
        ids, labels = [], []
        for i in members:
            row_ids, row_labels = items[i]
            fill = width - int(row_ids.numel())
            ids.append(torch.cat([row_ids, torch.full((fill,), pad_id, dtype=torch.long)]) if fill else row_ids)
            labels.append(torch.cat([row_labels, torch.full((fill,), -100, dtype=torch.long)]) if fill else row_labels)
        out.append((torch.stack(ids), torch.stack(labels)))
    rng.shuffle(out)
    return out


def update_count(rows: list[tuple[torch.Tensor, torch.Tensor]], tokens_per_update: int) -> int:
    """Updates one pass over ``rows`` produces, counted the way the loop spends them."""
    steps, budget = 0, 0
    for ids, _ in rows:
        if budget >= tokens_per_update:
            steps, budget = steps + 1, 0
        budget += int(ids.numel())
    return steps + 1 if budget else steps


def evaluate(model: torch.nn.Module, rows: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> float:
    model.eval()
    loss_sum, tokens = 0.0, 0
    with torch.no_grad():
        for ids, labels in rows:
            ids, labels = ids.to(device), labels.to(device)
            n = int((labels != -100).sum().item())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, labels=labels, compute_loss=True, use_cache=False)
            loss_sum += float(out.loss.detach().float()) * n
            tokens += n
    model.train()
    return loss_sum / max(tokens, 1)


def save(model: torch.nn.Module, path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, **meta}, tmp)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-spec", default="configs/model/sophia.json")
    parser.add_argument("--tokenizer-path", default="ml/modeling/text")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-tokens", type=int, default=8192)
    parser.add_argument("--max-rows", type=int, default=32)
    parser.add_argument("--tokens-per-update", type=int, default=131072)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-checkpoint", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.tokenizer_path)
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id)
    dataset = Path(args.dataset)
    train_items = encode_rows(tokenizer, read_rows(dataset / "train.jsonl"), args.seq_len)
    val_items = encode_rows(tokenizer, read_rows(dataset / "val.jsonl"), args.seq_len)
    val_rows = batches(val_items, int(args.batch_tokens), int(args.max_rows), pad_id, random.Random(0))
    epoch_rows = [
        batches(train_items, int(args.batch_tokens), int(args.max_rows), pad_id, random.Random(args.seed + e))
        for e in range(args.epochs)
    ]
    padded = sum(int(i.numel()) for i, _ in epoch_rows[0])
    real = sum(int(i.numel()) for i, _ in train_items)
    # Every epoch reshuffles, so every epoch packs into its own number of
    # updates; multiplying the first epoch's count by the epoch count is off by
    # however much the later shuffles differ, and the scheduler then runs out of
    # schedule one step before the run ends.
    steps_per_epoch = [update_count(rows, int(args.tokens_per_update)) for rows in epoch_rows]
    total_steps = sum(steps_per_epoch)
    supervised = sum(int((labels != -100).sum()) for _, labels in train_items)
    plan = {
        "conversations": len(train_items),
        "batches_per_epoch": len(epoch_rows[0]),
        "tokens_per_epoch": padded,
        "supervised_tokens_per_epoch": supervised,
        "pad_waste": round(1.0 - real / max(padded, 1), 4),
        "longest_row": max(int(i.numel()) for i, _ in train_items),
        "tokens_per_update": int(args.tokens_per_update),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "val_batches": len(val_rows),
        "learning_rate": args.learning_rate,
    }
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return

    device = torch.device("cuda")
    model = _model_from_spec(Path(args.model_spec), tokenizer=tokenizer, batch_size=int(args.max_rows))
    payload = torch.load(args.parent, map_location="cpu", weights_only=True)
    state = payload["model"] if "model" in payload else payload["model_state_dict"]
    model.load_state_dict(state, strict=True)
    parent_step = payload.get("step")
    del payload, state
    model.to(device=device, dtype=torch.bfloat16)
    apply_gradient_checkpointing(model, enabled=bool(args.grad_checkpoint))

    optimizer = create_torch_muon_optimizer(
        model, lr=float(args.learning_rate), weight_decay=float(args.weight_decay), betas=(0.9, 0.95), eps=1.0e-8, muon_ns_steps=5
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

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = (output / "metrics.jsonl").open("a", encoding="utf-8", newline="\n")
    meta = {
        "stage": "sft",
        "parent": str(args.parent),
        "parent_step": parent_step,
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("dry_run",)},
        "plan": plan,
    }

    model.train()
    started = time.time()
    step = 0
    val_loss = evaluate(model, val_rows, device)
    print(json.dumps({"type": "eval", "step": 0, "val_loss": val_loss}), flush=True)
    metrics.write(json.dumps({"type": "eval", "step": 0, "val_loss": val_loss}) + "\n")
    for epoch in range(int(args.epochs)):
        rows = epoch_rows[epoch]
        cursor = 0
        while cursor < len(rows):
            # An update is however many batches it takes to reach the token
            # budget; short conversations pack more of them into one step.
            window, budget = [], 0
            while cursor < len(rows) and budget < int(args.tokens_per_update):
                window.append(rows[cursor])
                budget += int(rows[cursor][0].numel())
                cursor += 1
            optimizer.zero_grad(set_to_none=True)
            runner.begin_update(accum_steps=len(window))
            tokens_total, loss_sum = 0, 0.0
            for batch_ids, batch_labels in window:
                ids = batch_ids.to(device, non_blocking=True)
                labels = batch_labels.to(device, non_blocking=True)
                n = int((labels != -100).sum().item())
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(input_ids=ids, labels=labels, compute_loss=True, use_cache=False)
                (out.loss * n).backward()
                loss_sum += float(out.loss.detach().float()) * n
                tokens_total += n
            for p in model.parameters():  # the hybrid optimizer's groups hold master copies, not these
                if p.grad is not None:
                    p.grad.div_(max(tokens_total, 1))
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
                "loss": loss_sum / max(tokens_total, 1),
                "grad_norm": float(grad_norm),
                "lr": float(scheduler.get_last_lr()[0]),
                "supervised_tokens": tokens_total,
                "elapsed_s": round(time.time() - started, 1),
                "max_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
            if not math.isfinite(record["loss"]):
                raise RuntimeError(f"non-finite loss at step {step}")
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            if step % 10 == 0 or step <= 3:
                print(json.dumps(record), flush=True)
            if step % int(args.eval_interval) == 0:
                val_loss = evaluate(model, val_rows, device)
                metrics.write(json.dumps({"type": "eval", "step": step, "val_loss": val_loss}) + "\n")
                metrics.flush()
                print(json.dumps({"type": "eval", "step": step, "val_loss": val_loss}), flush=True)
            if step % int(args.save_interval) == 0:
                save(model, output / "ckpt_latest.pt", {**meta, "step": step})
            if args.max_steps and step >= int(args.max_steps):
                break
        save(model, output / f"ckpt_epoch{epoch + 1}.pt", {**meta, "step": step, "epoch": epoch + 1})
        if args.max_steps and step >= int(args.max_steps):
            break
    val_loss = evaluate(model, val_rows, device)
    metrics.write(json.dumps({"type": "eval", "step": step, "val_loss": val_loss, "final": True}) + "\n")
    metrics.close()
    save(model, output / "ckpt_final.pt", {**meta, "step": step, "val_loss": val_loss})
    print(json.dumps({"saved": str(output / "ckpt_final.pt"), "steps": step, "val_loss": val_loss, "wall_min": round((time.time() - started) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
