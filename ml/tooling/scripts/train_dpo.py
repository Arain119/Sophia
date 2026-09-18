"""Run a bounded, reference-cached DPO pass over a SFT export.

The reference is the starting SFT model.  Its sequence scores are computed
once before updates and kept as CPU floats, so only one 1B model and one
optimizer occupy the GPU during the policy pass.  This is deliberately a
short alignment pass: preference data fixes response selection and refusal
boundaries, while the completed SFT remains the capability anchor.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ml.integrations.export.artifacts import export_model_artifacts
from ml.integrations.export.model_dir import load_local_export_tokenizer, load_trainable_decoder_from_export_dir
from ml.modeling.text.conversation_encode import encode_conversation
from ml.tooling.core.json_io import write_json_atomic
from ml.training.pretrain.optimizer import create_torch_muon_optimizer
from ml.training.pretrain.engine.train_step import check_finite_update_state
from ml.training.rl.dpo import PreferencePair, dpo_accuracy, dpo_loss, read_preference_pairs, sequence_logprob
from ml.training.sft.runner import SFTEagerStepRunner
from ml.training.runtime_tools import apply_gradient_checkpointing


@dataclass(frozen=True)
class EncodedPair:
    chosen: dict[str, torch.Tensor]
    rejected: dict[str, torch.Tensor]
    prompt_id: str


def _encode(tokenizer: Any, messages: list[dict[str, Any]], completion: str, max_seq_len: int) -> dict[str, torch.Tensor]:
    # A multi-turn prompt may contain assistant history.  It is context, not a
    # preference target; masking it keeps the DPO margin about the newly
    # judged completion rather than diluting it with identical history tokens.
    conversation = [
        {**dict(message), "mask": True}
        if str(message.get("role") or "") == "assistant"
        else dict(message)
        for message in messages
    ]
    conversation.append({"role": "assistant", "content": str(completion)})
    encoded = encode_conversation(
        tokenizer=tokenizer,
        messages=conversation,
        max_seq_len=int(max_seq_len),
        add_generation_prompt=False,
    )
    if int((encoded.labels != -100).sum().item()) <= 0:
        raise ValueError("preference completion has no supervised tokens after truncation")
    return {
        "input_ids": encoded.input_ids,
        "attention_mask": encoded.attention_mask,
        "labels": encoded.labels,
    }


def _encode_pairs(pairs: Iterable[PreferencePair], tokenizer: Any, max_seq_len: int) -> list[EncodedPair]:
    rows: list[EncodedPair] = []
    for pair in pairs:
        rows.append(
            EncodedPair(
                chosen=_encode(tokenizer, pair.messages, pair.chosen, max_seq_len),
                rejected=_encode(tokenizer, pair.messages, pair.rejected, max_seq_len),
                prompt_id=pair.prompt_id,
            )
        )
    if not rows:
        raise ValueError("DPO preference set is empty")
    return rows


def _batch(items: list[dict[str, torch.Tensor]], *, pad_token_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    width = max(int(item["input_ids"].numel()) for item in items)
    input_ids = torch.full((len(items), width), int(pad_token_id), dtype=torch.long)
    attention = torch.zeros((len(items), width), dtype=torch.long)
    labels = torch.full((len(items), width), -100, dtype=torch.long)
    for index, item in enumerate(items):
        length = int(item["input_ids"].numel())
        input_ids[index, :length] = item["input_ids"]
        attention[index, :length] = item["attention_mask"]
        labels[index, :length] = item["labels"]
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention.to(device),
        "labels": labels.to(device),
    }


def _model_logits(model: torch.nn.Module, batch: dict[str, torch.Tensor], *, device: torch.device) -> torch.Tensor:
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    with amp:
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            compute_loss=False,
            return_dict=True,
        )
    logits = getattr(output, "logits", None)
    if not torch.is_tensor(logits):
        raise RuntimeError("DPO model forward returned no logits")
    return logits


def _score_reference(model: torch.nn.Module, encoded: list[EncodedPair], *, pad_token_id: int, device: torch.device) -> list[tuple[float, float]]:
    model.eval()
    scores: list[tuple[float, float]] = []
    with torch.inference_mode():
        for pair in encoded:
            batch = _batch([pair.chosen, pair.rejected], pad_token_id=pad_token_id, device=device)
            logits = _model_logits(model, batch, device=device)
            values, counts = sequence_logprob(logits, batch["input_ids"], batch["labels"])
            if bool((counts <= 0).any().item()) or not bool(torch.isfinite(values).all().item()):
                raise RuntimeError(f"invalid reference score for {pair.prompt_id}")
            scores.append((float(values[0].cpu()), float(values[1].cpu())))
    return scores


def train_dpo(
    *,
    model_dir: Path,
    pairs_path: Path,
    output_dir: Path,
    steps: int = 256,
    batch_size: int = 1,
    accumulation_steps: int = 1,
    learning_rate: float = 1.0e-6,
    beta: float = 0.1,
    max_seq_len: int = 4096,
    seed: int = 42,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    if int(steps) <= 0 or int(batch_size) <= 0 or int(accumulation_steps) <= 0:
        raise ValueError("steps, batch_size and accumulation_steps must be positive")
    if int(accumulation_steps) != 1:
        raise ValueError("the bounded DPO runner currently requires accumulation_steps=1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DPO training requires CUDA")
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    tokenizer = load_local_export_tokenizer(str(model_dir))
    pairs = list(read_preference_pairs(pairs_path))
    encoded = _encode_pairs(pairs, tokenizer, int(max_seq_len))
    model = load_trainable_decoder_from_export_dir(
        export_dir=str(model_dir),
        device=device,
        base_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        gradient_checkpointing=True,
        gradient_checkpointing_exclude_first=0,
        gradient_checkpointing_exclude_last=0,
        apply_gradient_checkpointing_fn=apply_gradient_checkpointing,
    )
    pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id)
    reference_scores = _score_reference(model, encoded, pad_token_id=pad_token_id, device=device)
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
    accuracies: list[float] = []
    started = time.monotonic()
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        runner.begin_update(accum_steps=int(accumulation_steps))
        batch_indices = [
            (step * int(batch_size) + offset) % len(encoded)
            for offset in range(int(batch_size))
        ]
        policy_chosen: list[torch.Tensor] = []
        policy_rejected: list[torch.Tensor] = []
        ref_chosen: list[float] = []
        ref_rejected: list[float] = []
        for index in batch_indices:
            pair = encoded[index]
            batch = _batch([pair.chosen, pair.rejected], pad_token_id=pad_token_id, device=device)
            logits = _model_logits(model, batch, device=device)
            values, counts = sequence_logprob(logits, batch["input_ids"], batch["labels"])
            if bool((counts <= 0).any().item()):
                raise RuntimeError(f"DPO pair {pair.prompt_id} has no response tokens")
            policy_chosen.append(values[0])
            policy_rejected.append(values[1])
            ref_chosen.append(reference_scores[index][0])
            ref_rejected.append(reference_scores[index][1])
        loss = dpo_loss(
            torch.stack(policy_chosen),
            torch.stack(policy_rejected),
            torch.tensor(ref_chosen, device=device, dtype=torch.float32),
            torch.tensor(ref_rejected, device=device, dtype=torch.float32),
            beta=float(beta),
        )
        loss.backward()
        optimizer.step()
        runner.post_optimizer_step(optimizer)
        check_finite_update_state(model=model, optimizer=optimizer)
        losses.append(float(loss.detach().float().cpu()))
        accuracies.append(
            float(
                dpo_accuracy(
                    torch.stack(policy_chosen).detach(),
                    torch.stack(policy_rejected).detach(),
                    torch.tensor(ref_chosen, device=device),
                    torch.tensor(ref_rejected, device=device),
                ).cpu()
            )
        )
        if (step + 1) % 16 == 0 or step + 1 == int(steps):
            print(
                f"[DPO] step={step + 1}/{steps} loss={losses[-1]:.4f} "
                f"accuracy={accuracies[-1]:.3f}",
                flush=True,
            )
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_model_artifacts(
        model=model,
        tokenizer=tokenizer,
        output_dir=str(output_dir),
        safe_serialization=True,
    )
    report = {
        "schema": "sophia_dpo_run_report",
        "status": "complete",
        "model_dir": str(model_dir.resolve()),
        "pairs": len(encoded),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "beta": float(beta),
        "max_seq_len": int(max_seq_len),
        "mean_loss": sum(losses) / len(losses),
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "elapsed_seconds": time.monotonic() - started,
        "output_dir": str(output_dir.resolve()),
    }
    write_json_atomic(output_dir / "dpo_run_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = train_dpo(
        model_dir=args.model_dir,
        pairs_path=args.pairs,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
