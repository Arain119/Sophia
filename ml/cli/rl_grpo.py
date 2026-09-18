"""On-policy GRPO: rounds of sample, judge, update, resample.

RFT moves the policy onto its own best samples in one shot and then stops
having anything to say, because the samples it was fit to came from a policy
that no longer exists. GRPO is the same idea run as a loop -- and unlike RFT
it uses the whole group, so a prompt where every sample is mediocre still
pushes down on the worst of them instead of being discarded.

The reference is the frozen RFT policy, held resident on the card, and the KL
term is measured against it rather than against the previous round. Anchoring
to the previous round makes each step small but lets the total walk anywhere,
which is how a policy ends up fluent, high-reward and no longer speaking
Chinese. Two billion-parameter models in bf16 is four gigabytes; the anchor is
worth it.

Each round writes a checkpoint and the probe is re-run against it, so the
decision to stop is made on measured behaviour rather than on a step budget.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from ml.tooling.scripts.train_dpo import _batch, _encode, _model_logits
from ml.training.pretrain.model_setup import load_tokenizer
from ml.training.pretrain.optimizer import create_torch_muon_optimizer
from ml.training.pretrain.engine.train_step import check_finite_update_state
from ml.training.rl.engine import RolloutEngine, encode_prompt, read_jsonl
from ml.training.rl.grpo import group_advantages, grpo_loss
from ml.training.rl.judge import Judge, Ledger
from ml.training.rl.prompt_pool import derive_expectations
from ml.training.rl.reward import score_completion
from ml.training.runtime_tools import apply_gradient_checkpointing
from ml.training.sft.runner import SFTEagerStepRunner
from ml.training.sft.trainer import _model_from_spec



def token_logprobs_chunked(
    logits: torch.Tensor, input_ids: torch.Tensor, *, chunk: int = 128
) -> torch.Tensor:
    """Per-token log probabilities without materialising [B, T, 65536] in fp32.

    The vocabulary is 65,536, so one full fp32 log_softmax over a group of
    eight 900-token sequences is 1.9GB -- and the autograd graph keeps it,
    alongside another for the result. Slicing the sequence into chunks costs
    nothing in arithmetic and drops the peak to a chunk's worth, which is what
    makes the update fit next to the optimiser state.
    """
    targets = input_ids[:, 1:]
    shifted = logits[:, :-1]
    pieces = []
    for start in range(0, shifted.size(1), int(chunk)):
        window = shifted[:, start : start + int(chunk)].float()
        window = torch.log_softmax(window, dim=-1)
        picked = targets[:, start : start + int(chunk)].long().unsqueeze(-1)
        pieces.append(window.gather(-1, picked).squeeze(-1))
    return torch.cat(pieces, dim=1)



def group_logprobs(
    model: Any,
    rows: list[dict[str, torch.Tensor]],
    *,
    pad_token_id: int,
    device: Any,
    sub_batch: int,
) -> torch.Tensor:
    """Log probabilities for a whole group, forwarded a few sequences at a time.

    A group of 32 sequences produces a [32, ~900, 65536] logit tensor -- 3.8GB
    in bf16, kept alive for the backward pass. Splitting the forward and
    concatenating the per-token results costs nothing in arithmetic and keeps
    autograd connected, so the group size is free to be as large as pass@K
    wants it rather than as large as one logit tensor allows.
    """
    pieces = []
    for start in range(0, len(rows), int(sub_batch)):
        chunk = rows[start : start + int(sub_batch)]
        batch = _batch(chunk, pad_token_id=int(pad_token_id), device=device)
        pieces.append(
            token_logprobs_chunked(
                _model_logits(model, batch, device=device), batch["input_ids"]
            )
        )
    width = max(int(p.size(1)) for p in pieces)
    padded = [
        torch.nn.functional.pad(p, (0, width - int(p.size(1)))) for p in pieces
    ]
    return torch.cat(padded, dim=0)


def build_model(spec: str, tokenizer: Any, batch_size: int, checkpoint: str, device, *, trainable: bool):
    model = _model_from_spec(Path(spec), tokenizer=tokenizer, batch_size=int(batch_size))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    del payload
    gc.collect()
    model.to(device=device, dtype=torch.bfloat16)
    if trainable:
        apply_gradient_checkpointing(model, enabled=True)
        model.train()
    else:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


def rebuild(sample: Any) -> str:
    think = (sample.think or "").strip()
    answer = (sample.answer or "").strip()
    return ("<think>" + think + "</think>" + answer) if think else answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-spec", default="configs/model/sophia.json")
    parser.add_argument("--tokenizer-path", default="ml/modeling/text")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--prompts-per-round", type=int, default=384)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--inner-epochs", type=int, default=1)
    parser.add_argument("--groups-per-step", type=int, default=4)
    parser.add_argument("--update-group-chunk", type=int, default=0,
                        help="split group fwd+bwd into chunks of this many rows; 0 keeps whole-group")
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument(
        "--bc-weight",
        type=float,
        default=0.0,
        help="lambda on the gated best-of-K cloning term; 0 is plain GRPO",
    )
    parser.add_argument(
        "--bc-floor",
        type=float,
        default=2.2,
        help="absolute reward the group best must clear before it is cloned",
    )
    parser.add_argument(
        "--bc-margin",
        type=float,
        default=0.35,
        help="reward margin over the group mean at which cloning reaches full weight",
    )
    parser.add_argument("--min-reward-std", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=640,
        help="cap prompt history so rollout batches are wide enough to be worth running",
    )
    parser.add_argument("--forward-sub-batch", type=int, default=8)
    parser.add_argument(
        "--prefill-positions",
        type=int,
        default=110_000,
        help="batch x prompt_len ceiling for prefill; the real limit on batch width",
    )
    parser.add_argument("--max-batch", type=int, default=48)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--ledger", default="ops/rl/ledger.json")
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--judge-workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Prompt and reply share one 4096-token window; reserve the reply.
    # Two constraints, and the tighter one wins. The window reserves room
    # for the reply; the cap exists because this runtime can only batch
    # prompts of identical length, and unbounded multi-turn history gives
    # 2,574 distinct lengths over 6,000 prompts -- batches of 18, at which
    # point the card runs at a fifth of its measured decode rate.
    prompt_budget = min(
        4096 - int(args.max_new_tokens) - 8, int(args.max_prompt_tokens)
    )

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = (output_dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n")

    rng = random.Random(args.seed)
    pool = read_jsonl(args.pool)
    rng.shuffle(pool)

    tokenizer = load_tokenizer(str(args.tokenizer_path))
    pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id)

    policy = build_model(
        args.model_spec, tokenizer, args.max_batch, args.parent, device, trainable=True
    )
    reference = build_model(
        args.model_spec, tokenizer, args.max_batch, args.parent, "cpu", trainable=False
    )
    engine = RolloutEngine(
        model=policy,
        tokenizer=tokenizer,
        max_batch=args.max_batch,
        max_new_tokens=args.max_new_tokens,
        prefill_positions=int(args.prefill_positions),
    )
    optimizer = create_torch_muon_optimizer(
        policy,
        lr=float(args.learning_rate),
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        muon_ns_steps=5,
    )
    optimizer.initialize_state()
    runner = SFTEagerStepRunner(model=policy, base_dtype=torch.bfloat16, token_weighted=False)
    judge_model = os.environ.get("JUDGE_MODEL") or "claude-haiku-4-5"
    ledger = Ledger(args.ledger, args.budget, model=judge_model)
    judge = Judge(ledger=ledger, workers=args.judge_workers, model=judge_model)

    cursor = 0
    global_step = 0
    for round_index in range(int(args.rounds)):
        if ledger.exhausted():
            print(f"BUDGET EXHAUSTED - stopping before round {round_index}", flush=True)
            break
        slice_ = pool[cursor : cursor + int(args.prompts_per_round)]
        cursor += int(args.prompts_per_round)
        if len(slice_) < 8:
            print("pool exhausted", flush=True)
            break

        # ---- rollout ------------------------------------------------------
        policy.eval()
        started = time.time()
        requests = [
            (
                str(r["prompt_id"]),
                encode_prompt(tokenizer, r["messages"], max_prompt_tokens=prompt_budget),
                int(args.group_size),
            )
            for r in slice_
        ]
        produced = engine.run(
            requests,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
        )
        gen_seconds = time.time() - started
        by_id = {str(r["prompt_id"]): r for r in slice_}

        tasks, flat = [], []
        for prompt_id, samples in produced.items():
            for sample in samples:
                flat.append((prompt_id, sample))
                tasks.append(
                    {"conversation": by_id[prompt_id]["messages"], "answer": sample.answer}
                )
        judge_started = time.time()
        verdicts = judge.score_many(tasks)
        judge_seconds = time.time() - judge_started

        rewards_by_prompt: dict[str, list[tuple[Any, float]]] = {}
        judged_scores: list[float] = []
        for (prompt_id, sample), verdict in zip(flat, verdicts, strict=False):
            if not verdict.ok:
                continue
            breakdown = score_completion(
                sample.answer,
                coherence=verdict.score / 100.0,
                hit_eos=sample.hit_eos,
                expectations=derive_expectations(by_id[prompt_id]),
            )
            rewards_by_prompt.setdefault(prompt_id, []).append((sample, breakdown.total))
            judged_scores.append(verdict.score)

        # ---- keep only groups whose members disagree ----------------------
        groups: list[tuple[str, list[Any], torch.Tensor]] = []
        for prompt_id, rows in rewards_by_prompt.items():
            if len(rows) < 2:
                continue
            values = [value for _, value in rows]
            if statistics.pstdev(values) < float(args.min_reward_std):
                continue
            groups.append(
                (prompt_id, [s for s, _ in rows], torch.tensor(values, dtype=torch.float32))
            )
        signal_rate = len(groups) / max(len(rewards_by_prompt), 1)
        print(
            json.dumps(
                {
                    "round": round_index,
                    "prompts": len(slice_),
                    "gen_s": round(gen_seconds, 1),
                    "judge_s": round(judge_seconds, 1),
                    "judge_mean": round(statistics.fmean(judged_scores), 2) if judged_scores else None,
                    "signal_groups": len(groups),
                    "signal_rate": round(signal_rate, 3),
                    "usd": ledger.spent(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if not groups:
            print("no group carried signal this round; stopping", flush=True)
            break

        # ---- encode and cache old / reference logprobs ---------------------
        encoded_groups = []
        for prompt_id, samples, values in groups:
            try:
                rows = [
                    _encode(tokenizer, by_id[prompt_id]["messages"], rebuild(s), int(args.max_seq_len))
                    for s in samples
                ]
            except ValueError:
                continue
            encoded_groups.append((prompt_id, rows, values))

        cached = []
        policy.eval()
        # The anchor is only consulted while caching. Two gigabytes of frozen
        # weights sitting on the card through the update pass is two
        # gigabytes the optimiser state and activations do not get.
        reference.to(device)
        # no_grad, not inference_mode: tensors produced under inference_mode
        # are permanently barred from autograd, and these are read back inside
        # the surrogate loss on the next pass.
        with torch.no_grad():
            for _prompt_id, rows, values in encoded_groups:
                batch = _batch(rows, pad_token_id=pad_token_id, device=device)
                mask = batch["labels"][:, 1:].ne(-100)
                old = group_logprobs(
                    policy, rows, pad_token_id=pad_token_id, device=device,
                    sub_batch=int(args.forward_sub_batch),
                )
                ref = group_logprobs(
                    reference, rows, pad_token_id=pad_token_id, device=device,
                    sub_batch=int(args.forward_sub_batch),
                )
                try:
                    advantages = group_advantages(
                        values.unsqueeze(0).to(device), min_std=float(args.min_reward_std)
                    )
                except ValueError:
                    continue
                # Held on CPU. One round keeps a few hundred groups, and at
                # eight sequences of ~900 tokens each these fp32 logprobs are
                # about twenty gigabytes if they stay on the card -- which is
                # the whole card, and the backward pass then has nowhere to
                # live. They are read back one window at a time.
                # Gated best-of-K cloning. GRPO already knows which member of
                # the group won; it just spends that knowledge on a
                # standardised coefficient. Cloning the winner directly is
                # RFT's dense signal -- but only when the winner is worth
                # cloning. Measured: best-of-8 reaches 97 on the easy pool and
                # the group mean is 52 on the hard one, so an ungated clone
                # would hold up the least-bad nonsense as a target on exactly
                # the prompts that are already the problem.
                values_list = [float(v) for v in values.tolist()]
                best_index = max(range(len(values_list)), key=lambda i: values_list[i])
                best_value = values_list[best_index]
                group_mean = sum(values_list) / len(values_list)
                gate = 1.0 if best_value >= float(args.bc_floor) else 0.0
                margin = (best_value - group_mean) / max(float(args.bc_margin), 1e-6)
                bc_weight = gate * max(0.0, min(1.0, margin))
                cached.append(
                    {
                        "rows": rows,
                        "old": old.detach().to("cpu"),
                        "ref": ref.detach().to("cpu"),
                        "mask": mask.detach().to("cpu"),
                        "adv": advantages.detach().to("cpu"),
                        "best_index": best_index,
                        "bc_weight": bc_weight,
                    }
                )
        reference.to("cpu")
        del encoded_groups
        gc.collect()
        torch.cuda.empty_cache()

        # ---- policy updates -----------------------------------------------
        policy.train()
        losses: list[float] = []
        for _ in range(int(args.inner_epochs)):
            rng.shuffle(cached)
            for start in range(0, len(cached) - int(args.groups_per_step) + 1, int(args.groups_per_step)):
                window = cached[start : start + int(args.groups_per_step)]
                optimizer.zero_grad(set_to_none=True)
                runner.begin_update(accum_steps=len(window))
                step_loss = 0.0
                cloned = 0
                for item in window:
                    chunk = int(args.update_group_chunk)
                    rows = item["rows"]
                    spans = (
                        [(i, min(i + chunk, len(rows))) for i in range(0, len(rows), chunk)]
                        if chunk > 0 and len(rows) > chunk
                        else [(0, len(rows))]
                    )
                    total_den = item["mask"].sum().clamp_min(1.0)
                    bc_weight = float(item.get("bc_weight", 0.0))
                    best_row = int(item["best_index"])
                    item_loss = 0.0
                    for k0, k1 in spans:
                        policy_logprobs = group_logprobs(
                            policy, rows[k0:k1], pad_token_id=pad_token_id, device=device,
                            sub_batch=int(args.forward_sub_batch),
                        )
                        t_h = policy_logprobs.size(1)
                        mask_h = item["mask"][k0:k1, :t_h]
                        den_h = mask_h.sum().clamp_min(1.0)
                        loss = grpo_loss(
                            policy_logprobs.unsqueeze(0),
                            item["old"][k0:k1, :t_h].to(device).unsqueeze(0),
                            item["ref"][k0:k1, :t_h].to(device).unsqueeze(0),
                            item["adv"][:, k0:k1].to(device),
                            mask_h.to(device).unsqueeze(0),
                            clip_epsilon=float(args.clip_epsilon),
                            kl_beta=float(args.kl_beta),
                        )
                        # loss is a masked token mean over this span; rescale
                        # by the span's mask share so spans sum to the group loss
                        scaled = loss * (den_h / total_den)
                        if float(args.bc_weight) > 0.0 and bc_weight > 0.0 and k0 <= best_row < k1:
                            row_mask = mask_h[best_row - k0].to(device).float()
                            chosen = policy_logprobs[best_row - k0]
                            clone = -(chosen * row_mask).sum() / row_mask.sum().clamp_min(1.0)
                            scaled = scaled + float(args.bc_weight) * bc_weight * clone
                        if not torch.isfinite(scaled):
                            raise RuntimeError("non-finite GRPO loss")
                        (scaled / len(window)).backward()
                        item_loss += float(scaled.detach().float())
                    step_loss += item_loss
                    cloned += bc_weight > 0.0
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                runner.post_optimizer_step(optimizer)
                check_finite_update_state(model=policy, optimizer=optimizer)
                global_step += 1
                losses.append(step_loss / len(window))
                record = {
                    "type": "grpo",
                    "round": round_index,
                    "step": global_step,
                    "loss": losses[-1],
                    "grad_norm": float(grad_norm),
                    "cloned_groups": cloned,
                }
                metrics.write(json.dumps(record) + "\n")
                if global_step % 10 == 0:
                    print(json.dumps(record), flush=True)
        metrics.flush()

        del cached
        gc.collect()
        torch.cuda.empty_cache()

        checkpoint = output_dir / f"ckpt_round{round_index + 1}.pt"
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                "step": global_step,
                "stage": "grpo",
                "round": round_index + 1,
                "parent": str(args.parent),
                "mean_loss": statistics.fmean(losses) if losses else None,
                "judge_mean": statistics.fmean(judged_scores) if judged_scores else None,
                "signal_rate": signal_rate,
            },
            checkpoint,
        )
        summary = {
            "type": "round",
            "round": round_index + 1,
            "checkpoint": str(checkpoint),
            "updates": len(losses),
            "mean_loss": round(statistics.fmean(losses), 5) if losses else None,
            "judge_mean": round(statistics.fmean(judged_scores), 2) if judged_scores else None,
            "signal_rate": round(signal_rate, 3),
            "usd": ledger.spent(),
        }
        metrics.write(json.dumps(summary) + "\n")
        metrics.flush()
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    metrics.close()
    ledger.flush()
    print(json.dumps({"done": True, "usd_total": ledger.spent()}), flush=True)


if __name__ == "__main__":
    main()
