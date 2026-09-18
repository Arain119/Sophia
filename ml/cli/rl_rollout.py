"""Sample the policy against the pool and attach a reward to every completion.

This is the expensive half of post-training and the only part that both RFT
and GRPO need, so it is a stage that writes to disk rather than a step inside
a training loop. Having the scored rollouts on disk means the selection rule
can be changed, or the judge re-run on a subset, without paying for generation
again -- and it means a crashed trainer resumes from data rather than from the
GPU.

The reward is the programmatic floor from `reward.py` plus the judge's 0-100
coherence score mapped into [0, 1]. Judge failures are dropped, not scored
zero: an API timeout is not evidence about the completion, and feeding it back
as the worst possible reward would be pure noise pointed straight at whatever
the policy happened to say.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

from ml.training.rl.engine import (
    RolloutEngine,
    encode_prompt,
    load_policy,
    read_jsonl,
)
from concurrent.futures import ThreadPoolExecutor

from ml.training.rl.judge import DEFAULT_MODEL, Judge, Ledger
from ml.training.rl.reward import score_completion
from ml.training.rl.prompt_pool import derive_expectations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts", type=int, default=4000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=640,
        help="cap prompt history so rollout batches are wide enough to be worth running",
    )
    parser.add_argument("--max-batch", type=int, default=48)
    parser.add_argument("--chunk", type=int, default=200, help="prompts per flush")
    parser.add_argument("--ledger", default="ops/rl/ledger.json")
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--judge-workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--prefill",
        default="<think></think>",
        help="half of every group is sampled with this after the assistant tag; "
        "empty string samples every completion the same way",
    )
    parser.add_argument(
        "--rejudge-top",
        type=int,
        default=2,
        help="re-score the top-k of each group and rank on the mean of both "
        "judgments, so the argmax is not the judge's noise",
    )
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

    rng = random.Random(args.seed)
    pool = read_jsonl(args.pool)
    rng.shuffle(pool)
    pool = pool[args.offset : args.offset + args.prompts]

    model, tokenizer, meta = load_policy(
        checkpoint=args.checkpoint, batch_size=args.max_batch
    )
    engine = RolloutEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch=args.max_batch,
        max_new_tokens=args.max_new_tokens,
    )
    ledger = Ledger(args.ledger, args.budget, model=args.judge_model)
    judge = Judge(ledger=ledger, workers=args.judge_workers, model=args.judge_model)
    prefill = tokenizer.encode(args.prefill, add_special_tokens=False) if args.prefill else []
    print(json.dumps({"policy": meta, "prompts": len(pool)}, ensure_ascii=False), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = output.open("w", encoding="utf-8", newline="\n")
    started = time.time()
    written = 0
    all_scores: list[float] = []
    signal_groups = 0

    # Encode once, then order the whole pool by prompt length, so a chunk spans
    # a narrow band and its batches pad to close to their own width. Chunking at
    # all is what keeps a crash from losing the whole run.
    encoded_pool = [
        (
            str(row["prompt_id"]),
            encode_prompt(tokenizer, row["messages"], max_prompt_tokens=prompt_budget),
            row,
        )
        for row in pool
    ]
    encoded_pool.sort(key=lambda item: len(item[1]))
    print(
        json.dumps(
            {
                "encoded": len(encoded_pool),
                "prompt_len_median": len(encoded_pool[len(encoded_pool) // 2][1]),
                "prompt_len_max": len(encoded_pool[-1][1]),
            }
        ),
        flush=True,
    )

    # Judging is network-bound and generation is GPU-bound, so a chunk is handed
    # to a worker thread and settled after the next chunk has been generated.
    judge_pool = ThreadPoolExecutor(max_workers=1)
    pending: tuple[Any, ...] | None = None

    def settle(item: tuple[Any, ...]) -> None:
        nonlocal written, signal_groups
        by_id, flat, future, gen_seconds, done = item
        wait_started = time.time()
        verdicts = future.result()
        judge_seconds = time.time() - wait_started

        grouped: dict[str, list[dict[str, Any]]] = {}
        for (prompt_id, sample, mode), verdict in zip(flat, verdicts, strict=False):
            expectations = derive_expectations(by_id[prompt_id])
            breakdown = score_completion(
                sample.answer,
                coherence=(verdict.score / 100.0 if verdict.ok else 0.0),
                hit_eos=sample.hit_eos,
                expectations=expectations,
            )
            grouped.setdefault(prompt_id, []).append(
                {
                    "index": sample.index + (1000 if mode == "none" else 0),
                    "think_mode": mode,
                    "think": sample.think,
                    "answer": sample.answer,
                    "hit_eos": sample.hit_eos,
                    "new_tokens": sample.new_tokens,
                    "judge_ok": verdict.ok,
                    "judge_score": verdict.score if verdict.ok else None,
                    "flags": list(verdict.flags),
                    "why": verdict.why,
                    "repetition": round(breakdown.repetition, 4),
                    "responsiveness": round(breakdown.responsiveness, 4),
                    "reward": round(breakdown.total, 4) if verdict.ok else None,
                }
            )
            if verdict.ok:
                all_scores.append(verdict.score)

        if int(args.rejudge_top) > 0:
            picks: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for prompt_id, samples in grouped.items():
                ranked = sorted(
                    (s for s in samples if s["reward"] is not None),
                    key=lambda s: -float(s["reward"]),
                )[: int(args.rejudge_top)]
                for s in ranked:
                    picks.append(
                        (
                            s,
                            {
                                "conversation": by_id[prompt_id]["messages"],
                                "answer": s["answer"],
                                "think": s["think"],
                            },
                        )
                    )
            second = judge.score_many([task for _, task in picks])
            for (s, _), verdict in zip(picks, second, strict=False):
                if not verdict.ok:
                    continue
                first = float(s["judge_score"])
                mean_score = round((first + verdict.score) / 2, 1)
                s["judge_score_2"] = verdict.score
                s["judge_score"] = mean_score
                s["flags"] = sorted(set(s["flags"]) | set(verdict.flags))
                # coherence enters the reward as judge/100 with weight 1
                s["reward"] = round(float(s["reward"]) + (mean_score - first) / 100.0, 4)

        for prompt_id, samples in grouped.items():
            rewards = [s["reward"] for s in samples if s["reward"] is not None]
            has_signal = (
                len(rewards) >= 2
                and statistics.pstdev(rewards) >= 0.05
            )
            signal_groups += bool(has_signal)
            handle.write(
                json.dumps(
                    {
                        "prompt_id": prompt_id,
                        "messages": by_id[prompt_id]["messages"],
                        "category": by_id[prompt_id].get("category"),
                        "role": by_id[prompt_id].get("role"),
                        "move": by_id[prompt_id].get("move"),
                        "turns": by_id[prompt_id].get("turns"),
                        "has_signal": bool(has_signal),
                        "samples": sorted(samples, key=lambda s: s["index"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
        handle.flush()

        rate = done / max(time.time() - started, 1e-6)
        print(
            json.dumps(
                {
                    "prompts_done": done,
                    "of": len(pool),
                    "gen_s": round(gen_seconds, 1),
                    # What the judge still costs after overlapping with the next
                    # chunk's generation, not what it costs in total.
                    "judge_wait_s": round(judge_seconds, 1),
                    "prompts_per_s": round(rate, 3),
                    "eta_min": round((len(pool) - done) / max(rate, 1e-9) / 60, 1),
                    "judge_mean": round(statistics.fmean(all_scores), 2) if all_scores else None,
                    "signal_rate": round(signal_groups / max(written, 1), 3),
                    "usd": ledger.spent(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    for chunk_start in range(0, len(encoded_pool), args.chunk):
        window = encoded_pool[chunk_start : chunk_start + args.chunk]
        chunk = [row for _, _, row in window]
        by_id = {str(row["prompt_id"]): row for row in chunk}
        requests = []
        for prompt_id, tokens, row in window:
            count = int(row.get("samples") or args.group_size)
            if prefill:
                requests.append((prompt_id, tokens, count - count // 2))
                requests.append((prompt_id + "#nt", tokens + prefill, count // 2))
            else:
                requests.append((prompt_id, tokens, count))
        gen_started = time.time()
        produced = engine.run(
            requests,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
        )
        gen_seconds = time.time() - gen_started

        tasks: list[dict[str, Any]] = []
        flat: list[Any] = []
        for key, samples in produced.items():
            mode = key.endswith("#nt")
            prompt_id = key[:-3] if mode else key
            for sample in samples:
                flat.append((prompt_id, sample, "none" if mode else "think"))
                tasks.append(
                    {
                        "conversation": by_id[prompt_id]["messages"],
                        "answer": sample.answer,
                        "think": sample.think,
                    }
                )
        future = judge_pool.submit(judge.score_many, tasks)
        if pending is not None:
            settle(pending)
        pending = (by_id, flat, future, gen_seconds, chunk_start + len(chunk))
        if ledger.exhausted():
            print("BUDGET EXHAUSTED - stopping rollout", flush=True)
            break

    if pending is not None:
        settle(pending)
    judge_pool.shutdown()

    handle.close()
    ledger.flush()
    print(
        json.dumps(
            {
                "groups_written": written,
                "signal_rate": round(signal_groups / max(written, 1), 3),
                "judge_mean": round(statistics.fmean(all_scores), 2) if all_scores else None,
                "usd_total": ledger.spent(),
                "wall_min": round((time.time() - started) / 60, 1),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
