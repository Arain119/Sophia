"""Measure a checkpoint before spending GPU hours on it.

Two things are unknown at the start of the post-training stage and both are
cheap to find out: how fast this runtime actually samples, and where on the
temperature axis the policy is least bad. The handoff reports greedy decoding
looping and sampled decoding going incoherent, which are the two ends of the
same curve; the useful question is what the curve looks like in between, and
whether a checkpoint earlier than the last one sits higher on it.

Everything here is measurement. Nothing is trained.
"""

from __future__ import annotations

import argparse
import json
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
from ml.training.rl.judge import DEFAULT_MODEL, Judge, Ledger
from ml.training.rl.reward import distinct_ngram_ratio, loop_coverage


def summarise(
    samples: list[Any],
    verdicts: list[Any] | None,
) -> dict[str, Any]:
    loops = [loop_coverage(s.answer) for s in samples]
    distinct = [distinct_ngram_ratio(s.answer) for s in samples]
    lengths = [len(s.answer) for s in samples]
    row: dict[str, Any] = {
        "n": len(samples),
        "hit_eos": round(sum(s.hit_eos for s in samples) / max(len(samples), 1), 4),
        "loop_mean": round(statistics.fmean(loops), 4),
        "loop_gt_25pct": round(sum(x > 0.25 for x in loops) / max(len(loops), 1), 4),
        "distinct4_mean": round(statistics.fmean(distinct), 4),
        "empty": round(sum(not s.answer.strip() for s in samples) / max(len(samples), 1), 4),
        "answer_chars_median": int(statistics.median(lengths)) if lengths else 0,
        "new_tokens_mean": round(statistics.fmean([s.new_tokens for s in samples]), 1),
    }
    if verdicts:
        scores = [v.score for v in verdicts if v.ok]
        flags: dict[str, int] = {}
        for verdict in verdicts:
            for flag in verdict.flags:
                flags[flag] = flags.get(flag, 0) + 1
        row["judge_n"] = len(scores)
        row["judge_mean"] = round(statistics.fmean(scores), 2) if scores else None
        row["judge_median"] = round(statistics.median(scores), 2) if scores else None
        row["pass_70"] = (
            round(sum(s >= 70 for s in scores) / len(scores), 4) if scores else None
        )
        row["broken_below_50"] = (
            round(sum(s < 50 for s in scores) / len(scores), 4) if scores else None
        )
        row["flags"] = dict(sorted(flags.items(), key=lambda kv: -kv[1]))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True, help="jsonl with {id, messages}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperatures", default="0,0.4,0.7,0.9")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--ledger", default="ops/rl/ledger.json")
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=16,
        help="relay concurrency; measured 16 clean and 32 into a 429 storm",
    )
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prefill", default="", help="text appended after the assistant tag, e.g. an empty think block")
    args = parser.parse_args()

    # Prompt and reply share one 4096-token window; reserve the reply.
    prompt_budget = 4096 - int(args.max_new_tokens) - 8

    probe = read_jsonl(args.probe)
    if args.limit:
        probe = probe[: args.limit]

    model, tokenizer, meta = load_policy(
        checkpoint=args.checkpoint, batch_size=args.max_batch
    )
    engine = RolloutEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch=args.max_batch,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(meta, ensure_ascii=False), flush=True)

    prefill = tokenizer.encode(args.prefill, add_special_tokens=False) if args.prefill else []
    encoded = {
        str(row["id"]): encode_prompt(tokenizer, row["messages"], max_prompt_tokens=prompt_budget) + prefill
        for row in probe
    }
    by_id = {str(row["id"]): row for row in probe}

    judge = None
    if args.judge:
        judge = Judge(
            ledger=Ledger(args.ledger, args.budget, model=args.judge_model),
            workers=int(args.judge_workers),
            model=args.judge_model,
        )

    report: dict[str, Any] = {"checkpoint": args.checkpoint, "meta": meta, "rows": []}
    dumps: list[dict[str, Any]] = []

    for raw_temperature in str(args.temperatures).split(","):
        temperature = float(raw_temperature)
        count = 1 if temperature == 0.0 else int(args.samples)
        requests = [(key, tokens, count) for key, tokens in encoded.items()]
        started = time.time()
        produced = engine.run(
            requests,
            temperature=temperature,
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
        )
        elapsed = time.time() - started
        flat = [s for key in produced for s in produced[key]]
        generated_tokens = sum(s.new_tokens for s in flat)

        verdicts = None
        if judge is not None:
            tasks = [
                {"conversation": by_id[s.key]["messages"], "answer": s.answer, "think": s.think}
                for s in flat
            ]
            verdicts = judge.score_many(tasks)

        row = summarise(flat, verdicts)
        row["temperature"] = temperature
        row["wall_seconds"] = round(elapsed, 1)
        row["tokens_per_second"] = round(generated_tokens / max(elapsed, 1e-6), 1)
        report["rows"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        for position, sample in enumerate(flat):
            record = {
                "temperature": temperature,
                "id": sample.key,
                "index": sample.index,
                "messages": by_id[sample.key]["messages"],
                "think": sample.think,
                "answer": sample.answer,
                "hit_eos": sample.hit_eos,
                "new_tokens": sample.new_tokens,
            }
            if verdicts:
                record["score"] = verdicts[position].score
                record["flags"] = list(verdicts[position].flags)
                record["why"] = verdicts[position].why
            dumps.append(record)

    if judge is not None:
        judge.ledger.flush()
        report["usd_spent_total"] = judge.ledger.spent()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with Path(args.output).with_suffix(".samples.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for record in dumps:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("WROTE " + args.output, flush=True)


if __name__ == "__main__":
    main()
