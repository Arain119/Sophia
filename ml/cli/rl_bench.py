"""How fast does this runtime decode, as a function of batch size?

The probe measured 23 tokens/s at batch 1 and 41 at batch 2, which is the
difference between an RL budget that buys four rollout passes and one that
buys a third of a pass. Decode is normally bound by streaming the weights, so
a wider batch should be nearly free until it is not -- but "should be" is not
a number, and every plan after this one is priced off it.

Groups are rectangular by construction (one prompt, K samples), so the batch
sizes here are the ones a rollout can actually reach.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from ml.training.rl.engine import RolloutEngine, encode_prompt, load_policy, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--batches", default="1,8,16,32,64,96")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    sizes = [int(x) for x in args.batches.split(",")]
    largest = max(sizes)
    probe = read_jsonl(args.probe)
    model, tokenizer, meta = load_policy(
        checkpoint=args.checkpoint, batch_size=largest
    )
    engine = RolloutEngine(
        model=model, tokenizer=tokenizer, max_batch=largest, max_new_tokens=args.max_new_tokens
    )
    # One mid-length single-turn prompt, so every batch does the same work.
    row = next(r for r in probe if r.get("source") == "hand")
    tokens = encode_prompt(tokenizer, row["messages"], max_prompt_tokens=1024)
    print(json.dumps({"meta": meta, "prompt_tokens": len(tokens)}), flush=True)

    for size in sizes:
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        produced = engine.run(
            [("b", tokens, size)],
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=int(args.max_new_tokens),
        )
        elapsed = time.time() - started
        generated = sum(s.new_tokens for s in produced["b"])
        print(
            json.dumps(
                {
                    "batch": size,
                    "seconds": round(elapsed, 1),
                    "tokens": generated,
                    "tokens_per_second": round(generated / elapsed, 1),
                    "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
