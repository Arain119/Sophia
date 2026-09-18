"""Greedy CPU decode of two checkpoints on the same prompts.

The card is fully committed to training and peaks within 1.6 GB of the limit,
so anything that touches it risks killing a round that has already cost an
hour. The box has 208 cores sitting idle behind an 8-thread trainer; this
trades wall time for zero risk to the run.
"""
import argparse
import json
import time
import torch
from ml.training.rl.engine import load_policy, encode_prompt, RolloutEngine

WANT = [
    "filter_values",
    "什么是递归",
    "船那么重",
    "我不吃鱼",
    "小明比小红高",
    "第一部叫什么",
]

def pick(probe_path):
    rows, chosen = [], {}
    for line in open(probe_path, encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    for r in rows:
        text = " ".join(m["content"] for m in r["messages"] if m["role"] == "user")
        for w in WANT:
            if w in text and w not in chosen:
                chosen[w] = r
    return [chosen[w] for w in WANT if w in chosen]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(180)
    picks = pick(args.probe)
    model, tokenizer, meta = load_policy(
        checkpoint=args.checkpoint, batch_size=1,
        device="cpu", dtype=torch.float32,
    )
    engine = RolloutEngine(
        model=model, tokenizer=tokenizer, device="cpu",
        max_batch=1, max_new_tokens=args.max_new_tokens,
        prefill_positions=110_000,
    )
    out = []
    for row in picks:
        tokens = encode_prompt(tokenizer, row["messages"], max_prompt_tokens=640)
        t0 = time.time()
        produced = engine.run(
            [(row["id"], tokens, 1)], temperature=0.0, top_p=1.0,
        )
        sample = produced[row["id"]][0]
        secs = time.time() - t0
        user = " ".join(m["content"] for m in row["messages"] if m["role"] == "user")
        rec = {
            "label": args.label, "id": row["id"], "user": user[-200:],
            "answer": sample.answer, "new_tokens": sample.new_tokens,
            "hit_eos": bool(sample.hit_eos), "seconds": round(secs, 1),
        }
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
