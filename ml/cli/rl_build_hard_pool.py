"""A pool of prompts the policy actually fails, rather than ones it already passes.

The first pool was drawn by preferring late conversation turns, on the theory
that a deep turn carries more context to lose. What a deep turn actually
carries is "谢谢你" -- half that pool was closing pleasantries with a median
user message of fourteen characters, the policy scored 85.9 on it, and
best-of-8 selection learned to say "不客气" well.

The failures are somewhere else entirely. Read off the probe: buoyancy
explained as water molecules swimming on the surface, recursion explained as
an input filter, a code fence containing Chinese prose. They are all the same
shape -- a question with a mechanism in it, answered in the right register
with the wrong content. So this pool keeps only turns that ask for something,
and adds a distilled set aimed squarely at mechanism questions, where the
corpus is thin and the policy is worst.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ml.training.rl.judge import Judge, Ledger
from ml.training.rl.prompt_pool import memorised_skeletons, skeleton

CJK = re.compile(r"[一-鿿]")
CLOSING = re.compile(
    r"谢谢|多谢|感谢|好的|好嘞|明白了|懂了|晚安|睡了|再见|拜拜|辛苦了|收到|"
    r"行吧|就这样|没问题了|我去试试|太好了|不错|嗯嗯|哈哈"
)
# A turn that asks for something has one of these in it.
ASKING = re.compile(
    r"[?？]|为什么|怎么|如何|什么是|是什么|解释|说明|介绍|推荐|帮我|写一|给我|"
    r"能不能|可不可以|怎样|哪些|区别|原理|办法|建议"
)

SEED_INSTRUCTION = """你要为一个中文对话数据集生成**提问**，用来测试一个小模型的逻辑连贯性。

生成 {n} 条中文问题，要求：
- 每条都是一个**需要解释机制、因果或原理**的问题，或者一个**需要交付具体东西**的请求。
- 覆盖日常生活常识、自然现象、身体健康、社会经济、技术原理、语言文字等领域。
- 口语化，像真人会问的，长度 10-35 字。
- 不要出现"谢谢""好的"这类客套，不要是闲聊。
- 不要编号，不要解释，每行一条，只输出问题本身。

主题方向：{topic}"""

TOPICS = [
    "自然现象与天气", "厨房与食物", "身体与健康常识", "家用电器与工具原理",
    "交通工具与出行", "钱与经济常识", "动物与植物", "天文与地理",
    "计算机与网络原理", "语言文字与成语", "历史与社会习俗", "心理与情绪机制",
    "材料与日用品", "运动与锻炼", "睡眠与作息", "居家维修与生活技巧",
]


def is_chinese(text: str) -> bool:
    return len(CJK.findall(text)) >= max(4, len(text) * 0.15)


def prompt_of(row: dict[str, Any]) -> list[dict[str, str]] | None:
    messages = [
        {"role": str(m.get("role")), "content": str(m.get("content") or "")}
        for m in row.get("messages") or []
        if m.get("role") in {"user", "assistant"}
    ]
    while messages and messages[-1]["role"] != "user":
        messages.pop()
    return messages or None


def distil_prompts(judge: Judge, *, want: int, seed: int) -> list[str]:
    """Ask the judge model for mechanism questions, the gap the corpus has."""
    rng = random.Random(seed)
    batches = [(topic, 32) for topic in TOPICS for _ in range(max(want // (32 * len(TOPICS)), 1))]

    def one(job: tuple[str, int]) -> list[str]:
        topic, count = job
        text = judge._call(
            [
                {
                    "role": "user",
                    "content": SEED_INSTRUCTION.format(n=count, topic=topic),
                }
            ],
            max_tokens=1400,
        )
        if not text:
            return []
        out = []
        for line in text.splitlines():
            line = re.sub(r"^\s*[\d.、)\-*]+\s*", "", line).strip()
            if 8 <= len(line) <= 60 and is_chinese(line) and not CLOSING.match(line):
                out.append(line)
        return out

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(one, batches))
    seen: set[str] = set()
    flat: list[str] = []
    for group in results:
        for line in group:
            key = skeleton(line)
            if key in seen:
                continue
            seen.add(key)
            flat.append(line)
    rng.shuffle(flat)
    return flat[:want]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="release/corpus/train.jsonl")
    parser.add_argument("--out-pool", required=True)
    parser.add_argument("--corpus-prompts", type=int, default=9000)
    parser.add_argument("--distilled-prompts", type=int, default=2500)
    parser.add_argument("--min-user-chars", type=int, default=18)
    parser.add_argument("--ledger", default="ops/rl/ledger.json")
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    def rows(path: str) -> list[dict[str, Any]]:
        out = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    out.append(json.loads(line))
        return out

    train = rows(args.train)
    memorised = memorised_skeletons(train, threshold=50)

    kept: list[dict[str, Any]] = []
    seen_conversations: set[str] = set()
    rng.shuffle(train)
    for row in train:
        messages = prompt_of(row)
        if not messages:
            continue
        last = messages[-1]["content"].strip()
        if len(last) < int(args.min_user_chars):
            continue
        if CLOSING.search(last) and len(last) < 40:
            continue
        if not ASKING.search(last):
            continue
        if not is_chinese(last):
            continue
        if skeleton(last) in memorised:
            continue
        conversation = str(row.get("conversation_id") or "")
        if conversation in seen_conversations:
            continue
        seen_conversations.add(conversation)
        kept.append(
            {
                "prompt_id": f"hard_{len(kept):06d}",
                "messages": messages,
                "category": str(row.get("category") or "unknown"),
                "turns": sum(m["role"] == "user" for m in messages),
                "source": "corpus",
            }
        )
        if len(kept) >= int(args.corpus_prompts):
            break

    distilled: list[dict[str, Any]] = []
    if int(args.distilled_prompts) > 0:
        judge = Judge(ledger=Ledger(args.ledger, args.budget), workers=16)
        for text in distil_prompts(judge, want=int(args.distilled_prompts), seed=args.seed):
            distilled.append(
                {
                    "prompt_id": f"distil_{len(distilled):06d}",
                    "messages": [{"role": "user", "content": text}],
                    "category": "mechanism",
                    "turns": 1,
                    "source": "distilled",
                }
            )
        judge.ledger.flush()
        print(json.dumps({"distilled": len(distilled), "usd": judge.ledger.spent()}))

    pool = kept + distilled
    rng.shuffle(pool)
    Path(args.out_pool).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_pool).open("w", encoding="utf-8", newline="\n") as handle:
        for row in pool:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lengths = sorted(len(r["messages"][-1]["content"]) for r in pool)
    print(
        json.dumps(
            {
                "pool": len(pool),
                "by_source": dict(Counter(r["source"] for r in pool)),
                "by_category": dict(Counter(r["category"] for r in pool)),
                "multi_turn_share": round(
                    sum(r["turns"] > 1 for r in pool) / max(len(pool), 1), 3
                ),
                "last_user_chars_p25_p50_p75": [
                    lengths[len(lengths) // 4],
                    lengths[len(lengths) // 2],
                    lengths[3 * len(lengths) // 4],
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
