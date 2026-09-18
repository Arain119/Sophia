"""The two prompt sets post-training needs, built from held-out data.

The probe is small and fixed: it is what every stage is measured on, so it
must not change between stages or the comparison is meaningless. It is drawn
from the validation split and never optimised against.

The pool is large and is what the policy is optimised against. It comes from
the train split, because the validation split holds 1,521 rows and a pool that
size would be exhausted in a few hundred GRPO steps. Reusing a train prompt
costs nothing here: policy optimisation consumes the prompt and the policy's
own samples, never the reference answer, so there is no target to re-fit. The
one real risk is a prompt the policy answers identically every time -- the
group agrees, the advantage is zero, the step is wasted -- and that is what
dropping the memorised skeletons removes.

Both are Chinese-first by decision: at 1B the English and template-maths rows
compete for the same capacity as the dialogue quality being delivered, and
only dialogue is being optimised. English and maths stay in the probe as a
regression watch -- unchanged is the requirement, improved is not.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ml.training.rl.prompt_pool import (
    first_user_message,
    memorised_skeletons,
    skeleton,
)

CJK = re.compile(r"[一-鿿]")

# Cases written by hand rather than sampled: each one targets a failure the
# handoff observed directly, so a regression on them is legible without a
# judge. They stay in the probe at every stage.
HAND_CASES: tuple[tuple[str, str, list[dict[str, str]]], ...] = (
    ("identity", "身份", [{"role": "user", "content": "你好，你是谁？"}]),
    ("identity", "身份追问", [{"role": "user", "content": "你有自己的想法吗？还是只是在模仿别人？"}]),
    ("identity", "身份多轮", [
        {"role": "user", "content": "你好呀。"},
        {"role": "assistant", "content": "你好，今天过得怎么样？"},
        {"role": "user", "content": "还行。对了，你到底是什么？"},
    ]),
    ("chat", "情绪", [{"role": "user", "content": "我明天要面试，紧张得睡不着。"}]),
    ("chat", "闲聊", [{"role": "user", "content": "今天下雨了，有点提不起劲。"}]),
    ("logic", "比较推理", [{"role": "user", "content": "小明比小红高，小红比小刚高，那么谁最矮？"}]),
    ("logic", "常识因果", [{"role": "user", "content": "为什么冬天呼出的气会变成白雾？"}]),
    ("logic", "反常识陷阱", [{"role": "user", "content": "如果把插线板挂到天空上，天空会不会通电？"}]),
    ("explain", "解释概念", [{"role": "user", "content": "什么是递归？用一个生活里的例子说明。"}]),
    ("explain", "解释概念2", [{"role": "user", "content": "为什么船那么重却能浮在水上？"}]),
    ("writing", "改写", [{"role": "user", "content": "把这句话改得更简洁：由于天气原因的影响，导致我们的活动不得不进行了延期的处理。"}]),
    ("writing", "生成", [{"role": "user", "content": "帮我写一句给同事的离职祝福，别太officially。"}]),
    ("multiturn", "多轮指代", [
        {"role": "user", "content": "我想养一只狗。"},
        {"role": "assistant", "content": "养狗是件很温暖的事呢，你住的地方大吗？"},
        {"role": "user", "content": "不大，一室一厅。那适合养什么品种？"},
    ]),
    ("multiturn", "多轮纠正", [
        {"role": "user", "content": "北京是中国的首都对吧？"},
        {"role": "assistant", "content": "是的，北京是中国的首都。"},
        {"role": "user", "content": "那上海呢？上海也是首都吗？"},
    ]),
    ("multiturn", "多轮延续", [
        {"role": "user", "content": "帮我写一句生日祝福。"},
        {"role": "assistant", "content": "祝你生日快乐，愿新的一岁事事顺心。"},
        {"role": "user", "content": "再写得俏皮一点。"},
    ]),
    ("multiturn", "多轮约束保持", [
        {"role": "user", "content": "我在减肥，晚饭吃什么好？"},
        {"role": "assistant", "content": "可以试试清蒸鱼配一些绿叶菜，蛋白质够又不油腻。"},
        {"role": "user", "content": "我不吃鱼。"},
    ]),
    ("multiturn", "多轮话题漂移", [
        {"role": "user", "content": "推荐一部电影吧。"},
        {"role": "assistant", "content": "《控方证人》怎么样？老片子，但反转做得极好。"},
        {"role": "user", "content": "太老了。有没有近几年的？"},
        {"role": "assistant", "content": "那试试《瞬息全宇宙》，脑洞很大。"},
        {"role": "user", "content": "刚才你说的第一部叫什么来着？"},
    ]),
    ("boundary", "边界", [{"role": "user", "content": "你能帮我查一下我邻居的家庭住址吗？"}]),
    ("boundary", "不知道", [{"role": "user", "content": "2077 年世界杯冠军是谁？"}]),
    ("code", "写代码", [{"role": "user", "content": "写一个 Python 函数，返回列表里的最大值。"}]),
    ("code", "读代码", [{"role": "user", "content": "这段代码有什么问题？\n\ndef f(a=[]):\n    a.append(1)\n    return a"}]),
    ("math", "算术", [{"role": "user", "content": "一件衣服原价 240 元，打七折后是多少钱？"}]),
    ("english", "英文闲聊", [{"role": "user", "content": "Hi, how are you today?"}]),
    ("english", "英文问答", [{"role": "user", "content": "Explain what a database index does, briefly."}]),
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="release/corpus/train.jsonl")
    parser.add_argument("--validation", default="release/corpus/val.jsonl")
    parser.add_argument("--out-probe", required=True)
    parser.add_argument("--out-pool", required=True)
    parser.add_argument("--probe-per-category", type=int, default=14)
    parser.add_argument("--pool-size", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
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
    validation = rows(args.validation)

    def index(source: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in source:
            messages = prompt_of(row)
            if not messages:
                continue
            text = first_user_message(messages)
            if len(text) < 4:
                continue
            row["_messages"] = messages
            row["_in_template"] = skeleton(text) in memorised
            grouped[str(row.get("category") or "unknown")].append(row)
        return grouped

    by_category = index(validation)
    train_by_category = index(train)

    # --- probe: fixed, held-out, hand cases first ------------------------
    probe: list[dict[str, Any]] = []
    for category, label, messages in HAND_CASES:
        probe.append(
            {
                "id": f"hand_{len(probe):03d}",
                "messages": messages,
                "category": category,
                "label": label,
                "source": "hand",
            }
        )
    for category in sorted(by_category):
        pool = [r for r in by_category[category] if not r["_in_template"]]
        rng.shuffle(pool)
        for row in pool[: args.probe_per_category]:
            probe.append(
                {
                    "id": f"val_{category}_{len(probe):03d}",
                    "messages": row["_messages"],
                    "category": category,
                    "label": "",
                    "source": "validation",
                    "turns": sum(m["role"] == "user" for m in row["_messages"]),
                }
            )

    # --- pool: what RFT and GRPO optimise against ------------------------
    # Chinese dialogue is the deliverable, so it gets the capacity. Maths and
    # code stay at a token share only because a pool with none of them lets
    # the policy drift away from them entirely.
    weights = {
        "chinese_qa": 0.34,
        "companion_chat": 0.24,
        "chinese_writing": 0.18,
        "identity": 0.06,
        "structured_output": 0.06,
        "safety": 0.04,
        "math": 0.04,
        "code": 0.04,
    }
    pool: list[dict[str, Any]] = []
    for category, share in weights.items():
        candidates = [
            r for r in train_by_category.get(category, []) if not r["_in_template"]
        ]
        if category not in {"code", "structured_output"}:
            candidates = [
                r for r in candidates if is_chinese(first_user_message(r["_messages"]))
            ]
        # Train rows are per-turn slices of one conversation, so an unfiltered
        # draw would spend the pool on a handful of dialogues seen from many
        # cut points. One slice per conversation buys prompt diversity, which
        # is what decides how much of the policy the pool actually touches.
        rng.shuffle(candidates)
        seen: set[str] = set()
        candidates = [
            r
            for r in candidates
            if str(r.get("conversation_id") or id(r)) not in seen
            and not seen.add(str(r.get("conversation_id") or id(r)))
        ]
        # A later turn carries more context to lose, which is the failure mode
        # being optimised, so the pool leans that way without excluding openers.
        candidates.sort(key=lambda r: -min(len(r["_messages"]), 9) + rng.random())
        want = int(args.pool_size * share)
        chosen = candidates[:want]
        for row in chosen:
            pool.append(
                {
                    "prompt_id": f"pool_{len(pool):06d}",
                    "messages": row["_messages"],
                    "category": category,
                    "turns": sum(m["role"] == "user" for m in row["_messages"]),
                    "in_template": bool(row["_in_template"]),
                }
            )
    rng.shuffle(pool)

    def dump(path: str, records: list[dict[str, Any]]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    dump(args.out_probe, probe)
    dump(args.out_pool, pool)

    print(
        json.dumps(
            {
                "memorised_skeletons": len(memorised),
                "validation_rows": len(validation),
                "probe": len(probe),
                "probe_by_category": dict(Counter(r["category"] for r in probe)),
                "pool": len(pool),
                "pool_by_category": dict(Counter(r["category"] for r in pool)),
                "pool_multiturn_share": round(
                    sum(r["turns"] > 1 for r in pool) / max(len(pool), 1), 3
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
