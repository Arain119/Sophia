"""Hold real multi-turn conversations with the policy and write them down unedited.

A fixed script of follow-ups measures whether the model can continue a
conversation someone else was having. It cannot measure whether the model can
hold one, because the script does not react: it asks turn three regardless of
how turn two went, so context loss produces a transcript that still looks
plausible line by line.

So the user side is played by the judge model, told to behave like a person
rather than an assistant. It reacts to what was actually said, which is what
makes dropped constraints and contradictions surface -- a real interlocutor
notices being told to forget the fish, and asks again.

Everything sampled is written out. Nothing is filtered for the pack: the whole
point of handing Arain the transcripts is that the selection is not mine.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ml.training.rl.engine import RolloutEngine, encode_prompt, load_policy
from ml.training.rl.judge import Judge, Ledger

USER_SIM = """你在和一个叫 Sophia 的 AI 聊天。你要扮演的是**一个真实的普通人**，不是助手。

规则：
- 每次只说一句话或两句话，口语化，像在微信上打字。
- 你要**真的读** Sophia 刚才说的内容，然后自然地接下去：好奇就追问，
  觉得她说得不对就直说，她答非所问就抱怨，她说得好就顺着聊。
- 不要夸她，不要总结，不要说"好的我明白了"这种助手腔。
- 偶尔可以换个相关的话题，或者提一个和前面聊过的内容有关的新要求。
- 绝对不要扮演 Sophia，也不要替她说话。只输出你自己要说的那句话，不要加引号，不要加"用户："。"""

SEEDS = [
    "我最近老是失眠，你有什么办法吗",
    "帮我想个周末两天的短途旅行计划",
    "你觉得人为什么会怀旧",
    "我想学做饭，从哪道菜开始比较好",
    "解释一下什么是通货膨胀，用大白话",
    "我和室友因为卫生问题吵架了，怎么办",
    "推荐几本适合睡前看的书",
    "帮我写一段自我介绍，我要去面试产品经理",
    "为什么猫喜欢待在纸箱里",
    "我想养点好养活的植物，推荐一下",
    "你会觉得孤独吗",
    "帮我把这句话写得好听点：我不想去参加同学聚会",
    "跑步和游泳哪个更适合减肥",
    "给我讲讲相对论到底在说什么",
    "我妈总是催我结婚，怎么回她比较好",
    "写一个 Python 脚本，把文件夹里所有图片改名",
    "最近工作特别累，感觉没意义",
    "手机总是没电，是电池坏了吗",
    "帮我起一个咖啡店的名字",
    "为什么有的人怎么吃都不胖",
    "我想开始写日记，但总是坚持不下来",
    "解释一下什么叫内卷",
    "帮我写一封请假邮件，我要请三天",
    "你最喜欢什么颜色，为什么",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conversations", type=int, default=24)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--label", default="")
    parser.add_argument("--ledger", default="ops/rl/ledger.json")
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", default="claude-haiku-4-5")
    parser.add_argument("--prefill", default="")
    args = parser.parse_args()

    # Prompt and reply share one 4096-token window; reserve the reply.
    prompt_budget = 4096 - int(args.max_new_tokens) - 8

    rng = random.Random(args.seed)
    seeds = list(SEEDS)
    rng.shuffle(seeds)
    seeds = (seeds * (args.conversations // len(seeds) + 1))[: args.conversations]

    model, tokenizer, meta = load_policy(
        checkpoint=args.checkpoint, batch_size=max(args.conversations, 1)
    )
    engine = RolloutEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch=max(args.conversations, 1),
        max_new_tokens=args.max_new_tokens,
    )
    judge = Judge(
        ledger=Ledger(args.ledger, args.budget, model=args.judge_model),
        workers=32,
        model=args.judge_model,
    )
    prefill = tokenizer.encode(args.prefill, add_special_tokens=False) if args.prefill else []

    histories: list[list[dict[str, str]]] = [
        [{"role": "user", "content": seed}] for seed in seeds
    ]
    started = time.time()

    for turn in range(int(args.turns)):
        requests = [
            (f"c{index:03d}", encode_prompt(tokenizer, history, max_prompt_tokens=prompt_budget) + prefill, 1)
            for index, history in enumerate(histories)
        ]
        produced = engine.run(
            requests,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
        )
        for index, history in enumerate(histories):
            sample = produced[f"c{index:03d}"][0]
            history.append(
                {
                    "role": "assistant",
                    "content": (
                        "<think>" + sample.think + "</think>" + sample.answer
                        if sample.think
                        else sample.answer
                    ),
                    "_answer": sample.answer,
                    "_think": sample.think,
                    "_hit_eos": sample.hit_eos,
                }
            )
        print(
            f"turn {turn + 1}/{args.turns} done ({time.time() - started:.0f}s)",
            flush=True,
        )
        if turn == int(args.turns) - 1:
            break

        def next_user(history: list[dict[str, str]]) -> str:
            transcript = []
            for message in history:
                who = "我" if message["role"] == "user" else "Sophia"
                body = message.get("_answer", message["content"])
                transcript.append(who + "：" + body)
            reply = judge._call(
                [
                    {"role": "system", "content": USER_SIM},
                    {
                        "role": "user",
                        "content": "\n".join(transcript) + "\n\n（现在轮到你说话了）",
                    },
                ],
                max_tokens=600,
            )
            return (reply or "嗯，然后呢？").strip().strip('"').split("\n")[0][:120]

        with ThreadPoolExecutor(max_workers=32) as pool:
            replies = list(pool.map(next_user, histories))
        for history, reply in zip(histories, replies, strict=False):
            history.append({"role": "user", "content": reply})

    # Score every assistant turn so the pack carries numbers as well as text.
    tasks, positions = [], []
    for index, history in enumerate(histories):
        for position, message in enumerate(history):
            if message["role"] != "assistant":
                continue
            tasks.append(
                {
                    "conversation": [
                        {"role": m["role"], "content": m.get("_answer", m["content"])}
                        for m in history[:position]
                    ],
                    "answer": message.get("_answer", ""),
                    "think": message.get("_think", ""),
                }
            )
            positions.append((index, position))
    verdicts = judge.score_many(tasks)
    for (index, position), verdict in zip(positions, verdicts, strict=False):
        histories[index][position]["_score"] = verdict.score if verdict.ok else None
        histories[index][position]["_flags"] = list(verdict.flags)

    scores = [v.score for v in verdicts if v.ok]
    summary = {
        "checkpoint": args.checkpoint,
        "label": args.label,
        "step": meta.get("step"),
        "conversations": len(histories),
        "turns": args.turns,
        "temperature": args.temperature,
        "assistant_turns_scored": len(scores),
        "judge_mean": round(sum(scores) / len(scores), 2) if scores else None,
        "pass_70": round(sum(s >= 70 for s in scores) / len(scores), 4) if scores else None,
        "broken_below_50": round(sum(s < 50 for s in scores) / len(scores), 4) if scores else None,
        "usd": judge.ledger.spent(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "conversations": histories}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# Sophia 多轮对话样本包", "", "> " + json.dumps(summary, ensure_ascii=False), ""]
    for index, history in enumerate(histories):
        lines.append(f"## 对话 {index + 1}")
        lines.append("")
        for message in history:
            if message["role"] == "user":
                lines.append("**用户：** " + message["content"])
            else:
                score = message.get("_score")
                flags = ",".join(message.get("_flags") or []) or "-"
                lines.append(
                    "**Sophia：** " + (message.get("_answer") or "（空）")
                )
                lines.append("")
                lines.append(
                    "<sub>判分 {} ｜ flags {} ｜ {}</sub>".format(
                        f"{score:.0f}" if score is not None else "n/a",
                        flags,
                        "自然结束" if message.get("_hit_eos") else "未自然结束",
                    )
                )
            lines.append("")
        lines.append("---")
        lines.append("")
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("WROTE " + str(output.with_suffix(".md")), flush=True)


if __name__ == "__main__":
    main()
