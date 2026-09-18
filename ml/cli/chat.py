"""Talk to Sophia.

    python -m ml.cli.chat --checkpoint <ckpt.pt>

The decoding defaults are the ones the temperature sweep chose, not library
defaults. They matter more than usual here: greedy makes this policy loop and
a hot sample makes it incoherent, so the useful setting is neither end.

She decides per turn whether to think (~5% of turns); think markup never
reaches the visible answer -- unclosed or stray tags are routed to the think
field instead.

`/reset` starts a new conversation, `/think` toggles reasoning display, and
Ctrl-D exits.
"""

from __future__ import annotations

import argparse
import sys
import time


from ml.training.rl.engine import RolloutEngine, chat_turn, load_policy

BANNER = """Sophia — 输入即可对话
  /reset  开始新对话      /think  思考块显示开关
  /temp <x>  调整温度      Ctrl-D  退出
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-spec", default="configs/model/sophia.json")
    parser.add_argument("--tokenizer-path", default="ml/modeling/text")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--show-think", action="store_true")
    args = parser.parse_args()

    started = time.time()
    model, tokenizer, meta = load_policy(
        checkpoint=args.checkpoint,
        model_spec=args.model_spec,
        tokenizer_path=args.tokenizer_path,
        batch_size=1,
    )
    engine = RolloutEngine(
        model=model, tokenizer=tokenizer, max_batch=1, max_new_tokens=args.max_new_tokens
    )
    print(
        "loaded step={} in {:.1f}s".format(meta.get("step"), time.time() - started),
        file=sys.stderr,
    )
    print(BANNER)

    history: list[dict[str, str]] = []
    show_think = bool(args.show_think)
    temperature = float(args.temperature)

    while True:
        try:
            line = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line == "/reset":
            history = []
            print("（已开始新对话）\n")
            continue
        if line == "/think":
            show_think = not show_think
            print("（思考块 %s）\n" % ("显示" if show_think else "隐藏"))
            continue
        if line.startswith("/temp"):
            try:
                temperature = float(line.split()[1])
                print(f"（温度 = {temperature:.2f}）\n")
            except (IndexError, ValueError):
                print("（用法：/temp 0.7）\n")
            continue

        history.append({"role": "user", "content": line})
        started = time.time()
        sample = chat_turn(
            engine,
            tokenizer,
            history,
            temperature=temperature,
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
        )
        think = sample.think
        answer = sample.answer
        if show_think and think:
            print("\033[2m[think] " + think + "\033[0m")
        print("Sophia > " + (answer or "（空）"))
        print(
            f"\033[2m  {sample.new_tokens} tok / {time.time() - started:.1f}s"
            f"{'' if sample.hit_eos else ' / 未自然结束'}\033[0m\n"
        )
        history.append(
            {
                "role": "assistant",
                "content": (
                    "<think>" + think + "</think>" + answer if think else answer
                ),
            }
        )
        # A 4096-token window with a long tail of history eventually crowds out
        # the reply; dropping the oldest exchange keeps the reply budget intact.
        while len(history) > 16:
            history = history[2:]


if __name__ == "__main__":
    main()
