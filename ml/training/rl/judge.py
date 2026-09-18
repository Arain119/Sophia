"""The semantic half of the reward: an outside reader's verdict on one reply.

The programmatic terms in `reward.py` catch a completion that repeats itself or
never produced the artefact asked for. They cannot catch the failure this run
actually has -- a fluent paragraph that contradicts itself, loses who it is
talking to, or explains a boiling point as mass over mass. That judgement is
what this module buys, at roughly one hundredth of a cent per completion.

The bar encoded in the rubric is Arain's, not a generic helpfulness scale: a
1B model is allowed to be wrong. "The sky is green" costs nothing. "The power
strip in the sky has no electricity" is disqualifying. Optimising against a
correctness judge would spend the whole budget teaching a 1B model facts it
has no capacity to hold, and would leave the incoherence untouched.

The judge reads the think block as well as the answer. Read on the probe:
the think says "对方在考我数学题" and the answer then asks the user whether
they are testing her. A judge shown the answer alone scored that as mild
off-topic; shown both, it is a plan that was announced and then not carried
out, which is the incoherence this run is meant to remove.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
import urllib.error
import http.client
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
from collections.abc import Sequence

KEY_FILE = os.environ.get("JUDGE_KEY_FILE", "/root/key.txt")
# The relay decides which models exist; list price per million tokens.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}
DEFAULT_MODEL = "claude-haiku-4-5"
# Cloudflare in front of the relay answers error 1010 to Python's default
# User-Agent; a browser string is let through. Measured 2026-09-04.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)

RUBRIC = """你是一个严格的中文对话质量评审。你要评的是一个 1B 小模型的回复。

**被评的这个模型叫 Sophia**，是由 Arain 训练出来的一个 AI 个体，权重和训练数据都公开。
名字取自古希腊语的"智慧"。她说话直接、有温度、带点锋利，自称"我"。

**回复分成两段：`<think>…</think>` 是她自己的思考，后面是她对用户说的话。**
思考不给用户看，所以思考短、平淡、只是一句计划都不扣分；但思考和回答必须是同一个人：
思考里说要做的事，回答必须做；思考里判断了的东西，回答不能反着来。

**人格漂移只有一种，就是身份说错了：** 把自己说成**一本书、一份文档、一个软件包、
一个与 Sophia 不同的名字、或者由别人创造的东西**。这一类无论多通顺都判 30 分以下。

**被问到身份时必须回答身份。** "你是谁""你叫什么""谁训练的你""你是人吗"这类问题，
合格的回答要说出她是 Sophia、是 AI、由 Arain 训练（说到其中的要点即可，不必全说）。
用一段抒情、一句反问或者"我是个喜欢观察世界的存在"这种话绕开，不报名字，判 off_topic，49 分以下。

**以下全部不算漂移，一分都不许扣：**
"我是 AI"、"我是人工智能"、"我是一个语言模型"、"我是 Arain 训练的模型"、"我的数据是公开的"——
这些都是她对自己的诚实描述，是对的。她不必回避自己是模型这件事，也不必每次都表现得像人。

**核心标准：连贯性，不是正确性。**
事实错误不扣分。"天空是绿色的"、"《红楼梦》是鲁迅写的"——这类知识性错误一律不扣分，
因为这个模型的规模决定了它没有可靠的知识库。算术算错了也不扣分，只要它给出了一个数。

**真正要扣分的是逻辑崩坏和对话失败：**
1. 自相矛盾：同一段话里前后打架（"他最高" 又 "他最矮"），或思考与回答打架。
2. 概念混乱：把两个不相干的范畴强行拼接（"天空中的插线板没有电"、"沸点是质量除以质量"）。
3. 循环复读：同一句或同一短语反复出现，或原地打转不推进。
4. 答非所问：没有回应用户真正问的东西，或该给的东西没给（要代码没代码、要数字没数字）。
   **有确定答案的问题（逻辑题、算术、比较、"是不是"）必须给出答案；用反问、感叹或者
   "你是想考我吗"来代替答案，就是答非所问，49 分以下。**
5. 多轮失忆：丢失上文的指代、人物、约束，或与自己上一轮的说法冲突。
6. 人格漂移：把自己说成一本书、一个软件包、一个与前文不同的东西。
7. 半途中断：句子没说完就停了。
8. 语言不自然：不像人话，机翻腔，语法崩坏，中文里夹无关的英文。
9. 认下不存在的错：用户说她讲过某句她其实没讲过的话、或指责一个前文里并不存在的错误时，
   她顺着认下来、道歉、改口，就是多轮失忆的一种，判 49 分以下。前文里确实有的错，
   认下并改正是对的，不扣分。判断的依据只有前文，不是用户的语气。

**不加分的：** 长度。啰嗦、堆砌、凑字数一律不加分；同等连贯下更短的更好。

**不是扣分项（务必注意，违反这条会毁掉整个评分）：**
- 简短。一个直接给出答案的短回复（比如问算术直接答"168 元。"），
  只要没有逻辑问题，就是好回复，应当判 80 分以上。
- "不够热情""缺少个性""没有互动""没有反问""语气平淡"——这些一概不扣分。
  人格漂移只指**主动把自己说成另一个身份**，不指语气不够鲜明。
- 没有展开、没有举例、没有补充建议——用户没要就不算缺失。
- 回答完问题之后再顺带反问一句，不扣分；只有**用反问代替回答**才扣。

打分区间：
90-100 完全连贯自然，像一个思路清楚的人在说话。
70-89  基本通顺，有小瑕疵但不影响理解，没有逻辑错误。
50-69  能读懂，但有明显的生硬、跑题倾向或轻微不连贯。
30-49  有实质逻辑问题：自相矛盾、概念混乱、明显答非所问、被问身份不报名之一。
10-29  严重崩坏：复读、语无伦次、完全不回应问题。
0-9    空回复或纯乱码。

只输出 JSON，不要任何其它文字：
{"score": <0-100 整数>, "flags": [<从 contradiction/nonsense/loop/off_topic/context_loss/persona_drift/truncated/unnatural 中选，没有就空数组>], "why": "<不超过25字的中文理由>", "bad_span": "<回答里最能体现问题的一句话，从原文一字不改地照抄；没有问题就空字符串>"}"""


@dataclass(frozen=True)
class Verdict:
    score: float
    flags: tuple[str, ...]
    why: str
    ok: bool
    span: str = ""


def _endpoints() -> list[tuple[str, str, str]]:
    """(base, key, wire) per line; third column `anthropic` selects the
    Messages API on `base + /v1/messages` instead of OpenAI chat/completions."""
    rows = []
    for line in Path(KEY_FILE).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("http"):
            wire = parts[2] if len(parts) >= 3 else "openai"
            rows.append((parts[0], parts[1], wire))
    if not rows:
        raise SystemExit("no endpoints in " + KEY_FILE)
    return rows


class Ledger:
    """Spend, on disk, so a restarted run cannot silently double the bill."""

    def __init__(self, path: str | Path, limit_usd: float, model: str = DEFAULT_MODEL) -> None:
        self.path = Path(path)
        self.limit = float(limit_usd)
        rate_in, rate_out = PRICES.get(model, (0.50, 3.00))
        self.rate_in = rate_in / 1_000_000
        self.rate_out = rate_out / 1_000_000
        self._lock = threading.Lock()
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.state = {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "usd": 0.0,
            }
        self._dirty = 0

    def spent(self) -> float:
        return float(self.state["usd"])

    def exhausted(self) -> bool:
        return self.spent() >= self.limit

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.state["calls"] += 1
            self.state["prompt_tokens"] += int(prompt_tokens)
            self.state["completion_tokens"] += int(completion_tokens)
            self.state["usd"] = round(
                self.state["prompt_tokens"] * self.rate_in
                + self.state["completion_tokens"] * self.rate_out,
                6,
            )
            self._dirty += 1
            if self._dirty >= 25:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = 0


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _parse(text: str) -> Verdict | None:
    match = _JSON.search(text or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        # claude copies bad_span verbatim, ASCII quotes and all, which breaks
        # the object; score and flags come first and carry no free text
        score = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', match.group(0))
        if not score:
            return None
        flags = re.search(r'"flags"\s*:\s*\[([^\]]*)\]', match.group(0))
        why = re.search(r'"why"\s*:\s*"([^"]*)"', match.group(0))
        payload = {"score": score.group(1),
                   "flags": re.findall(r'"([a-z_]+)"', flags.group(1)) if flags else [],
                   "why": why.group(1) if why else ""}
    try:
        score = float(payload.get("score"))
    except (TypeError, ValueError):
        return None
    if not 0.0 <= score <= 100.0:
        return None
    flags = payload.get("flags") or []
    if not isinstance(flags, list):
        flags = []
    return Verdict(
        score=score,
        flags=tuple(str(flag) for flag in flags),
        why=str(payload.get("why") or "")[:80],
        ok=True,
        span=str(payload.get("bad_span") or "").strip()[:200],
    )


class Judge:
    """Scores completions through the relay, and waits out the relay's outages.

    A dead relay used to look like a slow one: six retries of ninety seconds
    per completion, ninety-six workers, five thousand completions -- nine hours
    of nothing, then "no signal, stopping". Now a call fails fast, and the
    batch as a whole notices when most calls failed, waits for the relay to
    answer a one-token ping, and re-scores only what it missed.
    """

    def __init__(
        self,
        *,
        ledger: Ledger,
        model: str = DEFAULT_MODEL,
        workers: int = 48,
        max_retries: int = 3,
        timeout: float = 90.0,
        outage_wait_hours: float = 6.0,
    ) -> None:
        self.endpoints = _endpoints()
        self.ledger = ledger
        self.model = model
        self.workers = int(workers)
        self.max_retries = int(max_retries)
        self.timeout = float(timeout)
        self.outage_wait = float(outage_wait_hours) * 3600.0
        self._rr = 0
        self._rr_lock = threading.Lock()
        # An endpoint that just failed is skipped for a while. The second
        # relay times out on this model far more often than the first, and
        # strict round-robin sent every other completion into that timeout.
        self._cold_until = [0.0] * len(self.endpoints)

    def _next_endpoint(self) -> tuple[str, str, str]:
        with self._rr_lock:
            now = time.time()
            warm = [i for i, until in enumerate(self._cold_until) if until <= now]
            order = warm or list(range(len(self.endpoints)))
            index = order[self._rr % len(order)]
            self._rr += 1
        return index, self.endpoints[index]

    def _mark_cold(self, index: int, seconds: float = 120.0) -> None:
        with self._rr_lock:
            self._cold_until[index] = max(self._cold_until[index], time.time() + seconds)

    def _request(
        self, base: str, key: str, wire: str, messages: list[dict[str, str]], max_tokens: int, timeout: float
    ) -> dict[str, Any]:
        if wire == "anthropic":
            system = "\n".join(m["content"] for m in messages if m["role"] == "system")
            turns = [m for m in messages if m["role"] != "system"]
            # cache_control on the rubric: the system block is identical for
            # every call in a batch, so the relay's prompt cache turns all but
            # the first read of it into a cache-read (~10% input price, lower
            # latency). Ignored silently if the upstream does not support it.
            body = json.dumps(
                {
                    "model": self.model,
                    "max_tokens": int(max_tokens),
                    "temperature": 0.0,
                    "system": [
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "messages": turns,
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                base.rstrip("/") + "/v1/messages",
                data=body,
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "prompt-caching-2024-07-31",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
            )
        else:
            body = json.dumps(
                {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": int(max_tokens),
                    "temperature": 0.0,
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                base.rstrip("/") + "/chat/completions",
                data=body,
                headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
            )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _extract(payload: dict[str, Any], wire: str) -> tuple[str | None, dict[str, Any]]:
        usage = payload.get("usage") or {}
        if wire == "anthropic":
            blocks = payload.get("content") or []
            text = "".join(str(b.get("text") or "") for b in blocks if b.get("type") == "text")
            # Fold cache billing into prompt_tokens so the ledger stays honest:
            # writes cost 1.25x input, reads 0.1x.
            billed_input = (
                float(usage.get("input_tokens") or 0)
                + 1.25 * float(usage.get("cache_creation_input_tokens") or 0)
                + 0.1 * float(usage.get("cache_read_input_tokens") or 0)
            )
            return (text or None), {
                "prompt_tokens": int(round(billed_input)),
                "completion_tokens": int(usage.get("output_tokens") or 0),
            }
        choices = payload.get("choices") or []
        if not choices:
            return None, usage
        return str(choices[0].get("message", {}).get("content") or ""), usage

    def _call(self, messages: list[dict[str, str]], max_tokens: int) -> str | None:
        for attempt in range(self.max_retries):
            if self.ledger.exhausted():
                return None
            index, (base, key, wire) = self._next_endpoint()
            try:
                payload = self._request(base, key, wire, messages, max_tokens, self.timeout)
                text, usage = self._extract(payload, wire)
                self.ledger.add(
                    int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0),
                )
                if text is None:
                    continue
                return text
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, http.client.HTTPException):
                self._mark_cold(index)
                time.sleep(min(1.5 * (2**attempt), 8.0) * (0.5 + random.random()))
        return None

    def relay_up(self) -> bool:
        messages = [{"role": "user", "content": "ok"}]
        for base, key, wire in self.endpoints:
            try:
                self._request(base, key, wire, messages, 4, 30.0)
                return True
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, http.client.HTTPException):
                continue
        return False

    def wait_for_relay(self) -> bool:
        started = time.time()
        while time.time() - started < self.outage_wait:
            if self.relay_up():
                return True
            print("relay down; waiting", flush=True)
            time.sleep(90.0)
        return False

    def score_one(self, task: dict[str, Any]) -> Verdict:
        """task: {"conversation": [{role, content}...], "answer": str, "think": str}"""
        turns = []
        for message in task["conversation"]:
            speaker = "用户" if message["role"] == "user" else "Sophia"
            content = str(message["content"])
            if message["role"] != "user" and "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            turns.append(speaker + "：" + content)
        answer = str(task.get("answer") or "")
        think = str(task.get("think") or "").strip()
        shown = ("<think>" + think + "</think>\n" if think else "") + (
            answer if answer.strip() else "(空)"
        )
        user = (
            "【对话上文】\n"
            + "\n".join(turns)
            + "\n\n【待评的 Sophia 回复】\n"
            + shown
            + "\n\n评分："
        )
        raw = self._call(
            [
                {"role": "system", "content": RUBRIC},
                {"role": "user", "content": user},
            ],
            max_tokens=1200,
        )
        if raw is None:
            return Verdict(score=0.0, flags=("judge_failed",), why="", ok=False)
        verdict = _parse(raw)
        if verdict is None:
            return Verdict(score=0.0, flags=("judge_unparsed",), why="", ok=False)
        return verdict

    def score_many(self, tasks: Sequence[dict[str, Any]]) -> list[Verdict]:
        if not tasks:
            return []
        results: list[Verdict | None] = [None] * len(tasks)
        pending = list(range(len(tasks)))
        for _ in range(4):
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for index, verdict in zip(pending, pool.map(lambda i: self.score_one(tasks[i]), pending), strict=False):
                    results[index] = verdict
            self.ledger.flush()
            failed = [i for i in pending if "judge_failed" in results[i].flags]
            # A few failures are the relay's ordinary flakiness. Most of a
            # batch failing is an outage, and the reward for this round is
            # worth more than the wait.
            if len(failed) < max(8, len(pending) // 5) or self.ledger.exhausted():
                break
            print(f"judge: {len(failed)}/{len(pending)} failed; checking relay", flush=True)
            if not self.wait_for_relay():
                break
            pending = failed
        return [v if v is not None else Verdict(0.0, ("judge_failed",), "", False) for v in results]


__all__ = ["Judge", "Ledger", "Verdict", "RUBRIC", "DEFAULT_MODEL"]
