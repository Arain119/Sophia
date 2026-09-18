"""Sampling groups of completions for one prompt, and reading them back.

Both policy-optimisation stages consume the same thing: several completions of
one prompt, scored and compared against each other. DPO takes the best and the
worst of a group; GRPO centres the group on its own mean. So the unit here is
the group, not the completion -- which also happens to be the only batch the
runtime will take, since it requires every row to share a prompt length.

Completions are decoded with the special tokens left in. Skipping them merges
the reasoning block into the answer and hides whether the model stopped on its
own, and both of those are things the reward has to see.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

THINK_CLOSE = "</think>"
THINK_OPEN = "<think>"


@dataclass(frozen=True)
class RolloutPrompt:
    """One prompt, with the artefacts its answer is required to contain."""

    prompt_id: str
    messages: list[dict[str, str]]
    expectations: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RolloutPrompt:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"prompt {payload.get('prompt_id')!r} has no messages")
        return cls(
            prompt_id=str(payload.get("prompt_id") or payload.get("id") or ""),
            messages=[dict(message) for message in messages],
            expectations=dict(payload.get("expectations") or {}),
            tags=tuple(payload.get("tags") or ()),
        )


@dataclass(frozen=True)
class Completion:
    prompt_id: str
    index: int
    raw: str
    think: str
    answer: str
    new_tokens: int
    hit_eos: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "index": self.index,
            "raw": self.raw,
            "think": self.think,
            "answer": self.answer,
            "new_tokens": self.new_tokens,
            "hit_eos": self.hit_eos,
        }


def split_think(raw: str, *, eos: str = "") -> tuple[str, str]:
    """Separate the reasoning block from the answer the reader would see."""
    think, answer = "", raw
    if THINK_CLOSE in raw:
        head, _, answer = raw.partition(THINK_CLOSE)
        think = head.replace(THINK_OPEN, "").strip()
    if eos:
        answer = answer.replace(eos, "")
    return think.strip(), answer.strip()


def load_prompts(path: str | Path) -> list[RolloutPrompt]:
    prompts: list[RolloutPrompt] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid prompt row at line {number}") from exc
            prompts.append(RolloutPrompt.from_payload(payload))
    if not prompts:
        raise ValueError(f"prompt pool is empty: {path}")
    return prompts


def group_has_signal(rewards: Sequence[float], *, min_std: float = 0.05) -> bool:
    """Whether a group can contribute a gradient at all.

    GRPO centres each group on its own mean, so a group whose completions all
    earn the same reward produces an advantage of exactly zero for every token
    in it. Sampling more of those is not a small inefficiency; it is the whole
    step spent on nothing. Prompts inside the memorised template set behave
    exactly this way, which is why the pool excludes them.
    """
    if len(rewards) < 2:
        return False
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    return variance**0.5 >= float(min_std)


def rollout_group(
    prompt: RolloutPrompt,
    *,
    generate_fn: Callable[[list[dict[str, str]], int], list[tuple[str, int, bool]]],
    group_size: int,
) -> list[Completion]:
    """Sample ``group_size`` completions of one prompt.

    ``generate_fn`` returns (decoded_text, new_token_count, hit_eos) per row.
    It is injected so that grouping, parsing and scoring can be exercised
    without a model on the device.
    """
    if int(group_size) < 2:
        raise ValueError("a rollout group needs at least two completions to compare")
    rows = generate_fn(prompt.messages, int(group_size))
    if len(rows) != int(group_size):
        raise RuntimeError(
            f"generator returned {len(rows)} rows for a group of {group_size}"
        )
    completions: list[Completion] = []
    for index, (raw, new_tokens, hit_eos) in enumerate(rows):
        think, answer = split_think(raw)
        completions.append(
            Completion(
                prompt_id=prompt.prompt_id,
                index=index,
                raw=raw,
                think=think,
                answer=answer,
                new_tokens=int(new_tokens),
                hit_eos=bool(hit_eos),
            )
        )
    return completions


def write_groups(
    path: str | Path,
    groups: Iterable[tuple[RolloutPrompt, list[Completion]]],
) -> int:
    """Append rollout groups as one JSON object per prompt."""
    written = 0
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        for prompt, completions in groups:
            handle.write(
                json.dumps(
                    {
                        "prompt_id": prompt.prompt_id,
                        "messages": prompt.messages,
                        "expectations": prompt.expectations,
                        "tags": list(prompt.tags),
                        "completions": [c.to_payload() for c in completions],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written


def read_groups(path: str | Path) -> Iterator[tuple[RolloutPrompt, list[Completion]]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt = RolloutPrompt.from_payload(payload)
            completions = [
                Completion(
                    prompt_id=prompt.prompt_id,
                    index=int(row.get("index", position)),
                    raw=str(row.get("raw", "")),
                    think=str(row.get("think", "")),
                    answer=str(row.get("answer", "")),
                    new_tokens=int(row.get("new_tokens", 0)),
                    hit_eos=bool(row.get("hit_eos", False)),
                )
                for position, row in enumerate(payload.get("completions") or [])
            ]
            yield prompt, completions


def build_model_generator(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 0,
    max_new_tokens: int = 256,
) -> Callable[[list[dict[str, str]], int], list[tuple[str, int, bool]]]:
    """A generate_fn backed by the runtime.

    Temperature is deliberately high: a group whose completions agree carries
    no gradient, so diversity within the group is the point rather than a cost.
    """
    from ml.runtime.controls import maybe_reset_runtime_cache
    from ml.runtime.inference.eval.common import torch
    from ml.runtime.inference.eval.generation_runtime import generate_with_kv_cache
    from ml.runtime.inference.eval.runtime import build_inputs_from_conversation

    eos_id = int(tokenizer.eos_token_id)
    eos_text = tokenizer.decode([eos_id], skip_special_tokens=False)

    def generate_fn(
        messages: list[dict[str, str]], group_size: int
    ) -> list[tuple[str, int, bool]]:
        inputs = build_inputs_from_conversation(
            tokenizer=tokenizer,
            conversation=[dict(message) for message in messages],
        )
        input_ids = inputs["input_ids"].to(device)
        # Every row of the group is the same prompt, so the batch is already
        # rectangular -- the one shape the runtime accepts.
        batch = input_ids.expand(int(group_size), -1).contiguous()
        prompt_len = int(batch.shape[1])
        maybe_reset_runtime_cache(model)
        with torch.inference_mode():
            generated = generate_with_kv_cache(
                model=model,
                input_ids=batch,
                attention_mask=None,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                eos_token_id=eos_id,
                max_cache_len=prompt_len + int(max_new_tokens) + 8,
            )
        rows: list[tuple[str, int, bool]] = []
        for row in generated:
            tokens = [int(t) for t in row[prompt_len:].detach().cpu().tolist()]
            hit_eos = eos_id in tokens
            if hit_eos:
                tokens = tokens[: tokens.index(eos_id) + 1]
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            if eos_text and text.endswith(eos_text):
                text = text[: -len(eos_text)]
            rows.append((text, len(tokens), hit_eos))
        return rows

    return generate_fn


__all__ = [
    "Completion",
    "RolloutPrompt",
    "build_model_generator",
    "group_has_signal",
    "load_prompts",
    "read_groups",
    "rollout_group",
    "split_think",
    "write_groups",
]
