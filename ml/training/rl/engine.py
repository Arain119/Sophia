"""Sampling many completions from the policy, as fast as this runtime allows.

Decode here is launch-bound, not bandwidth-bound: 7.9 tok/s at batch 1 against
991 at batch 96, with peak memory flat near 4.9 GB. Batch width is therefore the
whole story, and it used to be capped by exact prompt length -- a KDA layer
carries a recurrent state per row, so padding had to be reproduced inside the
state update.

It is now. A pad position forces the decay gate to 1 and beta to 0, which makes
the state update the identity, and MLA masks the pad keys; both are exact rather
than approximate, and `parity_leftpad` holds them to the unpadded result. So the
batch is bounded by memory instead of by length coincidence, and rows are sorted
by length so that a batch pads to close to its own width.
"""

from __future__ import annotations

import contextlib
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence

import torch

from ml.runtime.controls import maybe_reset_runtime_cache
from ml.runtime.generation.decode import sample_token_from_last_logits
from ml.runtime.model.attention import set_pad_mask
from ml.runtime.resolve import resolve_runtime_lock
from ml.runtime.inference.eval.runtime import build_inputs_from_conversation
from ml.training.pretrain.model_setup import load_tokenizer
from ml.training.sft.trainer import _model_from_spec

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class Sample:
    key: str
    index: int
    tokens: tuple[int, ...]
    raw: str
    think: str
    answer: str
    hit_eos: bool

    @property
    def new_tokens(self) -> int:
        return len(self.tokens)


def load_policy(
    *,
    checkpoint: str | Path,
    model_spec: str | Path = "configs/model/sophia.json",
    tokenizer_path: str | Path = "ml/modeling/text",
    batch_size: int = 32,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Any, Any, dict[str, Any]]:
    tokenizer = load_tokenizer(str(tokenizer_path))
    model = _model_from_spec(
        Path(model_spec), tokenizer=tokenizer, batch_size=int(batch_size)
    )
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state = payload.get("model") if isinstance(payload, dict) else None
    if state is None:
        raise SystemExit(f"checkpoint has no model state: {checkpoint}")
    missing, unexpected = model.load_state_dict(state, strict=True)
    meta = {
        "checkpoint": str(checkpoint),
        "step": payload.get("step"),
        "missing": len(missing),
        "unexpected": len(unexpected),
    }
    del payload, state
    model.to(device=torch.device(device), dtype=dtype)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, meta


def encode_prompt(
    tokenizer: Any,
    conversation: Sequence[dict[str, str]],
    *,
    max_prompt_tokens: int = 0,
) -> list[int]:
    """Encode a conversation, dropping its oldest turns until the reply fits.

    A 4096-token window shared between prompt and reply means a long enough
    history leaves no room to answer, and the runtime raises rather than
    truncating. Dropping whole leading exchanges keeps the chat markup intact,
    which slicing the token sequence would not: a prompt cut mid-turn trains
    and samples a conversation shape the model has never seen.
    """
    messages = [dict(m) for m in conversation]
    while True:
        inputs = build_inputs_from_conversation(
            tokenizer=tokenizer, conversation=messages
        )
        tokens = [int(t) for t in inputs["input_ids"][0].tolist()]
        if max_prompt_tokens <= 0 or len(tokens) <= int(max_prompt_tokens):
            return tokens
        # Keep the final user turn: it is the thing being answered.
        if len(messages) <= 1:
            return tokens[-int(max_prompt_tokens) :]
        messages = messages[1:]
        while len(messages) > 1 and messages[0]["role"] != "user":
            messages = messages[1:]


def split_think(raw: str) -> tuple[str, str]:
    if THINK_CLOSE not in raw:
        return "", raw.strip()
    head, _, answer = raw.partition(THINK_CLOSE)
    return head.replace(THINK_OPEN, "").strip(), answer.strip()


def chat_turn(
    engine: RolloutEngine,
    tokenizer: Any,
    conversation: Sequence[dict[str, str]],
    *,
    temperature: float = 0.7,
    top_p: float = 0.92,
    max_new_tokens: int = 320,
    ctx: int = 4096,
) -> Sample:
    tokens = encode_prompt(
        tokenizer, conversation, max_prompt_tokens=ctx - max_new_tokens - 8
    )
    sample = engine.run(
        [("chat", tokens, 1)],
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )["chat"][0]
    think = (sample.think or "").strip()
    answer = sample.answer or ""
    # Think markup must never reach the visible answer: an unclosed block
    # becomes the whole answer after split_think, and stray tags inside a
    # normal answer are just noise. Route both to the think field.
    if THINK_OPEN in answer:
        head, _, tail = answer.partition(THINK_OPEN)
        think = (head + tail).strip() or think
        answer = ""
    return replace(
        sample,
        think=think,
        answer=answer.replace(THINK_OPEN, "").replace(THINK_CLOSE, "").strip(),
    )


class RolloutEngine:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        max_batch: int = 32,
        max_new_tokens: int = 320,
        prefill_positions: int = 110_000,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_batch = int(max_batch)
        self.max_new_tokens = int(max_new_tokens)
        # Decode cost per step is nearly independent of batch, so the batch
        # wants to be as wide as possible -- but prefill materialises
        # [batch, prompt_len, hidden] activations, and that product is what
        # runs the card out of memory. Bounding positions rather than rows
        # lets short prompts keep the wide batch that makes them cheap while
        # long ones narrow automatically.
        self.prefill_positions = int(prefill_positions)
        self.eos_id = int(tokenizer.eos_token_id)
        self.eos_text = tokenizer.decode([self.eos_id], skip_special_tokens=False)

    def _decode(self, key: str, index: int, tokens: list[int]) -> Sample:
        hit_eos = self.eos_id in tokens
        if hit_eos:
            tokens = tokens[: tokens.index(self.eos_id)]
        raw = self.tokenizer.decode(tokens, skip_special_tokens=False)
        if self.eos_text:
            raw = raw.replace(self.eos_text, "")
        think, answer = split_think(raw)
        return Sample(
            key=key,
            index=index,
            tokens=tuple(tokens),
            raw=raw,
            think=think,
            answer=answer,
            hit_eos=hit_eos,
        )

    def _generate_padded(
        self,
        prompt_rows: list[list[int]],
        *,
        budget: int,
        temperature: float,
        top_p: float,
        top_k: int,
        eos_id: int | None = None,
    ) -> list[list[int]]:
        """One left-padded batch of unequal prompts, prefilled then decoded.

        Left padding puts every row's last real token at the same index, so the
        prefill's last position is the sampling position for all of them and no
        per-row bookkeeping is needed.  Padding exists only in the prompt, so the
        mask is dropped once the prefill is done.

        `replay_with_cache` is the same entry prefill.py and decode.py use, so the
        wrapper keeps owning the recurrent, conv and latent caches.
        """
        model = self.model
        eos = self.eos_id if eos_id is None else int(eos_id)
        rows = len(prompt_rows)
        width = max(len(row) for row in prompt_rows)
        ids = torch.full((rows, width), eos, dtype=torch.long, device=self.device)
        mask = torch.zeros((rows, width), dtype=torch.bool, device=self.device)
        for index, row in enumerate(prompt_rows):
            ids[index, width - len(row):] = torch.tensor(row, dtype=torch.long, device=self.device)
            mask[index, width - len(row):] = True

        produced: list[list[int]] = [[] for _ in range(rows)]
        done = torch.zeros(rows, dtype=torch.bool, device=self.device)
        lock = resolve_runtime_lock(model)
        with (lock if lock is not None else contextlib.nullcontext()):
            maybe_reset_runtime_cache(model)
            set_pad_mask(model, mask)
            try:
                with torch.inference_mode():
                    logits, _ = model.replay_with_cache(ids, start_pos=0, return_all_logits=False)
                    for step in range(int(budget)):
                        nxt = sample_token_from_last_logits(
                            logits, temperature=float(temperature), top_k=int(top_k), top_p=float(top_p)
                        )
                        token = nxt.view(-1)
                        for index in range(rows):
                            if not bool(done[index]):
                                produced[index].append(int(token[index]))
                        done |= token.eq(eos)
                        if bool(done.all()):
                            break
                        logits, _ = model.replay_with_cache(
                            nxt.view(rows, 1), start_pos=width + step, return_all_logits=False
                        )
            finally:
                set_pad_mask(model, None)
        return [list(prompt_rows[i]) + produced[i] for i in range(rows)]

    def _generate(
        self,
        prompt_rows: list[list[int]],
        *,
        budget: int,
        temperature: float,
        top_p: float,
        top_k: int,
        eos_id: int | None = None,
    ) -> list[list[int]]:
        """Generate one rectangular batch, splitting it if the card says no.

        The position bound is a prediction, and predictions about peak memory
        are wrong at the margin -- a bucket of unusually long prompts, or a
        fragmented allocator, can still overflow. Halving and retrying costs
        one wasted forward pass; the alternative is losing an hour of rollout
        that has already been paid for.
        """
        rows = [prompt_rows]
        out: list[list[int]] = []
        while rows:
            batch = rows.pop(0)
            try:
                out.extend(
                    self._generate_padded(
                        batch,
                        budget=int(budget),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        top_k=int(top_k),
                        eos_id=eos_id,
                    )
                )
            except torch.OutOfMemoryError:
                if len(batch) == 1:
                    raise
                torch.cuda.empty_cache()
                middle = len(batch) // 2
                rows[:0] = [batch[:middle], batch[middle:]]
        return out

    def run(
        self,
        requests: Sequence[tuple[str, list[int], int]],
        *,
        temperature: float,
        top_p: float = 0.95,
        top_k: int = 0,
        max_new_tokens: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, list[Sample]]:
        """requests: (key, prompt_token_ids, how_many_samples) -> samples by key."""
        budget = int(max_new_tokens or self.max_new_tokens)
        rows: list[tuple[str, int, tuple[int, ...]]] = []
        for key, tokens, count in requests:
            for index in range(int(count)):
                rows.append((key, index, tuple(tokens)))

        out: dict[str, list[Sample]] = defaultdict(list)
        done = 0
        for chunk, generated in self._batched(
            rows, budget=budget, temperature=temperature, top_p=top_p, top_k=top_k, eos_id=None
        ):
            for (key, index, prompt), produced in zip(chunk, generated, strict=False):
                out[key].append(self._decode(key, index, list(produced)[len(prompt) :]))
            done += len(chunk)
            if progress is not None:
                progress(done, len(rows))
        for key in out:
            out[key].sort(key=lambda s: s.index)
        return dict(out)

    def _batched(self, rows, *, budget: int, temperature: float, top_p: float, top_k: int = 0, eos_id: int | None = None):
        # Sorted by length, so each batch pads to close to its own width and the
        # position bound is priced off the longest row actually in it.
        ordered = sorted(rows, key=lambda row: len(row[2]))
        start = 0
        while start < len(ordered):
            longest = len(ordered[min(start + self.max_batch, len(ordered)) - 1][2])
            width = max(1, min(self.max_batch, self.prefill_positions // max(longest, 1)))
            chunk = ordered[start : start + width]
            produced = self._generate([list(r[2]) for r in chunk], budget=budget, temperature=temperature, top_p=top_p, top_k=top_k, eos_id=eos_id)
            yield chunk, produced
            start += len(chunk)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


__all__ = [
    "RolloutEngine",
    "Sample",
    "chat_turn",
    "encode_prompt",
    "load_policy",
    "read_jsonl",
    "split_think",
]
