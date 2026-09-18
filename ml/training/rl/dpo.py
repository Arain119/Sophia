"""Small, explicit DPO primitives used by the post-SFT alignment stage.

The policy and reference model score the same rendered sequence.  Only
assistant tokens contribute, and their log probability is averaged over the
response tokens so a longer answer is not preferred merely for having more
terms in the sum.  Reference scores are intentionally passed in as tensors:
the reference is frozen and can be scored once, then released before policy
training to keep a 1B run inside the 32 GB device budget.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional


IGNORE_INDEX = -100


@dataclass(frozen=True)
class PreferencePair:
    """One prompt with a judged preferred and rejected completion."""

    prompt_id: str
    messages: list[dict[str, Any]]
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    source: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "messages": self.messages,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_score": float(self.chosen_score),
            "rejected_score": float(self.rejected_score),
            "source": self.source,
        }


def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return causal next-token log probabilities with shape ``[B, T-1]``."""

    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits must be [B,T,V] and input_ids must be [B,T]")
    if logits.size(0) != input_ids.size(0) or logits.size(1) != input_ids.size(1):
        raise ValueError("logits and input_ids must have the same batch and length")
    if input_ids.size(1) < 2:
        return logits.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    # Accumulate in fp32 even when the model runs in bf16.  This is material
    # for the small margin that DPO uses to distinguish two fluent answers.
    log_probs = functional.log_softmax(logits[:, :-1].float(), dim=-1)
    return log_probs.gather(-1, input_ids[:, 1:].long().unsqueeze(-1)).squeeze(-1)


def sequence_logprob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    normalize: bool = True,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score assistant tokens and return ``(logprob, token_count)`` per row.

    ``labels`` uses the same positions as SFT labels.  The causal shift is
    therefore applied to both logits and labels, exactly matching the SFT
    cross-entropy objective.
    """

    if labels.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("labels must be [B,T] and match input_ids")
    values = token_logprobs(logits, input_ids)
    if values.size(1) == 0:
        count = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)
        return values.sum(dim=1), count
    mask = labels[:, 1:].ne(int(ignore_index))
    count = mask.sum(dim=1).to(dtype=torch.long)
    summed = (values * mask.to(dtype=values.dtype)).sum(dim=1)
    if not normalize:
        return summed, count
    return summed / count.clamp_min(1).to(dtype=summed.dtype), count


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    """The stable Bradley-Terry DPO objective, averaged over a batch."""

    if float(beta) <= 0.0:
        raise ValueError("DPO beta must be positive")
    tensors = (policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    if any(value.ndim != 1 for value in tensors):
        raise ValueError("DPO log probabilities must be one-dimensional batches")
    shape = tensors[0].shape
    if any(value.shape != shape for value in tensors[1:]):
        raise ValueError("DPO log probability batches must have the same shape")
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected
    logits = float(beta) * (policy_margin - reference_margin)
    return functional.softplus(-logits).mean()


def dpo_accuracy(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
) -> torch.Tensor:
    """Fraction of pairs whose policy margin is positive after the reference."""

    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected
    return (policy_margin - reference_margin > 0).to(dtype=torch.float32).mean()


def read_preference_pairs(path: str | Path) -> Iterator[PreferencePair]:
    """Read one JSON preference pair per line."""

    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"preference row {line_number} is not an object")
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"preference row {line_number} has no messages")
            chosen = str(payload.get("chosen") or "")
            rejected = str(payload.get("rejected") or "")
            if not chosen or not rejected or chosen == rejected:
                raise ValueError(f"preference row {line_number} has invalid completions")
            yield PreferencePair(
                prompt_id=str(payload.get("prompt_id") or f"row_{line_number:06d}"),
                messages=[dict(message) for message in messages],
                chosen=chosen,
                rejected=rejected,
                chosen_score=float(payload.get("chosen_score", 0.0)),
                rejected_score=float(payload.get("rejected_score", 0.0)),
                source=str(payload.get("source") or ""),
            )


def write_preference_pairs(path: str | Path, pairs: Iterable[PreferencePair]) -> int:
    """Write a newline-delimited preference file and return its row count."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.to_payload(), ensure_ascii=False) + "\n")
            written += 1
    if written == 0:
        raise ValueError("cannot write an empty preference set")
    return written


__all__ = [
    "IGNORE_INDEX",
    "PreferencePair",
    "dpo_accuracy",
    "dpo_loss",
    "read_preference_pairs",
    "sequence_logprob",
    "token_logprobs",
    "write_preference_pairs",
]
