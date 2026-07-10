"""Shared helpers for task-owned capability reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ml.core.common.io import write_json_atomic
from ml.training.posttrain.data import ChatExample


def metadata_family(metadata: dict[str, Any]) -> str:
    family = str(
        metadata.get("capability_family")
        or metadata.get("family")
        or metadata.get("final_raw_family")
        or "unknown"
    ).strip()
    return family or "unknown"


def metadata_bucket(metadata: dict[str, Any]) -> str:
    bucket = str(metadata.get("bucket") or metadata.get("source_bucket") or "unknown").strip()
    return bucket or "unknown"


def safe_mean(total: float, count: int) -> float:
    if int(count) <= 0:
        return 0.0
    return float(total) / float(count)


def write_json(path: str, payload: dict[str, Any]) -> None:
    write_json_atomic(str(path), payload, sort_keys=True)


def limit_examples(
    *,
    split: str,
    examples: list[ChatExample],
    max_examples_per_split: int,
) -> list[ChatExample]:
    limit = max(int(max_examples_per_split), 0)
    if limit <= 0 or len(examples) <= limit:
        print(
            f"[INFO] capability report split={split} examples={len(examples)}",
            flush=True,
        )
        return list(examples)
    print(
        f"[INFO] capability report split={split} examples={len(examples)} "
        f"capped_to={limit}",
        flush=True,
    )
    return list(examples[:limit])


def maybe_log_progress(
    *,
    kind: str,
    split: str,
    index: int,
    total: int,
    progress_every: int,
) -> None:
    every = max(int(progress_every), 0)
    if every <= 0:
        return
    if index != total and index % every != 0:
        return
    print(
        f"[INFO] {kind} capability report split={split} progress={index}/{total}",
        flush=True,
    )


def numeric_reward_details(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    details = value.get("details")
    if not isinstance(details, dict):
        return out
    for key, item in details.items():
        if isinstance(item, (int, float)):
            out[str(key)] = float(item)
    return out


@dataclass
class SupervisedStats:
    rows: int = 0
    supervised_tokens: int = 0
    loss_sum: float = 0.0

    def update(self, *, loss_sum: float, supervised_tokens: int) -> None:
        self.rows += 1
        self.supervised_tokens += int(supervised_tokens)
        self.loss_sum += float(loss_sum)

    def payload(self) -> dict[str, Any]:
        return {
            "rows": int(self.rows),
            "supervised_tokens": int(self.supervised_tokens),
            "loss_mean": safe_mean(float(self.loss_sum), int(self.supervised_tokens)),
        }


@dataclass
class RewardStats:
    rows: int = 0
    reward_sum: float = 0.0
    completion_chars_sum: int = 0
    reward_details_sum: dict[str, float] = field(default_factory=dict)

    def update(
        self,
        *,
        reward: float,
        completion_chars: int,
        reward_details: dict[str, float],
    ) -> None:
        self.rows += 1
        self.reward_sum += float(reward)
        self.completion_chars_sum += int(completion_chars)
        for key, value in reward_details.items():
            self.reward_details_sum[str(key)] = float(
                self.reward_details_sum.get(str(key), 0.0)
            ) + float(value)

    def payload(self) -> dict[str, Any]:
        details = {
            key: safe_mean(float(value), int(self.rows))
            for key, value in sorted(self.reward_details_sum.items())
        }
        return {
            "rows": int(self.rows),
            "reward_mean": safe_mean(float(self.reward_sum), int(self.rows)),
            "completion_chars_mean": safe_mean(
                float(self.completion_chars_sum),
                int(self.rows),
            ),
            "reward_details_mean": details,
        }
