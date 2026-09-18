"""Export every SFT think-bearing and masked assistant message for inspection."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_THINK = re.compile(r"<think>(?P<body>.*?)</think>", re.DOTALL | re.IGNORECASE)


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _short(value: object, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


def export(*, rows_path: Path, seeds_path: Path, output: Path) -> None:
    rows = _jsonl(rows_path)
    seeds = _jsonl(seeds_path)
    assistant_messages = [
        message
        for row in rows
        for message in row.get("messages") or []
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    masked = [message for message in assistant_messages if message.get("mask") is True]
    tagged = [message for message in assistant_messages if _THINK.search(str(message.get("content") or ""))]
    lines = [
        "# Sophia SFT think audit",
        "",
        f"- seeds: {len(seeds)} (seed plans intentionally have no think field)",
        f"- rows: {len(rows)}; rows with top-level think=true: "
        f"{sum(row.get('think') is True for row in rows)}",
        f"- assistant messages: {len(assistant_messages)}",
        f"- assistant messages with think tags: {len(tagged)}",
        f"- masked assistant messages without think tags: "
        f"{len(masked) - sum(_THINK.search(str(message.get('content') or '')) is not None for message in masked)}",
        "",
        "The sections below are exhaustive. A masked assistant message is prior "
        "conversation context; its hidden reasoning is intentionally removed.",
        "",
        "## 1. Seed plans",
        "",
        "Seeds contain prompts and planned turn counts only. They are not training rows.",
        "",
    ]
    for seed in seeds:
        lines.append(
            f"- `{seed.get('prompt_id')}` [{seed.get('category')}] "
            f"turns={seed.get('turns')}: {_short(seed.get('prompt'), 200)}"
        )

    lines.extend(["", "## 2. Every training row", ""])
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        lines.extend(
            [
                f"### `{sample_id}` [{row.get('category')}]",
                f"- top-level `think`: `{str(row.get('think')).lower()}`",
                f"- turn: `{row.get('turn_index')}/{int(row.get('conversation_turns') or 1) - 1}`",
            ]
        )
        messages = row.get("messages") or []
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = str(message.get("content") or "")
            match = _THINK.search(content)
            label = "masked context" if message.get("mask") is True else "supervised/tool"
            if match:
                lines.extend(
                    [
                        f"- message `{index}` ({label}) think:",
                        "",
                        "```text",
                        match.group("body").strip(),
                        "```",
                    ]
                )
            else:
                lines.append(
                    f"- message `{index}` ({label}) has **no think tag**; "
                    f"answer preview: {_short(content)}"
                )
            if message.get("tool_calls"):
                lines.append(
                    "  - tool calls: "
                    + json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True)
                )
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote exhaustive think audit for {len(rows)} rows to {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(rows_path=args.rows, seeds_path=args.seeds, output=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
