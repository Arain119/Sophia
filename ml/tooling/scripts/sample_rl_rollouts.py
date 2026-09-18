"""Sample fixed-size rollout groups from an exported Sophia policy."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ml.runtime.inference.eval.model_io import init_model
from ml.runtime.inference.eval.runtime_config import EvalRuntimeConfig
from ml.training.rl.prompt_pool import IN_TEMPLATE, OUTSIDE_TEMPLATE
from ml.training.rl.rollout import (
    RolloutPrompt,
    build_model_generator,
    load_prompts,
    rollout_group,
    write_groups,
)


_CATEGORY_ORDER = (
    "identity",
    "safety",
    "companion_chat",
    "chinese_qa",
    "chinese_writing",
    "english_general",
    "structured_output",
    "code",
    "math",
    "unknown",
)


def _user_turns(prompt: RolloutPrompt) -> int:
    return sum(message.get("role") == "user" for message in prompt.messages)


def select_prompts(
    prompts: list[RolloutPrompt],
    *,
    max_prompts: int,
    include_templates: bool = False,
    seed: int = 42,
) -> list[RolloutPrompt]:
    """Select a deterministic, category-round-robin RL evaluation slice.

    The policy is most useful where the SFT corpus does not already contain a
    high-frequency answer skeleton. Round-robin prevents the large QA bucket
    from consuming a pilot before identity, safety, and dialogue are observed;
    sorting within a bucket gives multi-turn prompts first.
    """
    candidates = [
        prompt
        for prompt in prompts
        if include_templates or OUTSIDE_TEMPLATE in prompt.tags
    ]
    if max_prompts <= 0 or max_prompts >= len(candidates):
        limit = len(candidates)
    else:
        limit = int(max_prompts)
    buckets: dict[str, list[RolloutPrompt]] = defaultdict(list)
    for prompt in candidates:
        category = next(
            (tag for tag in prompt.tags if tag not in {IN_TEMPLATE, OUTSIDE_TEMPLATE}),
            "unknown",
        )
        buckets[category].append(prompt)
    rng = random.Random(int(seed))
    for bucket in buckets.values():
        rng.shuffle(bucket)
        bucket.sort(key=_user_turns)
    ordered_categories = [
        category for category in _CATEGORY_ORDER if category in buckets
    ] + sorted(set(buckets) - set(_CATEGORY_ORDER))
    selected: list[RolloutPrompt] = []
    while len(selected) < limit:
        progressed = False
        for category in ordered_categories:
            bucket = buckets[category]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
                progressed = True
        if not progressed:
            break
    return selected


def sample_rollouts(
    *,
    model_dir: Path,
    prompts_path: Path,
    output_path: Path,
    max_prompts: int = 256,
    group_size: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    include_templates: bool = False,
    seed: int = 42,
    device: str = "cuda:0",
) -> dict[str, Any]:
    if int(group_size) < 2:
        raise ValueError("GRPO rollout groups need at least two completions")
    if output_path.exists():
        raise ValueError(f"rollout output already exists: {output_path}")
    prompts = load_prompts(prompts_path)
    selected = select_prompts(
        prompts,
        max_prompts=int(max_prompts),
        include_templates=bool(include_templates),
        seed=int(seed),
    )
    if not selected:
        raise ValueError("no prompts remain after RL selection")
    runtime = EvalRuntimeConfig(
        export_dir=str(model_dir.resolve()),
        seed=42,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=0,
        do_sample=True,
        use_cache=True,
        device=str(device),
        mode="auto",
    )
    model, tokenizer = init_model(runtime)
    generate_fn = build_model_generator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=float(temperature),
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(selected, start=1):
        group = rollout_group(
            prompt,
            generate_fn=generate_fn,
            group_size=int(group_size),
        )
        write_groups(output_path, [(prompt, group)])
        if index % 16 == 0 or index == len(selected):
            print(f"[RL-ROLLOUT] {index}/{len(selected)}", flush=True)
    report = {
        "schema": "sophia_rl_rollout_report",
        "status": "complete",
        "model_dir": str(model_dir.resolve()),
        "prompts": len(selected),
        "candidate_prompts": len(prompts),
        "include_templates": bool(include_templates),
        "seed": int(seed),
        "categories": {
            category: sum(
                category in prompt.tags for prompt in selected
            )
            for category in sorted(
                {
                    tag
                    for prompt in selected
                    for tag in prompt.tags
                    if tag not in {IN_TEMPLATE, OUTSIDE_TEMPLATE}
                }
            )
        },
        "outside_template_prompts": sum(
            OUTSIDE_TEMPLATE in prompt.tags for prompt in selected
        ),
        "group_size": int(group_size),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "decode": {
            "skip_special_tokens": False,
            "preserve_think_tags": True,
            "remove_terminal_eos": True,
        },
        "output": str(output_path.resolve()),
    }
    report_path = output_path.with_name(output_path.name + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--include-templates",
        action="store_true",
        help="also sample high-frequency training skeletons (diagnostic only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = sample_rollouts(
        model_dir=args.model_dir,
        prompts_path=args.prompts,
        output_path=args.output,
        max_prompts=args.max_prompts,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        include_templates=args.include_templates,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
