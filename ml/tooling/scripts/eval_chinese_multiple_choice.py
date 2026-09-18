from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ml.core.common.io import write_json_atomic


CHOICES = ("A", "B", "C", "D")


def load_cases(path: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"benchmark row must be an object: {path}:{line_number}")
            case_id = str(payload.get("id") or "").strip()
            choices = payload.get("choices")
            answer = str(payload.get("answer") or "").strip().upper()
            if not case_id or case_id in seen:
                raise ValueError(f"benchmark id must be non-empty and unique: {case_id!r}")
            if not isinstance(choices, dict) or any(
                not str(choices.get(key) or "").strip() for key in CHOICES
            ):
                raise ValueError(f"benchmark choices must contain A-D: {case_id}")
            if answer not in CHOICES:
                continue
            seen.add(case_id)
            cases.append(payload)
    if not cases:
        raise ValueError(f"benchmark has no scored cases: {path}")
    return cases


def select_cases(
    cases: Iterable[dict[str, Any]],
    *,
    splits: set[str],
    max_cases_per_benchmark: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        if str(case.get("split") or "") not in splits:
            continue
        grouped[str(case.get("benchmark") or "unknown")].append(case)
    selected: list[dict[str, Any]] = []
    for benchmark, rows in sorted(grouped.items()):
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"{int(seed)}:{row['id']}".encode()
            ).hexdigest()
        )
        if int(max_cases_per_benchmark) > 0:
            rows = rows[: int(max_cases_per_benchmark)]
        selected.extend(rows)
        print(f"[SELECT] benchmark={benchmark} cases={len(rows):,}", flush=True)
    selected.sort(key=lambda row: str(row["id"]))
    if not selected:
        raise ValueError("no benchmark cases matched the requested splits")
    return selected


def build_prompt(case: dict[str, Any]) -> str:
    question = str(case.get("question") or "").strip()
    choices = case["choices"]
    lines = [
        "以下是单项选择题，请从 A、B、C、D 中选择正确答案。",
        "",
        question,
    ]
    lines.extend(f"{key}. {str(choices[key]).strip()}" for key in CHOICES)
    lines.append("答案：")
    return "\n".join(lines)


def aggregate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(group)
        return {
            "cases": total,
            "accuracy": (
                sum(bool(row["correct"]) for row in group) / float(max(total, 1))
            ),
            "prediction_distribution": dict(
                sorted(Counter(str(row["prediction"]) for row in group).items())
            ),
        }

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "by_benchmark": defaultdict(list),
        "by_split": defaultdict(list),
        "by_subject": defaultdict(list),
    }
    for row in rows:
        groups["by_benchmark"][str(row["benchmark"])].append(row)
        groups["by_split"][f"{row['benchmark']}:{row['split']}"] .append(row)
        groups["by_subject"][f"{row['benchmark']}:{row['subject']}"] .append(row)
    return {
        "overall": summarize(rows),
        **{
            group_name: {
                key: summarize(value) for key, value in sorted(group.items())
            }
            for group_name, group in groups.items()
        },
    }


def choice_token_ids(tokenizer: object) -> dict[str, int]:
    token_ids: dict[str, int] = {}
    for choice in CHOICES:
        encoded = tokenizer(choice, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise ValueError(
                f"choice label must map to one token for next-label scoring: {choice}={encoded}"
            )
        token_ids[choice] = int(encoded[0])
    if len(set(token_ids.values())) != len(CHOICES):
        raise ValueError(f"choice labels must map to distinct tokens: {token_ids}")
    return token_ids


def _sha256(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing evaluation binding file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_binding(
    *,
    protocol_json: str | Path | None,
    checkpoint: str | Path | None,
    checkpoint_sha256: str | None,
    eval_export_model: str | Path | None,
    evaluation_asset_sha256: str,
) -> dict[str, str | None] | None:
    values = (protocol_json, checkpoint, checkpoint_sha256, eval_export_model)
    if not any(value is not None and str(value).strip() for value in values):
        return None
    if not all(value is not None and str(value).strip() for value in (protocol_json, checkpoint, eval_export_model)):
        raise ValueError(
            "release-bound evaluation requires --protocol_json, --checkpoint, and --eval_export_model"
        )
    protocol_path = Path(str(protocol_json)).expanduser().resolve()
    checkpoint_path = Path(str(checkpoint)).expanduser().resolve()
    model_path = Path(str(eval_export_model)).expanduser().resolve()
    protocol_digest = _sha256(protocol_path)
    checkpoint_digest = _sha256(checkpoint_path)
    model_digest = _sha256(model_path)
    if checkpoint_sha256 and str(checkpoint_sha256).lower() != checkpoint_digest:
        raise ValueError("checkpoint_sha256 does not match checkpoint")
    return {
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "eval_export_model_path": str(model_path),
        "eval_export_model_sha256": model_digest,
        "evaluation_asset_sha256": str(evaluation_asset_sha256),
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    from ml.modeling.decoder_output import require_output_logits
    from ml.runtime.inference.eval.common import torch
    from ml.runtime.inference.eval.model_io import init_model
    from ml.runtime.inference.eval.runtime_config import EvalRuntimeConfig

    cases = select_cases(
        load_cases(str(args.benchmark_jsonl)),
        splits={value.strip() for value in str(args.splits).split(",") if value.strip()},
        max_cases_per_benchmark=int(args.max_cases_per_benchmark),
        seed=int(args.seed),
    )
    runtime = EvalRuntimeConfig(
        export_dir=str(args.export_dir),
        device=str(args.device),
        seed=int(args.seed),
        use_cache=False,
    )
    model, tokenizer = init_model(runtime)
    device = torch.device(str(args.device))
    label_ids = choice_token_ids(tokenizer)
    label_index = torch.tensor(
        [label_ids[key] for key in CHOICES],
        dtype=torch.long,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    batch_size = max(int(args.batch_size), 1)
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        prompts = [build_prompt(case) for case in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(args.max_prompt_tokens),
            add_special_tokens=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                use_cache=False,
                logits_to_keep=1,
            )
            logits = require_output_logits(
                output,
                context="Chinese multiple-choice evaluation",
            )[:, -1].float()
            choice_logits = logits.index_select(-1, label_index)
            probabilities = torch.softmax(choice_logits, dim=-1)
            predictions = torch.argmax(choice_logits, dim=-1).cpu().tolist()
        for offset, case in enumerate(batch):
            prediction = CHOICES[int(predictions[offset])]
            answer = str(case["answer"])
            rows.append(
                {
                    "id": str(case["id"]),
                    "benchmark": str(case["benchmark"]),
                    "subject": str(case["subject"]),
                    "split": str(case["split"]),
                    "answer": answer,
                    "prediction": prediction,
                    "correct": prediction == answer,
                    "choice_probabilities": {
                        key: float(probabilities[offset, index].item())
                        for index, key in enumerate(CHOICES)
                    },
                }
            )
        completed = min(start + len(batch), len(cases))
        if completed == len(cases) or completed % 100 == 0:
            correct = sum(bool(row["correct"]) for row in rows)
            print(
                f"[EVAL] cases={completed:,}/{len(cases):,} "
                f"accuracy={correct / float(max(len(rows), 1)):.4f}",
                flush=True,
            )
    report = {
        "schema": "sophia_chinese_multiple_choice_eval_v1",
        "evaluation_binding": _evaluation_binding(
            protocol_json=args.protocol_json,
            checkpoint=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            eval_export_model=args.eval_export_model,
            evaluation_asset_sha256=_sha256(Path(str(args.benchmark_jsonl))),
        ),
        "model_dir": os.path.abspath(str(args.export_dir)),
        "benchmark_jsonl": str(Path(str(args.benchmark_jsonl)).expanduser().resolve()),
        "benchmark_sha256": _sha256(Path(str(args.benchmark_jsonl))),
        "selection": {
            "splits": sorted(
                value.strip() for value in str(args.splits).split(",") if value.strip()
            ),
            "max_cases_per_benchmark": int(args.max_cases_per_benchmark),
            "seed": int(args.seed),
        },
        "scoring": "next-token likelihood over labels A/B/C/D",
        "label_token_ids": label_ids,
        "summary": aggregate_predictions(rows),
        "cases": rows,
    }
    write_json_atomic(
        args.output,
        report,
        ensure_ascii=False,
        sort_keys=True,
        make_parents=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate C-Eval and CMMLU accuracy")
    parser.add_argument("--export_dir", required=True)
    parser.add_argument("--benchmark_jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", default="dev,val")
    parser.add_argument("--max_cases_per_benchmark", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--protocol_json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint_sha256")
    parser.add_argument("--eval_export_model")
    return parser


def main() -> None:
    report = run_evaluation(build_parser().parse_args())
    print(
        f"[DONE] accuracy={report['summary']['overall']['accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
