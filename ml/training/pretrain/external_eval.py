from __future__ import annotations

import math
import os
from collections.abc import Callable

import torch

from ml.core.common.io import write_json_atomic
from ml.data.token_shards.token_shards_dataset import TokenStreamDataset
from ml.training.pretrain.run_config import PretrainRunConfig
from ml.data.pretrain_filters import (
    has_repeated_sentences,
    is_poison_repetitive_text,
    normalize_text,
)
from ml.training.pretrain import manifest_policy


def _resolve_contamination_interval(*, configured: int, max_steps: int) -> int:
    if int(configured) < 0:
        raise ValueError("external_eval_contamination_interval must be >= 0")
    if int(max_steps) < 1:
        raise ValueError("max_steps must be >= 1")
    return int(configured) if int(configured) > 0 else int(max_steps)


def _require_finite_metric(value: float) -> float:
    metric = float(value)
    if not math.isfinite(metric):
        raise ValueError(f"external evaluation metric must be finite, got {metric!r}")
    return metric


def _write_report(path: str, payload: dict[str, object]) -> None:
    write_json_atomic(
        path,
        payload,
        sort_keys=True,
        ensure_ascii=False,
        make_parents=True,
    )


def _current_step_from_model(model: torch.nn.Module) -> int:
    cfg = getattr(model, "config", None)
    raw = getattr(cfg, "_sophia_global_step", 0) if cfg is not None else 0
    try:
        step = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("model config _sophia_global_step must be an integer") from exc
    if step < 0:
        raise ValueError("model config _sophia_global_step must be >= 0")
    return step


def set_current_step_on_model(*, model: torch.nn.Module, step: int) -> None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    cfg._sophia_global_step = int(step)


def _sample_split_contamination(
    *,
    split_name: str,
    split_path: str,
    tokenizer: object,
    seed: int,
    sample_count: int,
    sample_seq_len: int,
) -> dict[str, object]:
    manifest_path = manifest_policy.resolve_manifest_path(str(split_path))
    dataset = TokenStreamDataset(
        manifest_path=str(manifest_path),
        seq_len=max(int(sample_seq_len), 8),
        seed=int(seed),
        batch_size=0,
    )
    iterator = iter(dataset)
    flagged_examples: list[dict[str, object]] = []
    poison_hits = 0
    repeated_hits = 0
    rows = 0
    for index in range(max(int(sample_count), 0)):
        batch = next(iterator)
        token_ids = batch.get("input_ids")
        if not isinstance(token_ids, torch.Tensor):
            continue
        text = normalize_text(
            str(
                tokenizer.decode(
                    [int(x) for x in token_ids.detach().cpu().tolist()],
                    skip_special_tokens=True,
                )
            )
        )
        poison = bool(is_poison_repetitive_text(text))
        repeated = bool(has_repeated_sentences(text))
        poison_hits += int(poison)
        repeated_hits += int(repeated)
        rows += 1
        if poison or repeated:
            flagged_examples.append(
                {
                    "sample_index": int(index),
                    "poison_repetitive": bool(poison),
                    "repeated_sentences": bool(repeated),
                    "preview": str(text[:240]),
                }
            )
    safe_rate = 1.0
    if int(rows) > 0:
        flagged = sum(
            1
            for item in flagged_examples
            if bool(item.get("poison_repetitive"))
            or bool(item.get("repeated_sentences"))
        )
        safe_rate = 1.0 - (float(flagged) / float(rows))
    return {
        "split": str(split_name),
        "split_path": os.path.abspath(str(split_path or "").strip()),
        "manifest_path": os.path.abspath(str(manifest_path or "").strip()),
        "samples": int(rows),
        "poison_hits": int(poison_hits),
        "repeated_sentence_hits": int(repeated_hits),
        "flagged_samples": int(
            sum(
                1
                for item in flagged_examples
                if bool(item.get("poison_repetitive")) or bool(item.get("repeated_sentences"))
            )
        ),
        "safe_rate": float(safe_rate),
        "examples": flagged_examples[:8],
    }


def run_contamination_scan_eval(
    *,
    args: PretrainRunConfig,
    model: torch.nn.Module,
    tokenizer: object,
    output_dir: str,
) -> float:
    step = _current_step_from_model(model)
    resolved_output_dir = os.path.abspath(str(output_dir or "").strip())
    report_dir = os.path.join(
        resolved_output_dir,
        "external_eval",
        f"step_{int(step):08d}",
    )
    report_path = os.path.join(report_dir, "contamination_scan.json")
    splits = {
        "train": str(args.data_path),
        "val": str(args.eval_data_path),
        "test": str(args.test_data_path),
    }
    results: list[dict[str, object]] = []
    safe_rates: list[float] = []
    for idx, (name, path) in enumerate(splits.items()):
        if not str(path or "").strip():
            continue
        result = _sample_split_contamination(
            split_name=str(name),
            split_path=str(path),
            tokenizer=tokenizer,
            seed=int(args.seed) + int(idx) * 17,
            sample_count=max(int(args.external_eval_contamination_samples), 1),
            sample_seq_len=min(max(int(args.seq_len), 256), 1024),
        )
        safe_rates.append(float(result["safe_rate"]))
        results.append(result)
    if not safe_rates:
        raise ValueError("contamination scan requires at least one configured data split")
    payload = {
        "kind": "pretrain_contamination_scan_report",
        "output_dir": resolved_output_dir,
        "safe_rate_mean": float(sum(safe_rates) / float(len(safe_rates))),
        "splits": results,
    }
    _write_report(report_path, payload)
    print(f"[INFO] wrote pretrain contamination scan report: {report_path}", flush=True)
    return _require_finite_metric(float(payload["safe_rate_mean"]))


def build_pretrain_external_evals(
    *,
    args: PretrainRunConfig,
    model: torch.nn.Module,
    tokenizer: object,
    output_dir: str,
    device: torch.device,
    max_steps: int,
) -> list[tuple[str, int, Callable[[], float]]]:
    del device
    contamination_interval = _resolve_contamination_interval(
        configured=int(args.external_eval_contamination_interval),
        max_steps=int(max_steps),
    )

    def _contamination_fn() -> float:
        return run_contamination_scan_eval(
            args=args,
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
        )

    return [
        ("pretrain_contamination_safe_rate", int(contamination_interval), _contamination_fn),
    ]


__all__ = [
    "build_pretrain_external_evals",
    "run_contamination_scan_eval",
    "set_current_step_on_model",
]
