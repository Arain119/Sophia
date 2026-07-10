from __future__ import annotations

import os
from collections import defaultdict

import torch

from ml.core.engine.session_runtime import reset_runtime_state
from ml.training.posttrain.contracts import SupportsPosttrainTokenizer
from ml.tasks.capability_report_support import (
    SupervisedStats,
    limit_examples,
    maybe_log_progress,
    metadata_bucket,
    metadata_family,
    write_json,
)
from ml.training.model_contracts import require_compute_loss
from ml.training.posttrain.data import (
    ChatExample,
    collate_supervised_examples,
    encode_supervised_example,
    load_supervised_examples,
)
from ml.training.posttrain.tokenizer_runtime import (
    resolve_tokenizer_pad_token_id,
)


def build_single_example_supervised_batch(
    *,
    tokenizer: SupportsPosttrainTokenizer,
    example: ChatExample,
    max_seq_len: int,
    pad_to_multiple_of: int,
) -> dict[str, torch.Tensor]:
    encoded = encode_supervised_example(
        tokenizer=tokenizer,
        example=example,
        max_seq_len=int(max_seq_len),
    )
    return collate_supervised_examples(
        [encoded],
        pad_token_id=resolve_tokenizer_pad_token_id(tokenizer),
        pad_to_multiple_of=int(pad_to_multiple_of),
    )


def write_supervised_capability_report(
    *,
    output_dir: str,
    report_name: str,
    model: torch.nn.Module,
    tokenizer: SupportsPosttrainTokenizer,
    split_paths: dict[str, str],
    max_seq_len: int,
    pad_to_multiple_of: int,
    max_examples_per_split: int = 0,
    progress_every: int = 0,
) -> dict[str, object]:
    resolved_splits = {
        str(split): os.path.abspath(str(path or "").strip())
        for split, path in split_paths.items()
        if str(path or "").strip()
    }
    if not resolved_splits:
        return {}
    device = next(model.parameters()).device
    require_compute_loss(model, context="SFT capability report")
    was_training = bool(model.training)
    model.eval()
    try:
        payload: dict[str, object] = {
            "kind": "sft_capability_report",
            "output_dir": os.path.abspath(str(output_dir or "").strip()),
            "splits": {},
        }
        with torch.inference_mode():
            for split, path in resolved_splits.items():
                examples = limit_examples(
                    split=str(split),
                    examples=load_supervised_examples(path),
                    max_examples_per_split=int(max_examples_per_split),
                )
                overall = SupervisedStats()
                by_family: defaultdict[str, SupervisedStats] = defaultdict(SupervisedStats)
                by_bucket: defaultdict[str, SupervisedStats] = defaultdict(SupervisedStats)
                total_examples = len(examples)
                for index, example in enumerate(examples, start=1):
                    batch = build_single_example_supervised_batch(
                        tokenizer=tokenizer,
                        example=example,
                        max_seq_len=int(max_seq_len),
                        pad_to_multiple_of=int(pad_to_multiple_of),
                    )
                    batch = {
                        key: value.to(device=device)
                        for key, value in batch.items()
                    }
                    supervised_tokens = int(
                        (batch["labels"][:, 1:] != -100).sum(dtype=torch.int64).item()
                    )
                    if supervised_tokens <= 0:
                        raise RuntimeError(
                            f"SFT capability report saw zero supervised tokens in split={split!r}"
                        )
                    model_inputs: dict[str, object] = dict(batch)
                    model_inputs["use_cache"] = False
                    model_inputs["compute_loss"] = True
                    outputs = model(**model_inputs)
                    if outputs.loss is None:
                        raise RuntimeError("SFT capability report forward returned loss=None")
                    loss_sum = float(outputs.loss.detach().float().item()) * float(
                        supervised_tokens
                    )
                    family = metadata_family(dict(example.metadata))
                    bucket = metadata_bucket(dict(example.metadata))
                    overall.update(
                        loss_sum=float(loss_sum),
                        supervised_tokens=int(supervised_tokens),
                    )
                    by_family[family].update(
                        loss_sum=float(loss_sum),
                        supervised_tokens=int(supervised_tokens),
                    )
                    by_bucket[bucket].update(
                        loss_sum=float(loss_sum),
                        supervised_tokens=int(supervised_tokens),
                    )
                    maybe_log_progress(
                        kind="SFT",
                        split=str(split),
                        index=int(index),
                        total=int(total_examples),
                        progress_every=int(progress_every),
                    )
                payload["splits"][split] = {
                    "dataset_path": path,
                    "overall": overall.payload(),
                    "by_family": {
                        key: value.payload()
                        for key, value in sorted(by_family.items())
                    },
                    "by_bucket": {
                        key: value.payload()
                        for key, value in sorted(by_bucket.items())
                    },
                }
    finally:
        reset_runtime_state(model)
        model.train(was_training)
    out_path = os.path.join(str(output_dir), str(report_name))
    write_json(out_path, payload)
    print(f"[INFO] wrote SFT capability report: {out_path}", flush=True)
    return payload


__all__ = [
    "build_single_example_supervised_batch",
    "write_supervised_capability_report",
]
