from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ml.core.common.io import write_json_atomic
from ml.core.spec import ModelSpec
from ml.modeling import create_model_from_spec


REPORT_SCHEMA = "sophia_hybrid_architecture_sweep_v1"


def _positive_int_list(value: str, *, label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{label} must be a comma-separated list of integers"
        ) from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{label} values must all be positive")
    return parsed


def count_instantiated_parameters(spec: ModelSpec) -> int:
    """Count unique parameters on the real runtime module without allocating storage."""
    with torch.device("meta"):
        model = create_model_from_spec(spec)
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _candidate_row(*, spec: ModelSpec, target_parameters: int) -> dict[str, Any]:
    parameter_count = count_instantiated_parameters(spec)
    delta = int(parameter_count - target_parameters)
    return {
        "model": asdict(spec),
        "parameter_count": parameter_count,
        "parameter_count_billions": round(parameter_count / 1_000_000_000.0, 9),
        "target_delta": delta,
        "target_relative_error": abs(delta) / float(target_parameters),
        "parameter_storage_bytes": {
            "bf16": parameter_count * 2,
            "fp32": parameter_count * 4,
        },
        "count_method": "meta_device_runtime_instantiation_unique_parameters",
    }


def enumerate_architectures(
    *,
    base_spec: ModelSpec,
    vocab_sizes: Iterable[int],
    layer_counts: Iterable[int],
    ffn_hidden_sizes: Iterable[int],
    target_parameters: int,
    minimum_parameters: int,
    maximum_parameters: int,
) -> dict[str, Any]:
    target = int(target_parameters)
    minimum = int(minimum_parameters)
    maximum = int(maximum_parameters)
    if target <= 0:
        raise ValueError("target_parameters must be positive")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("invalid parameter range")

    evaluated: list[dict[str, Any]] = []
    for vocab_size in sorted({int(value) for value in vocab_sizes}):
        for n_layers in sorted({int(value) for value in layer_counts}):
            for ffn_hidden in sorted({int(value) for value in ffn_hidden_sizes}):
                spec = base_spec.with_overrides(
                    vocab_size=vocab_size,
                    n_layers=n_layers,
                    ffn_hidden=ffn_hidden,
                )
                evaluated.append(_candidate_row(spec=spec, target_parameters=target))

    candidates = [
        row
        for row in evaluated
        if minimum <= int(row["parameter_count"]) <= maximum
    ]
    candidates.sort(
        key=lambda row: (
            abs(int(row["target_delta"])),
            int(row["model"]["vocab_size"]),
            int(row["model"]["n_layers"]),
            int(row["model"]["ffn_hidden"]),
        )
    )
    recommended_by_vocab: dict[str, dict[str, Any]] = {}
    for row in candidates:
        vocab_key = str(int(row["model"]["vocab_size"]))
        recommended_by_vocab.setdefault(vocab_key, row)

    return {
        "schema": REPORT_SCHEMA,
        "status": "ok" if candidates else "no_candidates_in_range",
        "target": {
            "parameter_count": target,
            "minimum_parameter_count": minimum,
            "maximum_parameter_count": maximum,
        },
        "fixed_model_fields": {
            key: value
            for key, value in asdict(base_spec).items()
            if key not in {"vocab_size", "n_layers", "ffn_hidden"}
        },
        "search_space": {
            "vocab_sizes": sorted({int(value) for value in vocab_sizes}),
            "layer_counts": sorted({int(value) for value in layer_counts}),
            "ffn_hidden_sizes": sorted({int(value) for value in ffn_hidden_sizes}),
        },
        "evaluated_count": len(evaluated),
        "candidate_count": len(candidates),
        "recommended": candidates[0] if candidates else None,
        "recommended_by_vocab": recommended_by_vocab,
        "candidates": candidates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate Sophia architecture candidates using real meta-device model "
            "instantiation and unique parameter counting."
        )
    )
    parser.add_argument("--vocab-sizes", default="65536")
    parser.add_argument("--layer-counts", default="27,28,29")
    parser.add_argument("--ffn-hidden-sizes", default="3584,3840,4096")
    parser.add_argument("--target-parameters", type=int, default=1_000_000_000)
    parser.add_argument("--minimum-parameters", type=int, default=950_000_000)
    parser.add_argument("--maximum-parameters", type=int, default=1_050_000_000)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Recorded model capacity; it does not affect parameter count.",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    vocab_sizes = _positive_int_list(args.vocab_sizes, label="vocab-sizes")
    layer_counts = _positive_int_list(args.layer_counts, label="layer-counts")
    ffn_hidden_sizes = _positive_int_list(
        args.ffn_hidden_sizes,
        label="ffn-hidden-sizes",
    )
    if int(args.max_seq_len) <= 0:
        parser.error("--max-seq-len must be positive")

    report = enumerate_architectures(
        base_spec=ModelSpec.default().with_overrides(
            max_seq_len=int(args.max_seq_len),
        ),
        vocab_sizes=vocab_sizes,
        layer_counts=layer_counts,
        ffn_hidden_sizes=ffn_hidden_sizes,
        target_parameters=int(args.target_parameters),
        minimum_parameters=int(args.minimum_parameters),
        maximum_parameters=int(args.maximum_parameters),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, report)
    print(
        f"wrote {report['candidate_count']} candidates from "
        f"{report['evaluated_count']} instantiated models to {output}"
    )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
