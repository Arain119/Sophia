from __future__ import annotations

from collections.abc import Callable
import os

import torch

from ml.tasks.sft.reports import write_supervised_capability_report
from ml.tasks.sft.spec import SftStageConfig
from ml.training.posttrain.contracts import SupportsPosttrainTokenizer
from ml.training.posttrain.runtime import finalize_posttrain_stage_outputs
from ml.training.posttrain.types import (
    PosttrainReportSpec,
    PosttrainRunContext,
)


def write_sft_report(
    *,
    output_dir: str,
    model: torch.nn.Module,
    tokenizer: SupportsPosttrainTokenizer,
    split_paths: dict[str, str],
    config: SftStageConfig,
    report_name: str = "sft_capability_report.json",
) -> None:
    write_supervised_capability_report(
        output_dir=str(output_dir),
        report_name=str(report_name),
        model=model,
        tokenizer=tokenizer,
        split_paths=split_paths,
        max_seq_len=int(config.max_seq_len),
        pad_to_multiple_of=int(config.pad_to_multiple_of),
        max_examples_per_split=int(config.report_max_examples_per_split),
        progress_every=int(config.report_progress_every),
    )


def build_sft_checkpoint_report_callback(
    *,
    runtime: PosttrainRunContext,
    model: torch.nn.Module,
    tokenizer: SupportsPosttrainTokenizer,
    config: SftStageConfig,
    global_step: int,
) -> Callable[[], None]:
    report_dir = os.path.join(
        str(runtime.output_dir),
        "checkpoint_capability",
        f"step_{int(global_step):08d}",
    )

    def _write_checkpoint_report() -> None:
        if not any(
            str(path or "").strip()
            for path in (str(config.eval_data_path or ""), str(config.test_data_path or ""))
        ):
            return
        write_sft_report(
            output_dir=report_dir,
            model=model,
            tokenizer=tokenizer,
            split_paths={
                "val": str(config.eval_data_path or ""),
                "test": str(config.test_data_path or ""),
            },
            config=config,
            report_name="sft_capability_report.json",
        )

    return _write_checkpoint_report


def build_sft_finalize_fn(
    *,
    runtime: PosttrainRunContext,
    model: torch.nn.Module,
    tokenizer: SupportsPosttrainTokenizer,
    config: SftStageConfig,
) -> Callable[[], None]:
    def _finalize_outputs() -> None:
        finalize_posttrain_stage_outputs(
            runtime=runtime,
            report=PosttrainReportSpec(
                split_paths={
                    "val": str(config.eval_data_path or ""),
                    "test": str(config.test_data_path or ""),
                },
                report_name="sft_capability_report.json",
            ),
            report_fn=lambda split_paths: write_sft_report(
                output_dir=str(runtime.output_dir),
                model=model,
                tokenizer=tokenizer,
                split_paths=split_paths,
                config=config,
            ),
        )

    return _finalize_outputs


__all__ = [
    "build_sft_checkpoint_report_callback",
    "build_sft_finalize_fn",
    "write_sft_report",
]
