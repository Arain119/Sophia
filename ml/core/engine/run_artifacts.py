from __future__ import annotations

import os
from dataclasses import dataclass

from ml.core.engine.metrics import append_metrics_row
from ml.core.engine.types import MetricsRow, RuntimeMetadata, StatePayload
from ml.core.common.io import write_json_atomic


@dataclass(frozen=True)
class RunArtifacts:
    output_dir: str
    args_payload: StatePayload
    meta: RuntimeMetadata
    start_row: MetricsRow


def build_run_artifacts(
    *,
    output_dir: str,
    args_payload: StatePayload,
    meta: RuntimeMetadata,
) -> RunArtifacts:
    resolved_output_dir = os.path.abspath(str(output_dir))
    return RunArtifacts(
        output_dir=resolved_output_dir,
        args_payload=dict(args_payload),
        meta=dict(meta),
        start_row={"type": "start", **dict(meta)},
    )


def write_run_artifacts(artifacts: RunArtifacts) -> None:
    run_args_path = os.path.join(str(artifacts.output_dir), "run_args.json")
    if not os.path.exists(run_args_path):
        write_json_atomic(run_args_path, dict(artifacts.args_payload), sort_keys=True)

    run_meta_path = os.path.join(str(artifacts.output_dir), "run_meta.json")
    if not os.path.exists(run_meta_path):
        write_json_atomic(run_meta_path, dict(artifacts.meta), sort_keys=True)
    append_metrics_row(str(artifacts.output_dir), dict(artifacts.start_row))


__all__ = [
    "RunArtifacts",
    "build_run_artifacts",
    "write_run_artifacts",
]
