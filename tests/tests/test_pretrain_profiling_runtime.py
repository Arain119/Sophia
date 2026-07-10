from __future__ import annotations

import torch

from ml.tasks.pretrain.observability.profiling import (
    resolve_update_profile_context,
)
from ml.training.pretrain.run_config import PretrainRunConfig


def test_resolve_update_profile_context_returns_context_for_cuda_run(tmp_path) -> None:
    args = PretrainRunConfig(
        data_path="dataset/pretrain_tokens",
        output_dir=str(tmp_path),
        batch_size=2,
        accumulation_steps=3,
        weight_decay=0.1,
    )

    ctx = resolve_update_profile_context(
        args=args,
        output_dir=str(tmp_path),
        device=torch.device("cuda:0"),
    )

    assert ctx is not None
    assert ctx.batch_size == 2
    assert ctx.accumulation_steps == 3
    assert ctx.output_path.endswith("update_profile.json")
