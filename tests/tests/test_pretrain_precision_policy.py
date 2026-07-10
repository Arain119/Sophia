from __future__ import annotations

import torch

from ml.training.pretrain import release_config as mod


def test_release_pretrain_precision_stack_names_pinned_bf16_path() -> None:
    metadata = mod.release_pretrain_runtime_metadata()
    status = mod.release_precision_status_payload()

    assert metadata["precision_stack"] == "bf16_sdpa_flash_liger"
    assert metadata["release_pretrain_precision"] == "bf16"
    assert (
        metadata["runtime_validation_status"]
        == "bf16_canary_pending_full_run"
    )
    assert (
        status["selected_runtime_validation_status"]
        == "bf16_canary_pending_full_run"
    )
    assert status["release_precision_safe"] is False
    assert status["project_precision_stack_validated"] is True
    assert status["selected_runtime_validation_requirements"] == [
        "runtime_context",
        "release_warmup",
    ]
    assert mod.resolve_pinned_precision_dtype() is torch.bfloat16
