from __future__ import annotations

from ml.training.pretrain import release_config as mod


def test_release_pretrain_precision_stack_names_pinned_bf16_path() -> None:
    metadata = mod.release_pretrain_runtime_metadata()

    assert metadata["precision_stack"] == "bf16_fla_kda_sdpa_mla_liger"
    assert metadata["precision"] == "bf16"
    assert metadata["mla_qk_clip_enabled"] == "true"
    assert metadata["mla_qk_clip_threshold"] == "100.0"
    assert mod.RELEASE_PRETRAIN_DTYPE is mod.torch.bfloat16
