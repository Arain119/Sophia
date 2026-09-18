"""Eager SFT step runner with the formal MLA QK-Clip protection."""

from __future__ import annotations

import torch

from ml.runtime.model.attention import SophiaMLA
from ml.training.pretrain.engine.step_runner_impl import EagerStepRunner
from ml.training.pretrain.release_config import RELEASE_QK_CLIP_THRESHOLD


class SFTEagerStepRunner(EagerStepRunner):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        base_dtype: torch.dtype,
        token_weighted: bool,
    ) -> None:
        super().__init__(
            model=model,
            base_dtype=base_dtype,
            token_weighted=token_weighted,
        )
        self._mla_modules = tuple(
            module for module in model.modules() if isinstance(module, SophiaMLA)
        )
        if not self._mla_modules:
            raise RuntimeError("SFT QK-Clip found no Sophia MLA modules")
        devices = {next(module.parameters()).device for module in self._mla_modules}
        head_counts = {int(module.num_heads) for module in self._mla_modules}
        if len(devices) != 1 or len(head_counts) != 1:
            raise RuntimeError("SFT MLA modules disagree on device or head count")
        self._attention_logit_max = torch.full(
            (len(self._mla_modules), head_counts.pop()),
            float("-inf"),
            device=devices.pop(),
            dtype=torch.float32,
        )
        for module, layer_buffer in zip(
            self._mla_modules,
            self._attention_logit_max,
            strict=True,
        ):
            module.enable_attention_logit_telemetry(layer_buffer)

    def begin_update(self, *, accum_steps: int) -> None:
        del accum_steps
        self._attention_logit_max.fill_(float("-inf"))

    def post_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        apply_clip = getattr(optimizer, "apply_mla_qk_clip", None)
        if not callable(apply_clip):
            raise RuntimeError("SFT QK-Clip requires the Muon hybrid optimizer")
        apply_clip(
            self._mla_modules,
            self._attention_logit_max,
            threshold=RELEASE_QK_CLIP_THRESHOLD,
        )


__all__ = ["SFTEagerStepRunner"]
