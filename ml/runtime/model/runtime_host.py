from __future__ import annotations

from threading import RLock

import torch

from ml.runtime.model.runtime_control import (
    RuntimeControlModel,
    enable_runtime_cache,
    ensure_batch_capacity,
    ensure_sequence_capacity,
    rebuild_runtime_buffers,
    refresh_state_buffers,
    reset_runtime_cache,
)
from ml.runtime.model.state import (
    LayerCacheSnapshot,
    RuntimeCacheSnapshot,
    TransformerRuntimeState,
)


class TransformerRuntime:
    def __init__(
        self,
        model: RuntimeControlModel,
        *,
        state: TransformerRuntimeState | None = None,
    ) -> None:
        self.model = model
        self.state = state or TransformerRuntimeState()

    @property
    def lock(self) -> RLock:
        return self.state.lock

    @lock.setter
    def lock(self, value: RLock) -> None:
        self.state.lock = value

    def supports_loss_chunk_size(self) -> bool:
        return True

    def supports_checkpoint_excludes(self) -> bool:
        return True

    def runtime_recipe_knobs(self) -> tuple[int, int, int]:
        return (
            int(self.state.loss_chunk_size),
            int(self.model.gradient_checkpointing_exclude_first),
            int(self.model.gradient_checkpointing_exclude_last),
        )

    def apply_runtime_recipe_knobs(
        self,
        *,
        loss_chunk_size: int,
        gradient_checkpointing_exclude_first: int,
        gradient_checkpointing_exclude_last: int,
    ) -> tuple[int, int, int]:
        chunk_size = int(loss_chunk_size)
        exclude_first = int(gradient_checkpointing_exclude_first)
        exclude_last = int(gradient_checkpointing_exclude_last)
        if chunk_size < 0:
            raise ValueError(f"loss_chunk_size must be >= 0, got {chunk_size}")
        if exclude_first < 0:
            raise ValueError(
                "gradient_checkpointing_exclude_first must be >= 0, "
                f"got {exclude_first}"
            )
        if exclude_last < 0:
            raise ValueError(
                "gradient_checkpointing_exclude_last must be >= 0, "
                f"got {exclude_last}"
            )
        layer_count = len(self.model.layers)
        if exclude_first + exclude_last > layer_count:
            raise ValueError(
                "gradient checkpoint exclusions exceed model layer count: "
                f"first={exclude_first} last={exclude_last} layers={layer_count}"
            )
        self.state.loss_chunk_size = chunk_size
        self.model.gradient_checkpointing_exclude_first = exclude_first
        self.model.gradient_checkpointing_exclude_last = exclude_last
        return self.runtime_recipe_knobs()

    def ensure_batch_capacity(self, batch_size: int) -> None:
        ensure_batch_capacity(self.model, batch_size)

    def ensure_sequence_capacity(self, max_seq_len: int) -> None:
        ensure_sequence_capacity(self.model, max_seq_len)

    def ensure_runtime_max_seq_len(self, max_seq_len: int) -> None:
        self.ensure_sequence_capacity(max_seq_len)

    def runtime_max_seq_len(self) -> int:
        return int(self.model.runtime_max_seq_len())

    def runtime_batch_capacity(self) -> int:
        return int(self.model.runtime_batch_capacity())

    def rebuild_runtime_buffers(self) -> None:
        rebuild_runtime_buffers(self.model)

    def sync_runtime_batch_capacity(self, max_batch_size: int) -> int:
        self.ensure_batch_capacity(max_batch_size)
        return self.runtime_batch_capacity()

    def reset_runtime_cache(self) -> None:
        reset_runtime_cache(self.model)

    def refresh_state_buffers(self) -> None:
        refresh_state_buffers(self.model)

    def _forward_with_last_hidden(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int,
        return_all_logits: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden, _ = self.model._forward_hidden(input_ids, start_pos=start_pos)
        last_hidden = hidden[:, -1, :]
        logits_hidden = hidden if return_all_logits else hidden[:, -1:, :]
        logits = self.model.head_mixer(
            logits_hidden, norm=self.model.norm, output=self.model.output
        )
        if not return_all_logits:
            logits = logits[:, 0, :]
        return logits, last_hidden

    def forward_with_last_hidden(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int = 0,
        return_all_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        enable_runtime_cache(self.model)
        return self._forward_with_last_hidden(
            input_ids,
            start_pos=int(start_pos),
            return_all_logits=return_all_logits,
        )

    def replay_with_cache(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int = 0,
        return_all_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        enable_runtime_cache(self.model)
        if int(input_ids.size(1)) <= 1 or int(start_pos) == 0:
            return self._forward_with_last_hidden(
                input_ids,
                start_pos=int(start_pos),
                return_all_logits=return_all_logits,
            )
        outputs: list[torch.Tensor] = []
        last_hidden = None
        for offset in range(int(input_ids.size(1))):
            logits, last_hidden = self._forward_with_last_hidden(
                input_ids[:, offset : offset + 1],
                start_pos=int(start_pos) + offset,
                return_all_logits=True,
            )
            outputs.append(logits)
        combined = torch.cat(outputs, dim=1)
        return (combined if return_all_logits else combined[:, -1, :]), last_hidden

    @staticmethod
    def cache_batch_size(batch_size: int | None, max_batch_size: int) -> int:
        active = int(max_batch_size if batch_size is None else batch_size)
        if active <= 0:
            raise ValueError(f"batch_size must be >= 1, got {active}")
        return active

    def cache_dump(
        self,
        *,
        device: str = "cpu",
        cache_pos: int | None = None,
        batch_size: int | None = None,
    ) -> RuntimeCacheSnapshot:
        active = self.cache_batch_size(batch_size, self.runtime_batch_capacity())
        self.ensure_batch_capacity(active)
        return RuntimeCacheSnapshot(
            layers=tuple(
                layer.attn.cache_snapshot(
                    device=device, batch_size=active, cache_pos=cache_pos
                )
                for layer in self.model.layers
            )
        )

    def cache_load(self, cache_snapshot: RuntimeCacheSnapshot) -> None:
        enable_runtime_cache(self.model)
        if not isinstance(cache_snapshot, RuntimeCacheSnapshot):
            raise TypeError("cache snapshot must be a RuntimeCacheSnapshot")
        if len(cache_snapshot.layers) > len(self.model.layers):
            raise ValueError("cache snapshot has more layers than the model")
        required = cache_snapshot.batch_size()
        if required:
            self.ensure_batch_capacity(required)
        for index, snapshot in enumerate(cache_snapshot.layers):
            if snapshot is not None:
                self.model.layers[index].attn.validate_cache_snapshot(snapshot)
        self.reset_runtime_cache()
        for index, snapshot in enumerate(cache_snapshot.layers):
            if snapshot is not None:
                self.model.layers[index].attn.load_cache_snapshot(snapshot)


__all__ = ["TransformerRuntime", "LayerCacheSnapshot"]
