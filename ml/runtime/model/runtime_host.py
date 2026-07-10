from __future__ import annotations

from threading import RLock

import torch

from ml.runtime.model.runtime_control import (
    RuntimeControlModel,
    enable_runtime_cache as _enable_runtime_cache,
    ensure_batch_capacity as _ensure_batch_capacity,
    ensure_sequence_capacity as _ensure_sequence_capacity,
    rebuild_runtime_buffers as _rebuild_runtime_buffers,
    refresh_state_buffers as _refresh_state_buffers,
    reset_runtime_cache as _reset_runtime_cache,
)
from ml.runtime.model.state import (
    AttentionCacheSnapshot,
    RuntimeCacheSnapshot,
    TransformerRuntimeState,
)


def _dump_full_attn_snapshot(
    attn: object,
    *,
    device: str,
    batch_size: int,
) -> AttentionCacheSnapshot:
    return AttentionCacheSnapshot(kv=attn.kv_cache[:batch_size].to(device).clone())


def _validate_tensor_capacity(name: str, src: torch.Tensor, dst: torch.Tensor) -> None:
    if src.dim() != dst.dim():
        raise ValueError(f"{name} rank mismatch: {tuple(src.shape)} vs {tuple(dst.shape)}")
    for axis, (src_dim, dst_dim) in enumerate(zip(src.shape, dst.shape, strict=True)):
        if int(src_dim) > int(dst_dim):
            raise ValueError(
                f"{name} shape exceeds cache capacity on dim {axis}: "
                f"{tuple(src.shape)} > {tuple(dst.shape)}"
            )


def _validate_attn_cache_snapshot(
    prefix: str,
    attn: object,
    snapshot: AttentionCacheSnapshot | None,
) -> None:
    if snapshot is None or snapshot.kv is None:
        return
    _validate_tensor_capacity(f"{prefix}_kv", snapshot.kv, attn.kv_cache)


def _load_attn_cache_snapshot(
    attn: object,
    snapshot: AttentionCacheSnapshot | None,
) -> None:
    if snapshot is None or snapshot.kv is None:
        return
    kv_cache = attn.kv_cache
    kv = snapshot.kv.to(kv_cache.device)
    kv_cache[: kv.size(0), : kv.size(1)].copy_(kv)


class TransformerRuntime:
    """Mutable runtime host for a Sophia Transformer."""

    def __init__(
        self,
        model: RuntimeControlModel,
        *,
        precompute_freqs_cis,
        state: TransformerRuntimeState | None = None,
    ) -> None:
        self.model = model
        self.state = state or TransformerRuntimeState()
        self._precompute_freqs_cis = precompute_freqs_cis

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
        chunk_size = max(int(loss_chunk_size), 0)
        exclude_first = int(gradient_checkpointing_exclude_first)
        exclude_last = int(gradient_checkpointing_exclude_last)
        self.state.loss_chunk_size = int(chunk_size)
        self.model.gradient_checkpointing_exclude_first = int(exclude_first)
        self.model.gradient_checkpointing_exclude_last = int(exclude_last)
        return (
            int(chunk_size),
            int(exclude_first),
            int(exclude_last),
        )

    def ensure_batch_capacity(self, batch_size: int) -> None:
        _ensure_batch_capacity(self.model, batch_size)

    def ensure_sequence_capacity(self, max_seq_len: int) -> None:
        _ensure_sequence_capacity(
            self.model,
            max_seq_len,
            precompute_freqs_cis=self._precompute_freqs_cis,
        )

    def ensure_runtime_max_seq_len(self, max_seq_len: int) -> None:
        self.ensure_sequence_capacity(int(max_seq_len))

    def runtime_max_seq_len(self) -> int:
        return int(self.model.runtime_max_seq_len())

    def runtime_prefix_replay_len(self) -> int:
        return int(self.model.runtime_prefix_replay_len())

    def runtime_batch_capacity(self) -> int:
        return int(self.model.runtime_batch_capacity())

    def rebuild_runtime_buffers(self) -> None:
        _rebuild_runtime_buffers(
            self.model,
            precompute_freqs_cis=self._precompute_freqs_cis,
        )

    def sync_runtime_batch_capacity(self, max_batch_size: int) -> int:
        _ensure_batch_capacity(self.model, int(max_batch_size))
        return int(self.model.runtime_batch_capacity())

    def reset_runtime_cache(self) -> None:
        _reset_runtime_cache(self.model)

    def refresh_state_buffers(self) -> None:
        _refresh_state_buffers(self.model)

    def forward_with_last_hidden(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int = 0,
        return_all_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _enable_runtime_cache(self.model)
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
        _enable_runtime_cache(self.model)
        return self._replay_with_cache(
            input_ids,
            start_pos=int(start_pos),
            return_all_logits=return_all_logits,
        )

    def _forward_with_last_hidden(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int = 0,
        return_all_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        model = self.model
        hidden, _ = model._forward_hidden(input_ids, start_pos=int(start_pos))
        last_hidden = hidden[:, -1, :]
        if not return_all_logits:
            hidden = hidden[:, -1:, :]
        logits = model.head_mixer(hidden, norm=model.norm, output=model.output)
        if not return_all_logits:
            logits = logits[:, 0, :]
        return logits, last_hidden

    def _replay_with_cache(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int = 0,
        return_all_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if int(input_ids.size(1)) <= 1 or int(start_pos) == 0:
            return self._forward_with_last_hidden(
                input_ids,
                start_pos=int(start_pos),
                return_all_logits=return_all_logits,
            )

        logits_all: list[torch.Tensor] = []
        last_hidden = None
        for offset in range(int(input_ids.size(1))):
            logits_step, last_hidden = self._forward_with_last_hidden(
                input_ids[:, offset : offset + 1],
                start_pos=int(start_pos) + int(offset),
                return_all_logits=True,
            )
            logits_all.append(logits_step)

        logits = torch.cat(logits_all, dim=1)
        if not return_all_logits:
            logits = logits[:, -1, :]
        return logits, last_hidden

    @staticmethod
    def cache_batch_size(batch_size: int | None, max_batch_size: int) -> int:
        active_batch = int(max_batch_size if batch_size is None else batch_size)
        if active_batch <= 0:
            raise ValueError(f"batch_size must be >= 1, got {active_batch}")
        return active_batch

    def cache_dump(
        self,
        *,
        device: str = "cpu",
        cache_pos: int | None = None,
        batch_size: int | None = None,
    ) -> RuntimeCacheSnapshot:
        del cache_pos
        active_batch = self.cache_batch_size(batch_size, self.model.runtime_batch_capacity())
        self.model.ensure_batch_capacity(active_batch)
        return RuntimeCacheSnapshot(
            layers=tuple(
                _dump_full_attn_snapshot(layer.attn, device=device, batch_size=active_batch)
                for layer in self.model.layers
            ),
        )

    def cache_dump_prefix(
        self,
        *,
        device: str = "cpu",
        prefix_len: int | None = None,
        batch_size: int | None = None,
    ) -> RuntimeCacheSnapshot:
        # Prefix dump is a full cache dump; prefix_len is bounded by the live cache.
        del prefix_len
        return self.cache_dump(device=device, batch_size=batch_size)

    def cache_load(self, cache_snapshot: RuntimeCacheSnapshot) -> None:
        _enable_runtime_cache(self.model)
        model = self.model
        if not isinstance(cache_snapshot, RuntimeCacheSnapshot):
            raise TypeError("cache snapshot must be a RuntimeCacheSnapshot")
        if len(cache_snapshot.layers) > len(model.layers):
            raise ValueError(
                "cache snapshot has more transformer layers than the model runtime"
            )

        required_batch = cache_snapshot.batch_size()
        if required_batch > 0:
            model.ensure_batch_capacity(required_batch)

        for index, snapshot in enumerate(cache_snapshot.layers):
            _validate_attn_cache_snapshot(f"layer_{index}", model.layers[index].attn, snapshot)

        _reset_runtime_cache(model)

        for index, snapshot in enumerate(cache_snapshot.layers):
            _load_attn_cache_snapshot(model.layers[index].attn, snapshot)

    def recompute_state_cache(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _enable_runtime_cache(self.model)
        return self._replay_with_cache(
            input_ids,
            start_pos=int(start_pos),
            return_all_logits=False,
        )


__all__ = ["TransformerRuntime"]
