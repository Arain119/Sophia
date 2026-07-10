"""Runtime transformer implementation used by the training stack."""

from __future__ import annotations

from threading import RLock

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ml.runtime.model.config import ModelArgs
from ml.runtime.model.rope import precompute_freqs_cis
from ml.runtime.model.ops import (
    apply_preserving_complex_buffers as _apply_preserving_complex_buffers,
)
from ml.runtime.model.runtime_host import TransformerRuntime
from ml.runtime.model.state import RuntimeCacheSnapshot
from ml.runtime.model.transformer_setup import init_weights, initialize_transformer


class Transformer(nn.Module):
    """
    Runtime Transformer implementation.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        initialize_transformer(
            self,
            args,
            precompute_freqs_cis_fn=precompute_freqs_cis,
        )

    def _apply(self, fn):
        return _apply_preserving_complex_buffers(
            self,
            fn,
            buffer_names=("freqs_cis",),
            apply_super=lambda: super(Transformer, self)._apply(fn),
        )

    def _init_weights(self) -> None:
        init_weights(self)

    @property
    def runtime_model(self):
        return self

    @property
    def runtime_host(self) -> TransformerRuntime:
        return self.runtime

    @property
    def runtime_lock(self) -> RLock:
        return self.runtime.lock

    @property
    def prefix_cache_namespace(self) -> str:
        return str(self._prefix_cache_namespace)

    @property
    def _runtime_lock(self) -> RLock:
        return self.runtime_lock

    @_runtime_lock.setter
    def _runtime_lock(self, value: RLock) -> None:
        self.runtime.lock = value

    def ensure_batch_capacity(self, batch_size: int) -> None:
        self.runtime.ensure_batch_capacity(int(batch_size))

    def ensure_sequence_capacity(self, max_seq_len: int) -> None:
        self.runtime.ensure_sequence_capacity(int(max_seq_len))

    def ensure_runtime_max_seq_len(self, max_seq_len: int) -> None:
        self.ensure_sequence_capacity(int(max_seq_len))

    def rebuild_runtime_buffers(self) -> None:
        self.runtime.rebuild_runtime_buffers()

    def reset_runtime_cache(self) -> None:
        self.runtime.reset_runtime_cache()

    def refresh_state_buffers(self) -> None:
        self.runtime.refresh_state_buffers()

    def supports_loss_chunk_size(self) -> bool:
        return self.runtime.supports_loss_chunk_size()

    def supports_checkpoint_excludes(self) -> bool:
        return self.runtime.supports_checkpoint_excludes()

    def runtime_max_seq_len(self) -> int:
        return int(self.args.runtime_max_seq_len())

    def runtime_sequence_capacity(self) -> int:
        return int(self.args.runtime_sequence_capacity())

    def runtime_prefix_replay_len(self) -> int:
        return int(self.args.runtime_prefix_replay_len())

    def runtime_batch_capacity(self) -> int:
        return int(self.args.runtime_batch_capacity())

    def runtime_rope_kwargs(
        self,
        *,
        max_seq_len: int | None = None,
    ) -> dict[str, int | float]:
        return self.args.runtime_rope_kwargs(max_seq_len=max_seq_len)

    def runtime_reserve_batch_capacity(self, batch_size: int) -> int:
        return int(self.args.ensure_runtime_batch_capacity(int(batch_size)))

    def runtime_reserve_sequence_capacity(self, max_seq_len: int) -> int:
        return int(self.args.ensure_runtime_sequence_capacity(int(max_seq_len)))

    def runtime_recipe_knobs(self) -> tuple[int, int, int]:
        return self.runtime.runtime_recipe_knobs()

    def apply_runtime_recipe_knobs(
        self,
        *,
        loss_chunk_size: int,
        gradient_checkpointing_exclude_first: int,
        gradient_checkpointing_exclude_last: int,
    ) -> tuple[int, int, int]:
        return self.runtime.apply_runtime_recipe_knobs(
            loss_chunk_size=int(loss_chunk_size),
            gradient_checkpointing_exclude_first=int(
                gradient_checkpointing_exclude_first
            ),
            gradient_checkpointing_exclude_last=int(
                gradient_checkpointing_exclude_last
            ),
        )

    def sync_runtime_batch_capacity(self, max_batch_size: int) -> int:
        return self.runtime.sync_runtime_batch_capacity(int(max_batch_size))

    def forward_with_last_hidden(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        return_all_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.runtime.forward_with_last_hidden(
            input_ids,
            start_pos=int(start_pos),
            return_all_logits=return_all_logits,
        )

    def replay_with_cache(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        return_all_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.runtime.replay_with_cache(
            input_ids,
            start_pos=int(start_pos),
            return_all_logits=return_all_logits,
        )

    @staticmethod
    def _cache_batch_size(batch_size: int | None, max_batch_size: int) -> int:
        return TransformerRuntime.cache_batch_size(batch_size, int(max_batch_size))

    def cache_dump(
        self,
        device: str = "cpu",
        *,
        cache_pos: int | None = None,
        batch_size: int | None = None,
    ) -> RuntimeCacheSnapshot:
        return self.runtime.cache_dump(
            device=device,
            cache_pos=cache_pos,
            batch_size=batch_size,
        )

    def cache_dump_prefix(
        self,
        device: str = "cpu",
        *,
        prefix_len: int | None = None,
        batch_size: int | None = None,
    ) -> RuntimeCacheSnapshot:
        return self.runtime.cache_dump_prefix(
            device=device,
            prefix_len=prefix_len,
            batch_size=batch_size,
        )

    def cache_load(self, cache_snapshot: RuntimeCacheSnapshot) -> None:
        self.runtime.cache_load(cache_snapshot)

    def recompute_state_cache(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.runtime.recompute_state_cache(
            input_ids,
            start_pos=int(start_pos),
        )

    def _validate_forward_inputs(self, input_ids: torch.Tensor, start_pos: int) -> None:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [B, S]")
        if int(start_pos) < 0:
            raise ValueError(f"start_pos must be >= 0, got {start_pos}")
        if int(start_pos) > 0 and int(input_ids.size(1)) > 1:
            raise ValueError(
                "multi-token incremental forward is not supported when start_pos > 0; "
                f"got start_pos={start_pos}, seqlen={int(input_ids.size(1))}"
            )
        capacity = int(self.runtime_sequence_capacity())
        if int(start_pos) + int(input_ids.size(1)) > capacity:
            raise ValueError(
                "sequence exceeds max_seq_len: "
                f"start_pos={start_pos}, seqlen={int(input_ids.size(1))}, "
                f"max_seq_len={capacity}"
            )

    def _should_checkpoint_main_layer(
        self,
        *,
        layer_idx: int,
        total_layers: int,
        start_pos: int,
    ) -> bool:
        if not bool(self.gradient_checkpointing):
            return False
        if not self.training or int(start_pos) != 0:
            return False
        exclude_first = max(int(self.gradient_checkpointing_exclude_first), 0)
        exclude_last = max(int(self.gradient_checkpointing_exclude_last), 0)
        idx = int(layer_idx)
        total = max(int(total_layers), 0)
        if idx < exclude_first:
            return False
        if idx >= max(total - exclude_last, 0):
            return False
        return True

    def _embedded_inputs(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen = input_ids.shape
        stateless_full_sequence = int(start_pos) == 0 and not bool(self.args.use_cache)
        if not stateless_full_sequence:
            self.ensure_batch_capacity(int(bsz))

        hidden = self.tok_embeddings(input_ids)
        hidden = self.dropout(hidden)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        return hidden, freqs_cis

    def _run_main_layers(
        self,
        hidden: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        layers = tuple(self.layers)
        for layer_idx, layer in enumerate(layers):
            if self._should_checkpoint_main_layer(
                layer_idx=layer_idx,
                total_layers=len(layers),
                start_pos=int(start_pos),
            ):
                def layer_fn(x, layer=layer):
                    return layer(x, freqs_cis, start_pos=start_pos)

                hidden = checkpoint(layer_fn, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden, freqs_cis, start_pos=start_pos)
        return hidden

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_forward_inputs(input_ids, int(start_pos))
        hidden, freqs_cis = self._embedded_inputs(input_ids, int(start_pos))
        hidden = self._run_main_layers(hidden, freqs_cis, int(start_pos))
        return hidden, freqs_cis

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        if attention_mask is not None and not torch.is_tensor(attention_mask):
            raise TypeError(
                "attention_mask must be a tensor or None; pass start_pos as a keyword argument"
            )
        hidden, _ = self._forward_hidden(input_ids, start_pos=int(start_pos))
        return self.head_mixer(hidden, norm=self.norm, output=self.output)

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.args.vocab_size}, "
            f"dim={self.args.dim}, "
            f"n_layers={self.args.n_layers}, "
            f"n_heads={self.args.n_heads}"
        )

    def forward_full(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        return self.forward(input_ids, attention_mask=attention_mask, start_pos=start_pos)


__all__ = [
    "ModelArgs",
    "Transformer",
]
