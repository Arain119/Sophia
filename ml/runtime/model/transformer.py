"""Native Sophia Hybrid runtime."""

from __future__ import annotations

from threading import RLock

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ml.runtime.model.config import ModelArgs
from ml.runtime.model.runtime_host import TransformerRuntime
from ml.runtime.model.state import RuntimeCacheSnapshot
from ml.runtime.model.transformer_setup import init_weights, initialize_transformer


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        initialize_transformer(self, args)

    def _apply(self, fn):
        result = super()._apply(fn)
        self.rebuild_runtime_buffers()
        return result

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
    def _runtime_lock(self) -> RLock:
        return self.runtime_lock

    @_runtime_lock.setter
    def _runtime_lock(self, value: RLock) -> None:
        self.runtime.lock = value

    def ensure_batch_capacity(self, batch_size: int) -> None:
        self.runtime.ensure_batch_capacity(batch_size)

    def ensure_sequence_capacity(self, max_seq_len: int) -> None:
        self.runtime.ensure_sequence_capacity(max_seq_len)

    def ensure_runtime_max_seq_len(self, max_seq_len: int) -> None:
        self.ensure_sequence_capacity(max_seq_len)

    def rebuild_runtime_buffers(self) -> None:
        self.runtime.rebuild_runtime_buffers()

    def reset_runtime_cache(self) -> None:
        self.runtime.reset_runtime_cache()

    def refresh_state_buffers(self) -> None:
        self.runtime.refresh_state_buffers()

    def supports_loss_chunk_size(self) -> bool:
        return True

    def supports_checkpoint_excludes(self) -> bool:
        return True

    def runtime_max_seq_len(self) -> int:
        return self.args.runtime_max_seq_len()

    def runtime_sequence_capacity(self) -> int:
        return self.args.runtime_sequence_capacity()

    def runtime_batch_capacity(self) -> int:
        return self.args.runtime_batch_capacity()

    def runtime_reserve_batch_capacity(self, batch_size: int) -> int:
        return self.args.ensure_runtime_batch_capacity(batch_size)

    def runtime_reserve_sequence_capacity(self, max_seq_len: int) -> int:
        return self.args.ensure_runtime_sequence_capacity(max_seq_len)

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
            loss_chunk_size=loss_chunk_size,
            gradient_checkpointing_exclude_first=gradient_checkpointing_exclude_first,
            gradient_checkpointing_exclude_last=gradient_checkpointing_exclude_last,
        )

    def sync_runtime_batch_capacity(self, max_batch_size: int) -> int:
        return self.runtime.sync_runtime_batch_capacity(max_batch_size)

    def forward_with_last_hidden(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        return_all_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.runtime.forward_with_last_hidden(
            input_ids,
            start_pos=start_pos,
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
            start_pos=start_pos,
            return_all_logits=return_all_logits,
        )

    def cache_dump(
        self,
        device: str = "cpu",
        *,
        cache_pos: int | None = None,
        batch_size: int | None = None,
    ) -> RuntimeCacheSnapshot:
        return self.runtime.cache_dump(
            device=device, cache_pos=cache_pos, batch_size=batch_size
        )

    def cache_load(self, cache_snapshot: RuntimeCacheSnapshot) -> None:
        self.runtime.cache_load(cache_snapshot)

    def _validate_forward_inputs(self, input_ids: torch.Tensor, start_pos: int) -> None:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [B, S]")
        if int(start_pos) < 0:
            raise ValueError(f"start_pos must be >= 0, got {start_pos}")
        if int(start_pos) > 0 and int(input_ids.size(1)) > 1:
            raise ValueError(
                "incremental forward accepts one token when start_pos > 0"
            )
        if int(start_pos) + int(input_ids.size(1)) > self.runtime_sequence_capacity():
            raise ValueError(
                "sequence exceeds max_seq_len: "
                f"start_pos={start_pos}, seqlen={int(input_ids.size(1))}, "
                f"max_seq_len={self.runtime_sequence_capacity()}"
            )

    def _should_checkpoint_main_layer(self, layer_idx: int, start_pos: int) -> bool:
        if not self.gradient_checkpointing or not self.training or int(start_pos) != 0:
            return False
        first = int(self.gradient_checkpointing_exclude_first)
        last = int(self.gradient_checkpointing_exclude_last)
        return first <= layer_idx < len(self.layers) - last

    def _run_main_layers(
        self, hidden: torch.Tensor, start_pos: int
    ) -> torch.Tensor:
        completed_blocks = hidden.new_empty(
            int(hidden.size(0)), int(hidden.size(1)), 0, int(hidden.size(2))
        )
        for layer_idx, layer in enumerate(self.layers):
            if self._should_checkpoint_main_layer(layer_idx, start_pos):
                hidden, completed_blocks = checkpoint(
                    lambda value, blocks, layer=layer: layer(
                        value, blocks, start_pos=start_pos
                    ),
                    hidden,
                    completed_blocks,
                    use_reentrant=False,
                )
            else:
                hidden, completed_blocks = layer(
                    hidden, completed_blocks, start_pos=start_pos
                )
        return self.output_attn_residual(hidden, completed_blocks)

    def _forward_hidden(
        self, input_ids: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, None]:
        self._validate_forward_inputs(input_ids, start_pos)
        stateless = int(start_pos) == 0 and not bool(self.args.use_cache)
        if not stateless:
            self.ensure_batch_capacity(int(input_ids.size(0)))
        hidden = self.dropout(self.tok_embeddings(input_ids))
        return self._run_main_layers(hidden, start_pos), None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        if attention_mask is not None and not torch.is_tensor(attention_mask):
            raise TypeError("attention_mask must be a tensor or None")
        hidden, _ = self._forward_hidden(input_ids, start_pos=start_pos)
        return self.head_mixer(hidden, norm=self.norm, output=self.output)

    def forward_full(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        return self.forward(input_ids, attention_mask=attention_mask, start_pos=start_pos)

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.args.vocab_size}, dim={self.args.dim}, "
            f"n_layers={self.args.n_layers}, num_heads={self.args.num_heads}, "
            "pattern=KDA,KDA,KDA,MLA"
        )


__all__ = ["ModelArgs", "Transformer"]
