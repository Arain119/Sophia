from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Protocol

import torch
from torch import nn

from ml.runtime.model.state import RuntimeCacheSnapshot
from ml.runtime.contracts import RuntimeHost


class DecoderRuntimeConfig(Protocol):
    max_seq_len: int
    max_batch_size: int
    loss_chunk_size: int
    gradient_checkpointing_exclude_first: int
    gradient_checkpointing_exclude_last: int


class DecoderRuntimeModel(Protocol):
    tok_embeddings: nn.Module
    output: nn.Module
    gradient_checkpointing: bool
    gradient_checkpointing_exclude_first: int
    gradient_checkpointing_exclude_last: int
    def modules(self): ...
    def parameters(self, recurse: bool = True): ...


class DecoderRuntimeBase:
    def _runtime_model(self) -> DecoderRuntimeModel:
        from typing import cast

        return cast(DecoderRuntimeModel, self.model)

    def _runtime_host(self) -> RuntimeHost:
        from typing import cast

        return cast(RuntimeHost, self.runtime_host)

    def _runtime_config(self) -> DecoderRuntimeConfig:
        from typing import cast

        return cast(DecoderRuntimeConfig, self.config)

    @property
    def runtime_model(self):
        return self._runtime_model()

    @property
    def runtime_host(self):
        return self.runtime

    @property
    def runtime_lock(self) -> RLock:
        return self._model_runtime_lock


class DecoderRuntimeMixin(DecoderRuntimeBase):
    def reset_runtime_cache(self) -> None:
        self._runtime_host().reset_runtime_cache()

    def refresh_state_buffers(self) -> None:
        self._runtime_host().refresh_state_buffers()

    def replay_with_cache(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        return_all_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._runtime_host().replay_with_cache(
            input_ids,
            start_pos=start_pos,
            return_all_logits=return_all_logits,
        )

    def forward_with_last_hidden(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        return_all_logits: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._runtime_host().forward_with_last_hidden(
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
        return self._runtime_host().cache_dump(
            device=device,
            cache_pos=cache_pos,
            batch_size=batch_size,
        )

    def cache_load(self, cache_snapshot: RuntimeCacheSnapshot) -> None:
        self._runtime_host().cache_load(cache_snapshot)

    def runtime_max_seq_len(self) -> int:
        return int(self._runtime_host().runtime_max_seq_len())

    def _sync_runtime_gradient_checkpointing(self, *, enable: bool) -> None:
        model = self._runtime_model()
        config = self._runtime_config()
        model.gradient_checkpointing = bool(enable)
        model.gradient_checkpointing_exclude_first = int(
            config.gradient_checkpointing_exclude_first
        )
        model.gradient_checkpointing_exclude_last = int(
            config.gradient_checkpointing_exclude_last
        )


class DecoderModelMixin(DecoderRuntimeBase):
    def _sync_tied_runtime_word_embeddings(self) -> None:
        config = getattr(self, "config", None)
        if not bool(getattr(config, "tie_word_embeddings", False)):
            return
        runtime_model = self._runtime_model()
        input_embeddings = runtime_model.tok_embeddings
        output_embeddings = runtime_model.output
        if not hasattr(input_embeddings, "weight") or not hasattr(
            output_embeddings, "weight"
        ):
            return
        input_weight = input_embeddings.weight
        output_weight = output_embeddings.weight
        if tuple(input_weight.shape) != tuple(output_weight.shape):
            raise ValueError(
                "tied word embeddings require matching embedding/output shapes; "
                f"got {tuple(input_weight.shape)!r} vs {tuple(output_weight.shape)!r}"
            )
        output_embeddings.weight = input_weight

    def _rebuild_runtime_buffers(self) -> None:
        self._runtime_host().rebuild_runtime_buffers()

    @staticmethod
    @contextmanager
    def get_input_embeddings(self) -> nn.Module:
        return self._runtime_model().tok_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        with self._model_runtime_lock:
            self.reset_runtime_cache()
            self._runtime_model().tok_embeddings = value
            self._sync_tied_runtime_word_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self._runtime_model().output

    def set_output_embeddings(self, value: nn.Module) -> None:
        with self._model_runtime_lock:
            self.reset_runtime_cache()
            self._runtime_model().output = value
            self._sync_tied_runtime_word_embeddings()

    def to(self, *args, **kwargs):
        with self._model_runtime_lock:
            runtime_model = self._runtime_model()
            complex_buffers: list[tuple[nn.Module, str, torch.Tensor]] = []
            for module in runtime_model.modules():
                for name, buf in list(getattr(module, "_buffers", {}).items()):
                    if torch.is_tensor(buf) and torch.is_complex(buf):
                        complex_buffers.append((module, name, module._buffers.pop(name)))
            try:
                module = super().to(*args, **kwargs)
            finally:
                root_param = next(runtime_model.parameters(), None)
                root_device = (
                    torch.device("cpu") if root_param is None else root_param.device
                )
                for owner, name, buf in complex_buffers:
                    owner_param = next(owner.parameters(), None)
                    target_device = (
                        root_device if owner_param is None else owner_param.device
                    )
                    owner.register_buffer(
                        name,
                        buf.to(device=target_device),
                        persistent=False,
                    )
                self._rebuild_runtime_buffers()
            return module

    def get_submodule(self, target: str) -> nn.Module:
        try:
            return super().get_submodule(target)
        except AttributeError:
            return self.model.get_submodule(target)

    def train(self, mode: bool = True):
        with self._model_runtime_lock:
            if bool(mode):
                self.reset_runtime_cache()
            return super().train(mode)

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        with self._model_runtime_lock:
            self.reset_runtime_cache()
            result = super().load_state_dict(
                state_dict,
                strict=strict,
                assign=assign,
            )
            self._sync_tied_runtime_word_embeddings()
            self._rebuild_runtime_buffers()
            return result


class DecoderRecipeMixin(DecoderRuntimeBase):
    def ensure_runtime_max_seq_len(self, max_seq_len: int) -> None:
        required = int(max_seq_len)
        config = self._runtime_config()
        if required <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
        if required > int(config.max_seq_len):
            raise ValueError(
                "runtime max_seq_len cannot exceed config.max_seq_len: "
                f"{required} > {int(config.max_seq_len)}"
            )
        with self._model_runtime_lock:
            self.reset_runtime_cache()
            self._runtime_host().ensure_runtime_max_seq_len(required)

    def supports_loss_chunk_size(self) -> bool:
        return bool(self._runtime_host().supports_loss_chunk_size())

    def supports_checkpoint_excludes(self) -> bool:
        return bool(self._runtime_host().supports_checkpoint_excludes())

    def runtime_recipe_knobs(self) -> tuple[int, int, int]:
        return self._runtime_host().runtime_recipe_knobs()

    def apply_runtime_recipe_knobs(
        self,
        *,
        loss_chunk_size: int,
        gradient_checkpointing_exclude_first: int,
        gradient_checkpointing_exclude_last: int,
    ) -> tuple[int, int, int]:
        with self._model_runtime_lock:
            chunk_size = int(loss_chunk_size)
            exclude_first = int(gradient_checkpointing_exclude_first)
            exclude_last = int(gradient_checkpointing_exclude_last)
            if chunk_size != 0 and not self.supports_loss_chunk_size():
                raise ValueError("decoder runtime does not support loss_chunk_size")
            if (exclude_first != 0 or exclude_last != 0) and not (
                self.supports_checkpoint_excludes()
            ):
                raise ValueError(
                    "decoder runtime does not support gradient checkpoint exclusions"
                )
            config = self._runtime_config()
            config.loss_chunk_size = int(chunk_size)
            config.gradient_checkpointing_exclude_first = int(exclude_first)
            config.gradient_checkpointing_exclude_last = int(exclude_last)
            return self._runtime_host().apply_runtime_recipe_knobs(
                loss_chunk_size=int(chunk_size),
                gradient_checkpointing_exclude_first=int(exclude_first),
                gradient_checkpointing_exclude_last=int(exclude_last),
            )

    def sync_runtime_batch_capacity(self, max_batch_size: int) -> int:
        with self._model_runtime_lock:
            batch_size = self._runtime_host().sync_runtime_batch_capacity(
                int(max_batch_size)
            )
            self._runtime_config().max_batch_size = int(batch_size)
            return int(batch_size)


__all__ = [
    "DecoderRuntimeConfig",
    "DecoderRuntimeBase",
    "DecoderRuntimeModel",
    "DecoderModelMixin",
    "DecoderRecipeMixin",
    "DecoderRuntimeMixin",
]
