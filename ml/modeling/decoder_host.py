from __future__ import annotations

from typing import Protocol

from ml.modeling.runtime_backend import (
    RuntimeBackend,
)


class DecoderConfig(Protocol):
    def to_model_args(
        self,
        *,
        runtime_max_seq_len: int | None = None,
    ) -> object: ...


class DecoderHostMixin:
    def _initialize_decoder_runtime(
        self,
        *,
        config: DecoderConfig,
        runtime_max_seq_len: int | None,
        runtime_backend: RuntimeBackend,
        gradient_checkpointing_enabled: bool,
        register_tied_weights: bool = False,
    ) -> None:
        self.model = runtime_backend.transformer_cls(
            config.to_model_args(runtime_max_seq_len=runtime_max_seq_len)
        )
        self.runtime = self.model.runtime
        if bool(register_tied_weights):
            self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(
                all_submodels=False
            )
        # Cache-enabled runtime execution mutates shared KV/index buffers, so a
        # single model instance cannot safely run concurrent calls.
        self._model_runtime_lock = self.model.runtime_lock
        self._sync_runtime_gradient_checkpointing(
            enable=bool(gradient_checkpointing_enabled)
        )

__all__ = ["DecoderConfig", "DecoderHostMixin"]
