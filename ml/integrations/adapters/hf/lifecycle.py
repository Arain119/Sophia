from __future__ import annotations

from typing import TypeVar, cast


ModelT = TypeVar("ModelT", bound="PreTrainedLifecycleMixin")


class PreTrainedLifecycleMixin:
    @classmethod
    def from_pretrained(
        cls: type[ModelT],
        pretrained_model_name_or_path,
        *model_args: object,
        **kwargs: object,
    ) -> ModelT:
        model = cast(
            ModelT,
            super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                **kwargs,
            ),
        )
        with model._model_runtime_lock:
            model._rebuild_runtime_buffers()
        return model


__all__ = ["PreTrainedLifecycleMixin"]
