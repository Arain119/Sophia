from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, TypeVar

from ml.runtime.contracts import RuntimeBackedModel

if TYPE_CHECKING:
    from types import ModuleType


RUNTIME_MODULE_NAME = "ml.runtime.model.transformer"
RuntimeModelArgsT = TypeVar("RuntimeModelArgsT")
RuntimeTransformerT = TypeVar("RuntimeTransformerT", bound=RuntimeBackedModel)


@dataclass(frozen=True)
class RuntimeBackend:
    module: ModuleType | None
    model_args_cls: type[RuntimeModelArgsT]
    transformer_cls: type[RuntimeTransformerT]


def _import_runtime_module(module_name: str) -> ModuleType:
    if module_name.startswith("."):
        package = __package__
        if not package:
            raise ImportError(
                "relative runtime module import requires a package context"
            )
        return import_module(module_name, package=package)
    return import_module(module_name)


def resolve_runtime_backend() -> RuntimeBackend:
    runtime_module = _import_runtime_module(RUNTIME_MODULE_NAME)
    return RuntimeBackend(
        module=runtime_module,
        model_args_cls=runtime_module.ModelArgs,
        transformer_cls=runtime_module.Transformer,
    )


__all__ = [
    "RUNTIME_MODULE_NAME",
    "RuntimeBackend",
    "resolve_runtime_backend",
]
