from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import TypeVar

from ml.runtime.contracts import (
    RuntimeBackedModel,
    RuntimeHost,
    Runtime,
)


def resolve_runtime_model(model: object) -> RuntimeBackedModel | None:
    if isinstance(model, Runtime):
        return model.runtime_model
    if isinstance(model, RuntimeBackedModel):
        return model
    return None


def resolve_runtime_host(model: object) -> RuntimeHost | None:
    if isinstance(model, Runtime):
        return model.runtime_host
    if isinstance(model, RuntimeBackedModel):
        return model.runtime
    return None


def resolve_runtime_lock(model: object) -> RLock | None:
    if isinstance(model, Runtime):
        return model.runtime_lock
    if isinstance(model, RuntimeBackedModel):
        return model.runtime_lock
    return None


RuntimeControlT = TypeVar("RuntimeControlT")


def resolve_runtime_control(
    model: object,
    control_type: type[RuntimeControlT],
) -> RuntimeControlT | None:
    runtime_host = resolve_runtime_host(model)
    if runtime_host is not None and isinstance(runtime_host, control_type):
        return runtime_host
    if isinstance(model, control_type):
        return model
    return None


def resolve_runtime_method(model: object, method_name: str) -> Callable[..., object] | None:
    method_name = str(method_name)
    runtime_host = resolve_runtime_host(model)
    if runtime_host is not None:
        host_method = getattr(runtime_host, method_name, None)
        if callable(host_method):
            return host_method
    model_method = getattr(model, method_name, None)
    if callable(model_method):
        return model_method
    return None


__all__ = [
    "resolve_runtime_control",
    "resolve_runtime_host",
    "resolve_runtime_lock",
    "resolve_runtime_method",
    "resolve_runtime_model",
]
