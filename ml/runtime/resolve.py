from __future__ import annotations

from threading import RLock

from ml.runtime.contracts import (
    RuntimeBackedModel,
    RuntimeHost,
    Runtime,
)


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


def resolve_runtime_control[RuntimeControlT](
    model: object,
    control_type: type[RuntimeControlT],
) -> RuntimeControlT | None:
    runtime_host = resolve_runtime_host(model)
    if runtime_host is not None and isinstance(runtime_host, control_type):
        return runtime_host
    if isinstance(model, control_type):
        return model
    return None


__all__ = [
    "resolve_runtime_control",
    "resolve_runtime_host",
    "resolve_runtime_lock",
]
