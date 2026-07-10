from __future__ import annotations

from ml.runtime.api import (
    SupportsRuntimeStateRefreshControl,
    SupportsRuntimeStateResetControl,
    resolve_runtime_control,
)

RuntimeControlTarget = (
    SupportsRuntimeStateRefreshControl
    | SupportsRuntimeStateResetControl
)


def maybe_refresh_runtime_state_buffers(model: RuntimeControlTarget) -> bool:
    control = resolve_runtime_control(model, SupportsRuntimeStateRefreshControl)
    if control is None:
        return False
    control.refresh_state_buffers()
    return True


def maybe_reset_runtime_cache(model: RuntimeControlTarget) -> bool:
    control = resolve_runtime_control(model, SupportsRuntimeStateResetControl)
    if control is None:
        return False
    control.reset_runtime_cache()
    return True
