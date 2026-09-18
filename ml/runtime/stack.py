"""Shared Sophia runtime stack API."""

from __future__ import annotations

from ml.runtime.stack_env import (
    Mode,
    disable_optional_ml_backends,
    enable_utf8_stdio,
    require_linux,
    require_python,
    set_low_cpu_noise_env,
)
from ml.runtime.stack_versions import (
    PINNED_ACCELERATE_VERSION,
    PINNED_DATASETS_VERSION,
    PINNED_HF_HUB_VERSION,
    PINNED_JINJA2_VERSION,
    PINNED_LIGER_KERNEL_VERSION,
    PINNED_NUMPY_VERSION,
    PINNED_ORJSON_VERSION,
    PINNED_PYARROW_VERSION,
    PINNED_SAFETENSORS_VERSION,
    PINNED_TENSORBOARD_VERSION,
    PINNED_TOKENIZERS_VERSION,
    PINNED_TORCH_VERSION,
    PINNED_TRANSFORMERS_VERSION,
    ensure_standard_stack,
    require_numpy_supported,
    require_package_exact_version,
    warn_if_numpy_unsupported,
)


__all__ = [
    "Mode",
    "PINNED_ACCELERATE_VERSION",
    "PINNED_DATASETS_VERSION",
    "PINNED_HF_HUB_VERSION",
    "PINNED_JINJA2_VERSION",
    "PINNED_LIGER_KERNEL_VERSION",
    "PINNED_NUMPY_VERSION",
    "PINNED_ORJSON_VERSION",
    "PINNED_PYARROW_VERSION",
    "PINNED_SAFETENSORS_VERSION",
    "PINNED_TENSORBOARD_VERSION",
    "PINNED_TOKENIZERS_VERSION",
    "PINNED_TORCH_VERSION",
    "PINNED_TRANSFORMERS_VERSION",
    "enable_utf8_stdio",
    "ensure_standard_stack",
    "require_linux",
    "require_numpy_supported",
    "require_package_exact_version",
    "require_python",
    "set_low_cpu_noise_env",
    "disable_optional_ml_backends",
    "warn_if_numpy_unsupported",
]
