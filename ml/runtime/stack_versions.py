from __future__ import annotations

import importlib.metadata

from ml.runtime.stack_env import (
    Mode,
    disable_optional_ml_backends,
    require_linux,
    require_python,
)


def warn_if_numpy_unsupported(*, max_major: int = 2) -> str | None:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        return (
            f"NumPy is missing ({exc}). "
            f"numpy<{int(max_major) + 1} is required (see requirements.txt)."
        )
    except Exception as exc:
        return (
            f"NumPy import failed ({exc}). "
            f"numpy<{int(max_major) + 1} is required (see requirements.txt)."
        )
    try:
        major = int(str(np.__version__).split(".", 1)[0])
    except (TypeError, ValueError, IndexError) as exc:
        return (
            f"Unable to parse NumPy version {getattr(np, '__version__', None)!r} ({exc})."
        )
    if major > int(max_major):
        return (
            f"NumPy {np.__version__} detected; "
            f"numpy<{int(max_major) + 1} is required (see requirements.txt)."
        )
    return None


def require_numpy_supported(*, max_major: int = 2) -> None:
    warn = warn_if_numpy_unsupported(max_major=max_major)
    if warn:
        raise RuntimeError(
            warn + " Install requirements.txt (recommended: a clean venv)."
        )


def require_package_exact_version(
    *,
    package: str,
    expected_version: str,
    install_hint: str = "pip install -r requirements.txt",
) -> str:
    try:
        ver = importlib.metadata.version(str(package))
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency: {package}. Install with: {install_hint}"
        ) from exc

    if str(ver) != str(expected_version):
        raise RuntimeError(
            f"{package}=={expected_version} is required (found {ver}).\n"
            f"Reinstall the pinned versions ({install_hint})."
        )
    return str(ver)


PINNED_TORCH_VERSION = "2.8.0+cu128"
PINNED_TRANSFORMERS_VERSION = "5.10.2"
PINNED_ACCELERATE_VERSION = "1.14.0"
PINNED_TOKENIZERS_VERSION = "0.22.2"
PINNED_SAFETENSORS_VERSION = "0.7.0"
PINNED_PYARROW_VERSION = "24.0.0"
PINNED_DATASETS_VERSION = "4.8.5"
PINNED_HF_HUB_VERSION = "1.22.0"
PINNED_LIGER_KERNEL_VERSION = "0.8.0"
PINNED_TENSORBOARD_VERSION = "2.20.0"
PINNED_NUMPY_VERSION = "2.3.2"
PINNED_ORJSON_VERSION = "3.11.7"
PINNED_JINJA2_VERSION = "3.1.6"


def ensure_standard_stack(
    *,
    mode: Mode,
    require_tensorboard: bool = False,
) -> dict[str, str]:
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in {"train", "infer", "tools"}:
        raise ValueError(f"unsupported Sophia stack mode: {mode!r}")

    disable_optional_ml_backends()
    require_python(min_version=(3, 12))
    if normalized_mode in {"train", "infer"}:
        require_linux(for_training=True)
    require_numpy_supported(max_major=2)

    versions = {
        "torch": require_package_exact_version(
            package="torch",
            expected_version=PINNED_TORCH_VERSION,
        ),
        "transformers": require_package_exact_version(
            package="transformers",
            expected_version=PINNED_TRANSFORMERS_VERSION,
        ),
        "accelerate": require_package_exact_version(
            package="accelerate",
            expected_version=PINNED_ACCELERATE_VERSION,
        ),
        "tokenizers": require_package_exact_version(
            package="tokenizers",
            expected_version=PINNED_TOKENIZERS_VERSION,
        ),
        "pyarrow": require_package_exact_version(
            package="pyarrow",
            expected_version=PINNED_PYARROW_VERSION,
        ),
        "datasets": require_package_exact_version(
            package="datasets",
            expected_version=PINNED_DATASETS_VERSION,
        ),
        "huggingface-hub": require_package_exact_version(
            package="huggingface-hub",
            expected_version=PINNED_HF_HUB_VERSION,
        ),
        "safetensors": require_package_exact_version(
            package="safetensors",
            expected_version=PINNED_SAFETENSORS_VERSION,
        ),
        "numpy": require_package_exact_version(
            package="numpy",
            expected_version=PINNED_NUMPY_VERSION,
        ),
        "orjson": require_package_exact_version(
            package="orjson",
            expected_version=PINNED_ORJSON_VERSION,
        ),
        "jinja2": require_package_exact_version(
            package="jinja2",
            expected_version=PINNED_JINJA2_VERSION,
        ),
        "liger-kernel": require_package_exact_version(
            package="liger-kernel",
            expected_version=PINNED_LIGER_KERNEL_VERSION,
        ),
    }
    if require_tensorboard:
        versions["tensorboard"] = require_package_exact_version(
            package="tensorboard",
            expected_version=PINNED_TENSORBOARD_VERSION,
        )
    return versions


__all__ = [
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
    "ensure_standard_stack",
    "require_numpy_supported",
    "require_package_exact_version",
    "warn_if_numpy_unsupported",
]
