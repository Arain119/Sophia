from __future__ import annotations

import os
import sys
from typing import Literal

Mode = Literal["train", "infer", "tools"]


def enable_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if not hasattr(stream, "reconfigure"):
            continue
        enc = getattr(stream, "encoding", None)
        if isinstance(enc, str) and enc.lower() == "utf-8":
            continue
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception as exc:
            raise RuntimeError("failed to configure UTF-8 standard streams") from exc


def require_linux(*, for_training: bool = True) -> None:
    if sys.platform.startswith("linux"):
        return
    if not for_training:
        return
    raise RuntimeError(
        f"Linux is required for GPU training/inference.\n"
        f"Detected platform: {sys.platform!r}\n"
        "Run training on a supported Linux GPU server."
    )


def disable_optional_ml_backends() -> None:
    enable_utf8_stdio()
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("USE_JAX", "0")


def _set_positive_int_env(name: str, default: int) -> None:
    raw = os.environ.get(name)
    if raw is None:
        os.environ[name] = str(max(int(default), 1))
        return
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be >= 1, got {value}")


def set_low_cpu_noise_env(*, omp_threads: int = 4, rayon_threads: int = 1) -> None:
    omp = max(int(omp_threads), 1)
    _set_positive_int_env("OMP_NUM_THREADS", omp)
    _set_positive_int_env("MKL_NUM_THREADS", omp)
    _set_positive_int_env("OPENBLAS_NUM_THREADS", omp)
    _set_positive_int_env("NUMEXPR_NUM_THREADS", omp)
    _set_positive_int_env("NUMEXPR_MAX_THREADS", omp)
    _set_positive_int_env("VECLIB_MAXIMUM_THREADS", omp)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _set_positive_int_env("RAYON_NUM_THREADS", max(int(rayon_threads), 1))


def require_python(*, min_version: tuple[int, int] = (3, 12)) -> None:
    if sys.version_info < (int(min_version[0]), int(min_version[1])):
        raise RuntimeError(
            f"Python>={min_version[0]}.{min_version[1]} is required "
            f"(got {sys.version_info.major}.{sys.version_info.minor})."
        )


__all__ = [
    "Mode",
    "disable_optional_ml_backends",
    "enable_utf8_stdio",
    "require_linux",
    "require_python",
    "set_low_cpu_noise_env",
]
