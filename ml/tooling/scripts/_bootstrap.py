from __future__ import annotations

import sys
from pathlib import Path

from typing import Literal

Mode = Literal["train", "infer", "tools"]

ML_SRC_ROOT = Path(__file__).resolve().parents[3]
# Repository root for the ML project (contains `dataset/`, `pyproject.toml`, etc.).
REPO_ROOT = ML_SRC_ROOT

__all__ = ["Mode", "ML_SRC_ROOT", "REPO_ROOT", "bootstrap"]


def bootstrap(*, mode: Mode = "tools") -> None:
    """
    Shared bootstrap for the tooling scripts.

    - ensures the repository root is importable when invoked as `python -m ml...`
    - forces UTF-8 stdio (best-effort) to avoid mojibake in logs
    - disables optional Transformers backends (TF/Flax/JAX) early
    """
    if str(ML_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(ML_SRC_ROOT))

    # Import after sys.path bootstrap so module execution works even without editable installs.
    from ml.runtime.stack import (
        disable_optional_ml_backends,
        ensure_standard_stack,
    )

    disable_optional_ml_backends()
    ensure_standard_stack(mode=mode)
