"""Pinned gradient-clip constants for the release pretrain stack.

Kept free of any ``ml.training.pretrain.engine`` import so lightweight config
modules (e.g. ``run_config.py``) can depend on the pinned values without
pulling in the full engine execution package as a side effect.
"""

from __future__ import annotations

PINNED_GRAD_CLIP_MODE = "hybrid_auto"
PINNED_AGC_CLIP = 0.01
PINNED_AGC_EPS = 1e-3
PINNED_AGC_EXCLUDE_BIAS_AND_NORM = True

__all__ = [
    "PINNED_AGC_CLIP",
    "PINNED_AGC_EPS",
    "PINNED_AGC_EXCLUDE_BIAS_AND_NORM",
    "PINNED_GRAD_CLIP_MODE",
]
