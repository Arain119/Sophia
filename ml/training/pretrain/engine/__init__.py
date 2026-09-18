"""
Unified single-GPU training engine (pinned stack).

This package is intentionally limited to PyTorch. The pretraining entrypoint handles
model and tokenizer setup and delegates loop execution to the engine.
"""

from __future__ import annotations

from ml.training.pretrain.engine.io import AsyncJsonlWriter
from ml.training.pretrain.engine.loop_execution import train_loop
from ml.training.pretrain.engine.step_runner_impl import StepRunner


__all__ = ["AsyncJsonlWriter", "StepRunner", "train_loop"]
