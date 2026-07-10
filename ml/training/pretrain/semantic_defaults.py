from __future__ import annotations

import math

from ml.training.pretrain.release_config import RELEASE_PRETRAIN_DEFAULTS

# Canonical source lives in the spec layer (keeps runtime self-contained); re-export
# here so existing training/tooling consumers keep importing it from this module.
from ml.core.spec.semantics import PINNED_SEMANTIC_ADAM_EPS

PINNED_SEMANTIC_TARGET_TOKENS_PER_UPDATE = 8_192
PINNED_SEMANTIC_LEARNING_RATE = 1.9091883092036785e-04
PINNED_SEMANTIC_WEIGHT_DECAY = 0.05
# Release pretrain is cost-bound on the RTX 5090 target machine. This update
# budget is the pinned measured production update budget for the 0.8B / seq4096 run.
PINNED_RELEASE_PRETRAIN_TARGET_TOKENS_PER_UPDATE = (
    RELEASE_PRETRAIN_DEFAULTS.target_tokens_per_update
)
# RTX 5090 canaries use these semantic optimizer values until a later signed
# recipe supersedes them.
PINNED_RELEASE_PRETRAIN_LEARNING_RATE = RELEASE_PRETRAIN_DEFAULTS.learning_rate
PINNED_RELEASE_PRETRAIN_ADAM_EPS = RELEASE_PRETRAIN_DEFAULTS.adam_eps
PINNED_RELEASE_PRETRAIN_EMBEDDING_LR_SCALE = (
    RELEASE_PRETRAIN_DEFAULTS.embedding_lr_scale
)
PINNED_RELEASE_PRETRAIN_EMA_DECAY = RELEASE_PRETRAIN_DEFAULTS.ema_decay


def learning_rate_scale_for_tokens_per_update(
    *,
    tokens_per_update: int,
    target_tokens_per_update: int,
) -> float:
    tokens = max(int(tokens_per_update), 1)
    target = max(int(target_tokens_per_update), 1)
    return math.sqrt(float(tokens) / float(target))


def learning_rate_for_tokens_per_update(
    *,
    semantic_learning_rate: float,
    tokens_per_update: int,
    target_tokens_per_update: int,
) -> float:
    return float(semantic_learning_rate) * float(
        learning_rate_scale_for_tokens_per_update(
            tokens_per_update=int(tokens_per_update),
            target_tokens_per_update=int(target_tokens_per_update),
        )
    )


def rescale_learning_rate_between_token_budgets(
    *,
    learning_rate: float,
    from_tokens_per_update: int,
    to_tokens_per_update: int,
) -> float:
    from_tokens = max(int(from_tokens_per_update), 1)
    to_tokens = max(int(to_tokens_per_update), 1)
    return float(learning_rate) * math.sqrt(float(to_tokens) / float(from_tokens))


def semantic_learning_rate_for_target_tokens_per_update(
    *,
    target_tokens_per_update: int,
    base_target_tokens_per_update: int = PINNED_SEMANTIC_TARGET_TOKENS_PER_UPDATE,
    base_learning_rate: float = PINNED_SEMANTIC_LEARNING_RATE,
) -> float:
    target = max(int(target_tokens_per_update), 1)
    base_target = max(int(base_target_tokens_per_update), 1)
    return float(base_learning_rate) * math.sqrt(float(target) / float(base_target))


__all__ = [
    "RELEASE_PRETRAIN_DEFAULTS",
    "PINNED_RELEASE_PRETRAIN_ADAM_EPS",
    "PINNED_RELEASE_PRETRAIN_EMA_DECAY",
    "PINNED_RELEASE_PRETRAIN_EMBEDDING_LR_SCALE",
    "PINNED_RELEASE_PRETRAIN_TARGET_TOKENS_PER_UPDATE",
    "PINNED_RELEASE_PRETRAIN_LEARNING_RATE",
    "PINNED_SEMANTIC_ADAM_EPS",
    "PINNED_SEMANTIC_LEARNING_RATE",
    "PINNED_SEMANTIC_TARGET_TOKENS_PER_UPDATE",
    "PINNED_SEMANTIC_WEIGHT_DECAY",
    "learning_rate_for_tokens_per_update",
    "learning_rate_scale_for_tokens_per_update",
    "rescale_learning_rate_between_token_budgets",
    "semantic_learning_rate_for_target_tokens_per_update",
]
