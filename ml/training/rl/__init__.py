"""Reinforcement-learning stage: rollout, reward, and policy optimisation."""

from .reward import RewardBreakdown, RewardWeights, score_completion
from .rollout import Completion, RolloutPrompt, group_has_signal, rollout_group
from .grpo import group_advantages, grpo_loss
from .dpo import (
    PreferencePair,
    dpo_accuracy,
    dpo_loss,
    read_preference_pairs,
    sequence_logprob,
    token_logprobs,
    write_preference_pairs,
)

__all__ = [
    "Completion",
    "RewardBreakdown",
    "RewardWeights",
    "RolloutPrompt",
    "group_has_signal",
    "rollout_group",
    "score_completion",
    "PreferencePair",
    "dpo_accuracy",
    "dpo_loss",
    "read_preference_pairs",
    "sequence_logprob",
    "token_logprobs",
    "write_preference_pairs",
    "group_advantages",
    "grpo_loss",
]
