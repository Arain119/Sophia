"""Group-relative policy optimisation primitives.

GRPO uses disagreement inside one rollout group as its learning signal.  The
old-policy log probabilities are detached rollout-time values; the reference
log probabilities anchor the update to the SFT policy.  All operations are
token masked and accumulated in fp32, while the caller may keep model compute
in bf16.
"""

from __future__ import annotations

import torch


def group_advantages(
    rewards: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    min_std: float = 0.05,
) -> torch.Tensor:
    """Standardise rewards across each group (the first dimension)."""

    if rewards.ndim != 2:
        raise ValueError("rewards must be [groups, group_size]")
    if rewards.size(1) < 2:
        raise ValueError("a GRPO group needs at least two completions")
    mean = rewards.float().mean(dim=1, keepdim=True)
    std = rewards.float().std(dim=1, keepdim=True, unbiased=False)
    if bool((std < float(min_std)).any().item()):
        raise ValueError("GRPO reward group has no usable signal")
    return (rewards.float() - mean) / (std + float(eps))


def grpo_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.02,
) -> torch.Tensor:
    """Clipped GRPO surrogate plus a token-level reference KL penalty.

    Shapes are ``[groups, group_size, response_tokens]`` for log probabilities
    and mask, and ``[groups, group_size]`` for advantages.  The objective is
    averaged by response tokens rather than padded positions.
    """

    values = (policy_logprobs, old_logprobs, reference_logprobs, response_mask)
    if any(value.ndim != 3 for value in values):
        raise ValueError("GRPO log probabilities and response_mask must be [G,K,T]")
    if any(value.shape != policy_logprobs.shape for value in values[1:]):
        raise ValueError("GRPO sequence tensors must have identical shapes")
    if advantages.ndim != 2 or advantages.shape != policy_logprobs.shape[:2]:
        raise ValueError("advantages must be [G,K]")
    if not 0.0 < float(clip_epsilon) < 1.0:
        raise ValueError("GRPO clip_epsilon must be in (0, 1)")
    if float(kl_beta) < 0.0:
        raise ValueError("GRPO kl_beta must be non-negative")
    mask = response_mask.to(dtype=torch.float32)
    token_advantages = advantages.float().unsqueeze(-1)
    log_ratio = (policy_logprobs.float() - old_logprobs.float()).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    unclipped = ratio * token_advantages
    clipped = ratio.clamp(1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon)) * token_advantages
    surrogate = torch.minimum(unclipped, clipped)
    # k3-style stable KL approximation: exp(ref-policy ratio) - ratio - 1,
    # which is non-negative in expectation and remains finite for bad tokens.
    ref_ratio = (reference_logprobs.float() - policy_logprobs.float()).clamp(-20.0, 20.0).exp()
    kl = ref_ratio - (reference_logprobs.float() - policy_logprobs.float()) - 1.0
    numerator = ((-surrogate + float(kl_beta) * kl) * mask).sum()
    denominator = mask.sum().clamp_min(1.0)
    return numerator / denominator


__all__ = ["group_advantages", "grpo_loss"]
