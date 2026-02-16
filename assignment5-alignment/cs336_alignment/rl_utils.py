import torch
import torch.nn.functional as F
from typing import Literal


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Loss = - A * log_pi
    """
    # raw_rewards_or_advantages: (B, 1)
    # policy_log_probs: (B, L)

    # Broadcast A to (B, L)
    adv = raw_rewards_or_advantages  # implicit broadcasting should work, or explicit:
    # adv = raw_rewards_or_advantages.expand_as(policy_log_probs)

    loss = -1.0 * adv * policy_log_probs
    return loss


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute rewards for each group of rollout responses, normalized by the group size."""

    # 1. Compute raw rewards
    rewards_list = []
    # format_rewards = []
    # answer_rewards = []

    # We process in a loop or batch if reward_fn supports it.
    # Usually reward_fn is per example.
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(resp, gt)
        rewards_list.append(scores["reward"])
        # format_rewards.append(scores.get("format_reward", 0.0))
        # answer_rewards.append(scores.get("answer_reward", 0.0))

    raw_rewards = torch.tensor(rewards_list, dtype=torch.float32)  # (N,)

    # 2. Group normalization
    # Reshape to (num_groups, group_size)
    num_rollouts = len(raw_rewards)
    assert num_rollouts % group_size == 0
    num_groups = num_rollouts // group_size

    reshaped_rewards = raw_rewards.view(num_groups, group_size)

    # Mean and Std per group
    group_means = reshaped_rewards.mean(dim=1, keepdim=True)
    group_stds = reshaped_rewards.std(dim=1, keepdim=True)

    if normalize_by_std:
        advantages = (reshaped_rewards - group_means) / (group_stds + advantage_eps)
    else:
        advantages = reshaped_rewards - group_means

    # Flatten back
    advantages = advantages.view(-1)

    # Metadata
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
    }

    return advantages, raw_rewards, metadata


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss."""
    # policy_log_probs: (B, L)
    # old_log_probs: (B, L)
    # advantages: (B, 1)

    # Ratio = exp(log_pi - log_old)
    # Prevent numerical instability?
    # Note: policy_log_probs and old_log_probs are log probabilities.
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Broadcast advantages
    adv = advantages  # (B, 1) broadcasts to (B, L)

    # Unclipped part
    surr1 = ratio * adv

    # Clipped part
    ratio_clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    surr2 = ratio_clipped * adv

    # Policy gradient maximize objective -> minimize (-objective)
    # Objective = min(surr1, surr2)
    loss = -torch.min(surr1, surr2)

    # Metadata: clip fraction
    # Tokens where ratio was clipped?
    # Or where the clipped version was chosen?
    # Usually clip fraction is where ratio is outside bounds.
    clipped_mask = (ratio < 1.0 - cliprange) | (ratio > 1.0 + cliprange)
    clip_fraction = clipped_mask.float().mean()

    metadata = {"clip_fraction": clip_fraction}

    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Wrapper that delegates to the appropriate policy gradient loss function."""

    metadata = {}

    if loss_type == "no_baseline":
        assert raw_rewards is not None
        # Provide raw_rewards as advantages (broadcasted)
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)

    elif loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        loss, grpo_meta = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
        metadata.update(grpo_meta)

    elif (
        loss_type == "GRPO-No-Clip"
    ):  # For ablation in problem grpo_off_policy_clip_ablation
        # Unclipped version of GRPO
        # Same as naive PG but using advantages and ratio reweighting?
        # The equation (34) in PDF: - (pi/pi_old) * A
        assert advantages is not None
        assert old_log_probs is not None

        log_ratio = policy_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        adv = advantages

        loss = -(ratio * adv)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, metadata
