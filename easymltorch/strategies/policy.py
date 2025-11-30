"""Policy gradient strategy for reward-based learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from easymltorch.model import Model


def compute_loss(model: Model, reward: float) -> torch.Tensor:
    """
    Compute the policy gradient loss for the given reward.

    This implements the REINFORCE algorithm loss:
        loss = -(reward * log_prob_of_predicted_action)

    The negative sign is used because we want to maximize reward,
    but optimizers minimize loss.

    Args:
        model: The Model instance containing the last prediction state.
        reward: The reward signal (positive = good, negative = bad).

    Returns:
        The computed loss tensor.

    Raises:
        RuntimeError: If there is no stored prediction state.
    """
    if model._last_logits is None:
        raise RuntimeError("reward() called without a prior predict() step")

    # Compute log probabilities using log_softmax
    log_probs = F.log_softmax(model._last_logits, dim=-1)

    # Get the log probability of the action that was taken
    # model._last_action stores the index of the predicted action
    log_prob_action = log_probs[model._last_action]

    # Policy gradient loss: negative because we want to maximize reward
    # When reward is positive, we want to increase log_prob (decrease loss)
    # When reward is negative, we want to decrease log_prob (increase loss)
    loss = -(reward * log_prob_action)

    return loss
