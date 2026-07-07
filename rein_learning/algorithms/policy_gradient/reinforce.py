from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class REINFORCEBatch:
    """A single on-policy episode prepared for a REINFORCE update."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    returns: torch.Tensor


def discounted_returns(
    rewards: Sequence[float] | torch.Tensor,
    gamma: float,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute G_t = r_t + gamma r_{t+1} + ... for one episode."""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    running_return = 0.0
    returns: list[float] = []
    reward_values = (
        rewards.detach().cpu().tolist()
        if isinstance(rewards, torch.Tensor)
        else list(rewards)
    )

    for reward in reversed(reward_values):
        running_return = float(reward) + gamma * running_return
        returns.append(running_return)

    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32, device=device)


def normalize_returns(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize episode returns to reduce REINFORCE gradient variance."""

    if returns.ndim != 1:
        raise ValueError("returns must be a 1D tensor")
    if returns.numel() <= 1:
        return returns - returns.mean()

    return (returns - returns.mean()) / (returns.std(unbiased=False) + eps)


def reinforce_objective(log_probs: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Monte Carlo policy-gradient objective for gradient ascent."""

    if log_probs.shape != returns.shape:
        raise ValueError("log_probs and returns must have the same shape")
    return torch.mean(log_probs * returns)


def reinforce_loss(log_probs: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Negative REINFORCE objective for PyTorch optimizers that minimize losses."""

    return -reinforce_objective(log_probs, returns)
