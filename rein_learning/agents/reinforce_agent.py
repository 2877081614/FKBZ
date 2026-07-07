from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..algorithms.policy_gradient import (
    discounted_returns,
    normalize_returns,
    reinforce_loss,
)
from ..models import DiscretePolicyNetwork


@dataclass
class REINFORCEConfig:
    gamma: float = 0.95
    learning_rate: float = 0.02
    hidden_sizes: tuple[int, ...] = (32,)
    normalize_returns: bool = True
    device: str = "auto"


@dataclass(frozen=True)
class REINFORCETrajectoryStep:
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


class REINFORCEAgent:
    """Vanilla Monte Carlo policy-gradient agent for discrete-action tasks."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        config: REINFORCEConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.config = config or REINFORCEConfig()
        torch.manual_seed(seed or 0)

        self.device = self._resolve_device(self.config.device)
        self.policy = DiscretePolicyNetwork(
            num_states=num_states,
            num_actions=num_actions,
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.policy.parameters(),
            lr=self.config.learning_rate,
        )
        self.update_steps = 0

    def select_action(self, state: int, *, greedy: bool = False) -> int:
        states = torch.tensor([state], dtype=torch.long, device=self.device)
        with torch.no_grad():
            distribution = self.policy.action_distribution(states)
            if greedy:
                return int(torch.argmax(distribution.probs, dim=-1).item())
            return int(distribution.sample().item())

    def action_probabilities(self, state: int) -> list[float]:
        states = torch.tensor([state], dtype=torch.long, device=self.device)
        with torch.no_grad():
            probabilities = self.policy.action_probabilities(states).squeeze(0)
        return [float(value) for value in probabilities.cpu().tolist()]

    def update_episode(
        self,
        trajectory: Sequence[REINFORCETrajectoryStep],
    ) -> float:
        if not trajectory:
            raise ValueError("trajectory must contain at least one step")

        states = torch.tensor(
            [step.state for step in trajectory],
            dtype=torch.long,
            device=self.device,
        )
        actions = torch.tensor(
            [step.action for step in trajectory],
            dtype=torch.long,
            device=self.device,
        )
        rewards = [step.reward for step in trajectory]
        returns = discounted_returns(
            rewards,
            self.config.gamma,
            device=self.device,
        )
        learning_signals = (
            normalize_returns(returns)
            if self.config.normalize_returns
            else returns
        )

        distribution = self.policy.action_distribution(states)
        log_probs = distribution.log_prob(actions)
        loss = reinforce_loss(log_probs, learning_signals)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1

        return float(loss.detach().cpu().item())

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
