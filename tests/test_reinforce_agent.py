import math

import pytest
import torch

from rein_learning.agents import (
    REINFORCEAgent,
    REINFORCEConfig,
    REINFORCETrajectoryStep,
)
from rein_learning.algorithms.policy_gradient import discounted_returns
from rein_learning.models import DiscretePolicyNetwork


def test_discounted_returns_are_computed_backward() -> None:
    returns = discounted_returns([1.0, 2.0, 3.0], gamma=0.5)

    assert returns.tolist() == pytest.approx([2.75, 3.5, 3.0])


def test_discrete_policy_network_outputs_action_logits() -> None:
    network = DiscretePolicyNetwork(num_states=5, num_actions=2, hidden_sizes=(8,))

    logits = network(torch.tensor([0, 1, 2], dtype=torch.long))

    assert logits.shape == (3, 2)


def test_reinforce_agent_updates_policy_and_returns_loss() -> None:
    agent = REINFORCEAgent(
        num_states=5,
        num_actions=2,
        config=REINFORCEConfig(
            gamma=0.9,
            learning_rate=0.05,
            hidden_sizes=(8,),
            normalize_returns=False,
            device="cpu",
        ),
        seed=0,
    )
    trajectory = [
        REINFORCETrajectoryStep(2, 1, 0.2, 3, False),
        REINFORCETrajectoryStep(3, 1, 1.0, 4, True),
    ]
    before = {
        name: parameter.detach().clone()
        for name, parameter in agent.policy.named_parameters()
    }

    loss = agent.update_episode(trajectory)

    assert loss >= 0.0
    assert math.isfinite(loss)
    assert agent.update_steps == 1
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in agent.policy.named_parameters()
    )
