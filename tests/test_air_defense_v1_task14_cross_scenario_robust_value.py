from __future__ import annotations

import numpy as np
import pytest
import torch

from rein_learning.common import (
    paired_delta_reliability,
    robust_state_conditioned_value_loss,
    scenario_class_balanced_loss,
)
from rein_learning.models import StateConditionedEngagementOutput


def test_paired_reliability_rewards_confident_nonzero_cost_delta() -> None:
    samples = np.asarray(
        [
            [2.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, -1.0, 1.0],
        ]
    )
    weights = paired_delta_reliability(samples)
    assert weights[0] > weights[1]
    assert weights[0] > weights[2]
    assert float(np.mean(weights)) == pytest.approx(1.0)


def test_scenario_class_loss_is_invariant_to_duplicate_easy_rows() -> None:
    base_scores = torch.as_tensor([2.0, -2.0, 0.2, -0.2])
    labels = torch.as_tensor([1, 0, 1, 0])
    scenarios = np.asarray(["a", "a", "b", "b"])
    base, _ = scenario_class_balanced_loss(
        base_scores, labels, scenarios, worst_block_weight=0.0
    )
    duplicated, parts = scenario_class_balanced_loss(
        torch.cat((base_scores, base_scores[:2].repeat(5))),
        torch.cat((labels, labels[:2].repeat(5))),
        np.concatenate((scenarios, np.asarray(["a", "a"] * 5))),
        worst_block_weight=0.0,
    )
    assert float(duplicated) == pytest.approx(float(base), rel=1e-6)
    assert float(parts["block_count"]) == 4.0


def test_worst_block_penalty_increases_robust_loss() -> None:
    scores = torch.as_tensor([3.0, -3.0, -2.0, 2.0])
    labels = torch.as_tensor([1, 0, 1, 0])
    scenarios = np.asarray(["a", "a", "b", "b"])
    mean_only, _ = scenario_class_balanced_loss(
        scores, labels, scenarios, worst_block_weight=0.0
    )
    robust, parts = scenario_class_balanced_loss(
        scores, labels, scenarios, worst_block_weight=0.5
    )
    assert robust > mean_only
    assert parts["worst_block"] > parts["mean_block"]


def test_robust_joint_loss_uses_cost_reliability_and_backpropagates() -> None:
    safety = torch.tensor([0.2, -0.2, 0.1, -0.1], requires_grad=True)
    cost = torch.tensor([0.3, 0.3, 0.4, 0.4], requires_grad=True)
    budget = torch.ones(4, requires_grad=True)
    output = StateConditionedEngagementOutput(
        safety_gain=safety,
        cost_delta=cost,
        budget_multiplier=budget,
        score=safety - budget * torch.relu(cost),
    )
    loss, parts = robust_state_conditioned_value_loss(
        output,
        torch.as_tensor([1.0, -1.0, 1.0, -1.0]),
        torch.as_tensor([0.5, 0.5, 0.5, 0.5]),
        torch.as_tensor([1, 0, 1, 0]),
        np.asarray(["a", "a", "b", "b"]),
        cost_reliability=torch.as_tensor([2.0, 1.0, 1.0, 0.5]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert float(parts["worst_block"].detach()) > 0.0
    assert safety.grad is not None
    assert cost.grad is not None
    assert budget.grad is not None
