from __future__ import annotations

import numpy as np
import pytest
import torch

from rein_learning.common import (
    constrained_value_metrics,
    engagement_delta_targets,
    state_conditioned_value_loss,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    StateConditionedEngagementValue,
    StateConditionedEngagementValueConfig,
)


def _components() -> dict[str, np.ndarray]:
    damage = np.asarray(
        [
            [[2.0, 2.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    leaks = np.asarray(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    costs = np.asarray(
        [
            [[1.0, 1.0], [4.0, 4.0]],
            [[0.0, 0.0], [2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    return {
        "damage_samples": damage,
        "high_threat_leak_samples": leaks,
        "resource_cost_samples": costs,
    }


def test_delta_targets_preserve_zero_and_counterfactual_direction() -> None:
    targets = engagement_delta_targets(_components())
    np.testing.assert_allclose(targets["safety_gain"], [50.0, 0.0])
    np.testing.assert_allclose(targets["cost_delta"], [3.0, 2.0])
    assert targets["safety_gain_samples"].shape == (2, 2)


@pytest.mark.parametrize("mode", ["safety_only", "global_budget", "state_budget"])
def test_state_conditioned_model_has_explicit_value_outputs(mode: str) -> None:
    env = AirDefenseResourceAssignmentEnvV1()
    observation, _ = env.reset(seed=4)
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space, env.action_space
    )
    model = StateConditionedEngagementValue(
        layout, StateConditionedEngagementValueConfig(budget_mode=mode)
    )
    mask = env.action_masks().reshape(layout.num_units, layout.num_targets + 1)[0]
    output = model(
        torch.as_tensor(observation).unsqueeze(0),
        torch.as_tensor([0]),
        torch.zeros((1, layout.num_targets)),
        torch.as_tensor(mask).unsqueeze(0),
        torch.zeros(1),
    )
    assert output.safety_gain.shape == (1,)
    assert output.cost_delta.shape == (1,)
    assert output.score.shape == (1,)
    assert torch.all(output.budget_multiplier >= 0.0)
    if mode == "safety_only":
        torch.testing.assert_close(output.score, output.safety_gain)
    env.close()


def test_global_budget_penalizes_only_positive_incremental_cost() -> None:
    env = AirDefenseResourceAssignmentEnvV1()
    observation, _ = env.reset(seed=5)
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space, env.action_space
    )
    model = StateConditionedEngagementValue(
        layout,
        StateConditionedEngagementValueConfig(budget_mode="global_budget"),
    )
    with torch.no_grad():
        model.safety_head.weight.zero_()
        model.safety_head.bias.fill_(1.0)
        model.cost_head.weight.zero_()
        model.cost_head.bias.fill_(1.0)
        assert model.global_budget is not None
        model.global_budget.fill_(0.0)
    mask = env.action_masks().reshape(layout.num_units, layout.num_targets + 1)[0]
    output = model(
        torch.as_tensor(observation).unsqueeze(0),
        torch.as_tensor([0]),
        torch.zeros((1, layout.num_targets)),
        torch.as_tensor(mask).unsqueeze(0),
        torch.zeros(1),
    )
    assert float(output.safety_gain.detach()) == pytest.approx(1.0)
    assert float(output.cost_delta.detach()) == pytest.approx(1.0)
    assert float(output.score.detach()) < 1.0
    env.close()


def test_joint_value_and_classification_loss_backpropagates() -> None:
    env = AirDefenseResourceAssignmentEnvV1()
    observation, _ = env.reset(seed=6)
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space, env.action_space
    )
    model = StateConditionedEngagementValue(layout)
    observations = torch.as_tensor(np.stack((observation, observation)))
    masks = env.action_masks().reshape(layout.num_units, layout.num_targets + 1)
    output = model(
        observations,
        torch.as_tensor([0, 1]),
        torch.zeros((2, layout.num_targets)),
        torch.as_tensor(np.stack((masks[0], masks[1]))),
        torch.zeros(2),
    )
    loss, parts = state_conditioned_value_loss(
        output,
        torch.as_tensor([1.0, -1.0]),
        torch.as_tensor([0.5, 1.0]),
        torch.as_tensor([1, 0]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert float(parts["classification"].detach()) > 0.0
    assert any(parameter.grad is not None for parameter in model.parameters())
    env.close()


def test_constrained_value_metrics_report_sign_and_error() -> None:
    metrics = constrained_value_metrics(
        np.asarray([2.0, -1.0]),
        np.asarray([1.0, 3.0]),
        np.asarray([1.5, -0.5]),
        np.asarray([1.5, 2.0]),
    )
    assert metrics["safety_mae"] == pytest.approx(0.5)
    assert metrics["cost_mae"] == pytest.approx(0.75)
    assert metrics["safety_sign_accuracy"] == 1.0
