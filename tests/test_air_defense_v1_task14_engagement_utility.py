from __future__ import annotations

import numpy as np
import pytest
import torch

from rein_learning.common import (
    EngagementUtilityConfig,
    engagement_utility_labels,
    lower_tail_cvar,
    oracle_classification_metrics,
    safety_resource_oracle,
    utility_oracle_metrics,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    RiskAwareEngagementCritic,
)


def _components(groups: int = 3, rollouts: int = 4) -> dict[str, np.ndarray]:
    shape = (groups, 2, rollouts)
    return {
        "operational_return_samples": np.zeros(shape, dtype=np.float32),
        "resource_cost_samples": np.zeros(shape, dtype=np.float32),
        "damage_samples": np.zeros(shape, dtype=np.float32),
        "high_threat_leak_samples": np.zeros(shape, dtype=np.float32),
    }


def test_lower_tail_cvar_uses_worst_returns() -> None:
    samples = np.asarray([[1.0, 2.0, 9.0, 10.0]])
    np.testing.assert_allclose(lower_tail_cvar(samples, 0.5), [1.5])


def test_risk_weight_penalizes_lower_tail() -> None:
    components = _components(groups=1)
    components["operational_return_samples"][0, 0] = [2.0, 2.0, 2.0, 2.0]
    components["operational_return_samples"][0, 1] = [-2.0, 4.0, 4.0, 4.0]
    mean_labels, _ = engagement_utility_labels(
        components, EngagementUtilityConfig(cvar_weight=0.0)
    )
    risk_labels, _ = engagement_utility_labels(
        components, EngagementUtilityConfig(cvar_weight=1.0, cvar_alpha=0.25)
    )
    assert mean_labels[0, 1] > mean_labels[0, 0]
    assert risk_labels[0, 1] < risk_labels[0, 0]


def test_safety_resource_oracle_separates_both_failure_tails() -> None:
    components = _components()
    components["damage_samples"][0, 0] = 1.0
    components["damage_samples"][0, 1] = 0.0
    components["resource_cost_samples"][0, 1] = 2.0

    components["damage_samples"][1] = 0.0
    components["resource_cost_samples"][1, 1] = 2.0

    components["damage_samples"][2] = 0.0
    components["resource_cost_samples"][2] = 0.0

    oracle = safety_resource_oracle(components)
    np.testing.assert_array_equal(oracle["labels"], [1, 0, -1])


def test_oracle_metrics_report_false_noop_and_wasteful_engage() -> None:
    oracle = np.asarray([1, 1, 0, 0, -1])
    predicted = np.asarray([1, 0, 0, 1, 1])
    metrics = oracle_classification_metrics(oracle, predicted)
    assert metrics["count"] == 4
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["false_noop_rate"] == pytest.approx(0.5)
    assert metrics["wasteful_engage_rate"] == pytest.approx(0.5)


def test_utility_oracle_metrics_use_engage_minus_noop_sign() -> None:
    utility = np.asarray([[0.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    metrics = utility_oracle_metrics(utility, np.asarray([1, 0]))
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)


def test_risk_aware_engagement_critic_outputs_binary_values() -> None:
    env = AirDefenseResourceAssignmentEnvV1()
    observation, _ = env.reset(seed=3)
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space, env.action_space
    )
    model = RiskAwareEngagementCritic(layout)
    batch = 2
    masks = env.action_masks().reshape(layout.num_units, layout.num_targets + 1)
    predictions = model(
        torch.as_tensor(np.stack((observation, observation))),
        torch.as_tensor([0, 1]),
        torch.zeros((batch, layout.num_targets)),
        torch.as_tensor(np.stack((masks[0], masks[1]))),
    )
    env.close()
    assert predictions.shape == (batch, 2)
    assert model.parameter_count() > 0


def test_utility_config_rejects_invalid_risk_parameters() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        EngagementUtilityConfig(cost_weight=-1.0)
    with pytest.raises(ValueError, match="cvar_alpha"):
        EngagementUtilityConfig(cvar_alpha=0.0)
