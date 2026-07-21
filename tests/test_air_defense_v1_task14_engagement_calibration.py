from __future__ import annotations

import numpy as np
import pytest
import torch

from rein_learning.common import (
    EngagementBoundaryConfig,
    EngagementBoundaryConstraints,
    apply_engagement_boundary,
    calibrate_engagement_boundary,
    resource_pressure_from_observations,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import AirDefenseV1ObservationLayout


def test_resource_pressure_uses_selected_unit_cost_and_ammo() -> None:
    env = AirDefenseResourceAssignmentEnvV1()
    observation, _ = env.reset(seed=3)
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space, env.action_space
    )
    batch = np.stack((observation, observation))
    indices = np.asarray([0, 1])
    pressure = resource_pressure_from_observations(
        batch, indices, **layout.signature()
    )
    units = layout.split(torch.as_tensor(batch)).units.numpy()
    expected = units[np.arange(2), indices, 10] * (
        2.0 - units[np.arange(2), indices, 3]
    )
    np.testing.assert_allclose(pressure, expected)
    env.close()


def test_resource_pressure_rises_with_cost_and_ammo_scarcity() -> None:
    observations = np.zeros((2, 30), dtype=np.float32)
    unit_start = 2
    observations[0, unit_start + 3] = 1.0
    observations[0, unit_start + 10] = 0.4
    observations[1, unit_start + 3] = 0.25
    observations[1, unit_start + 10] = 0.8
    pressure = resource_pressure_from_observations(
        observations,
        np.asarray([0, 0]),
        num_zones=1,
        num_targets=1,
        num_units=1,
        zone_feature_dim=1,
        target_feature_dim=1,
        unit_feature_dim=15,
        global_feature_dim=13,
    )
    assert pressure[1] > pressure[0]


def test_apply_boundary_adds_resource_dependent_stopping_cost() -> None:
    config = EngagementBoundaryConfig(
        threshold=0.2, dual_weight=1.0, logit_scale=0.5, feasible=True
    )
    predicted = apply_engagement_boundary(
        np.asarray([0.4, 0.4]), config, np.asarray([0.0, 1.0])
    )
    np.testing.assert_array_equal(predicted, np.asarray([1, 0]))


def test_calibration_selects_resource_boundary_when_global_is_infeasible() -> None:
    logits = np.asarray([0.9, 0.6, 0.8, 0.5])
    labels = np.asarray([1, 1, 0, 0])
    scenarios = np.asarray(["a", "b", "a", "b"])
    pressure = np.asarray([0.0, 0.0, 2.0, 2.0])
    config, rows = calibrate_engagement_boundary(
        logits,
        labels,
        scenarios,
        pressure,
        dual_weights=(0.0, 1.0, 2.0),
        constraints=EngagementBoundaryConstraints(
            balanced_accuracy=0.9,
            engage_recall=0.9,
            noop_recall=0.9,
            scenario_engage_recall=0.9,
            scenario_noop_recall=0.9,
        ),
    )
    assert config.feasible
    assert config.family == "resource_dual"
    assert any(bool(row["feasible"]) for row in rows)
    np.testing.assert_array_equal(
        apply_engagement_boundary(logits, config, pressure), labels
    )


def test_calibration_requires_both_oracle_classes() -> None:
    with pytest.raises(ValueError, match="both reliable oracle classes"):
        calibrate_engagement_boundary(
            np.asarray([0.1, 0.2]),
            np.asarray([1, 1]),
            np.asarray(["a", "a"]),
            np.asarray([0.0, 0.0]),
            dual_weights=(0.0,),
        )
