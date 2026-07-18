import numpy as np
import pytest

from rein_learning.common import (
    LEAK_ATTRIBUTION_CATEGORIES,
    AirDefenseV1DecisionTracker,
    aggregate_decision_rows,
    classify_high_threat_leak,
)
from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    AirDefenseV1EnvConfig,
    DefenseUnitV1Config,
    ProtectedZoneConfig,
    TargetV1Config,
)


def make_trace_env() -> AirDefenseResourceAssignmentEnvV1:
    config = AirDefenseV1EnvConfig(
        protected_zones=(
            ProtectedZoneConfig(position=(0.0, 0.0), radius=1.0, value=1.0),
        ),
        defense_units=(
            DefenseUnitV1Config(
                resource_type="missile",
                position=(10.0, 0.0),
                ammo=2,
                max_range=100.0,
                base_hit_probability=0.8,
                cost=1.0,
            ),
            DefenseUnitV1Config(
                resource_type="laser",
                position=(12.0, 0.0),
                ammo=2,
                max_range=100.0,
                base_hit_probability=0.6,
                cost=0.2,
            ),
        ),
        targets=(
            TargetV1Config(
                position=(20.0, 0.0),
                speed=0.0,
                threat=0.9,
                target_zone=0,
                payload=1.0,
            ),
            TargetV1Config(
                position=(25.0, 0.0),
                speed=0.0,
                threat=0.6,
                target_zone=0,
                payload=0.8,
            ),
        ),
        max_steps=2,
    )
    return AirDefenseResourceAssignmentEnvV1(config=config)


def test_decision_tracker_records_environment_indices_and_order_positions() -> None:
    env = make_trace_env()
    env.reset(seed=0)
    tracker = AirDefenseV1DecisionTracker(
        unit_order=(1, 0),
        num_units=2,
        num_targets=2,
        high_threat_threshold=0.8,
    )
    action = np.asarray([1, 0], dtype=np.int64)

    rows = tracker.before_step(env, action)
    _, _, _, _, info = env.step(action)
    tracker.after_step(env, info, rows)

    assert len(tracker.rows) == 2
    assert [row["unit_index"] for row in tracker.rows] == [1, 0]
    assert [row["unit_order_position"] for row in tracker.rows] == [0, 1]
    assert tracker.rows[0]["selected_target"] == 0
    assert tracker.rows[1]["selected_target"] == 1
    assert tracker.rows[1]["prefix_denied_target_count"] == 1
    assert all(row["conditional_action_legal"] for row in tracker.rows)
    assert all(row["shot_fired"] for row in tracker.rows)
    env.close()


def test_noop_uses_none_for_target_fields_and_is_marked_avoidable() -> None:
    env = make_trace_env()
    env.reset(seed=1)
    tracker = AirDefenseV1DecisionTracker(
        unit_order=(0, 1),
        num_units=2,
        num_targets=2,
    )

    rows = tracker.before_step(env, [env.noop_action, env.noop_action])

    assert rows[0]["selected_target"] is None
    assert rows[0]["target_threat"] is None
    assert rows[0]["expected_damage_reduction"] is None
    assert rows[0]["avoidable_noop"] is True
    env.close()


@pytest.mark.parametrize(
    ("expected", "kwargs"),
    (
        ("never_legal", {}),
        ("unassigned", {"ever_legal": True}),
        ("resource_exhausted", {
            "ever_geometrically_reachable": True,
            "ever_blocked_by_unavailability": True,
        }),
        ("attempted_miss", {"ever_assigned": True}),
        ("mismatched_resource", {
            "ever_assigned": True,
            "mismatched_resource": True,
        }),
        ("prefix_denied", {
            "ever_assigned": True,
            "prefix_denied_better_resource": True,
            "mismatched_resource": True,
        }),
    ),
)
def test_high_threat_leak_attribution_is_mutually_exclusive(expected, kwargs) -> None:
    inputs = {
        "ever_legal": False,
        "ever_assigned": False,
        "prefix_denied_better_resource": False,
        "mismatched_resource": False,
        "ever_geometrically_reachable": False,
        "ever_blocked_by_unavailability": False,
    }
    inputs.update(kwargs)

    assert classify_high_threat_leak(**inputs) == expected
    assert expected in LEAK_ATTRIBUTION_CATEGORIES


def test_decision_aggregation_preserves_opportunity_denominators() -> None:
    rows = [
        {
            "method": "test",
            "unit_index": 0,
            "selected_noop": False,
            "avoidable_noop": False,
            "selected_high_threat": True,
            "num_conditional_legal_targets": 2,
            "num_conditional_high_threat_targets": 1,
            "num_base_legal_targets": 2,
            "prefix_denied_target_count": 0,
            "target_threat": 0.9,
            "expected_damage_reduction": 0.7,
            "matching_efficiency": 1.0,
        },
        {
            "method": "test",
            "unit_index": 0,
            "selected_noop": True,
            "avoidable_noop": True,
            "selected_high_threat": False,
            "num_conditional_legal_targets": 1,
            "num_conditional_high_threat_targets": 0,
            "num_base_legal_targets": 2,
            "prefix_denied_target_count": 1,
            "target_threat": None,
            "expected_damage_reduction": None,
            "matching_efficiency": None,
        },
    ]

    summary = aggregate_decision_rows(
        rows,
        group_keys=("method", "unit_index"),
    )[0]

    assert summary["decision_opportunities"] == 2
    assert summary["assignment_rate"] == pytest.approx(0.5)
    assert summary["avoidable_noop_rate"] == pytest.approx(0.5)
    assert summary["high_threat_assignment_rate"] == pytest.approx(1.0)
    assert summary["prefix_denial_rate"] == pytest.approx(0.25)
