import numpy as np
import pytest

from rein_learning.baselines import (
    evaluate_air_defense_v1_policy,
    run_air_defense_v1_episode,
)
from rein_learning.common import aggregate_air_defense_v1_episode_metrics
from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    AirDefenseV1EnvConfig,
    DefenseUnitV1Config,
    ProtectedZoneConfig,
    TargetV1Config,
)


class FixedJointPolicy:
    def __init__(self, action: list[int]) -> None:
        self.action = np.asarray(action, dtype=np.int64)

    def select_action(self, env: AirDefenseResourceAssignmentEnvV1) -> np.ndarray:
        return self.action.copy()


def make_conflict_leak_env() -> AirDefenseResourceAssignmentEnvV1:
    config = AirDefenseV1EnvConfig(
        protected_zones=(
            ProtectedZoneConfig(position=(0.0, 0.0), radius=5.0, value=2.0),
        ),
        defense_units=(
            DefenseUnitV1Config(
                resource_type="missile",
                position=(0.0, 0.0),
                ammo=1,
                max_range=100.0,
                base_hit_probability=0.0,
                cost=1.0,
            ),
            DefenseUnitV1Config(
                resource_type="missile",
                position=(0.0, 0.0),
                ammo=1,
                max_range=100.0,
                base_hit_probability=0.0,
                cost=2.0,
            ),
            DefenseUnitV1Config(
                resource_type="laser",
                position=(0.0, 0.0),
                ammo=1,
                max_range=100.0,
                base_hit_probability=0.0,
                cost=3.0,
            ),
        ),
        targets=(
            TargetV1Config(
                position=(0.0, 0.0),
                speed=0.0,
                threat=0.9,
                target_zone=0,
                payload=2.0,
            ),
            TargetV1Config(
                position=(0.0, 0.0),
                speed=0.0,
                threat=0.5,
                target_zone=0,
                payload=1.0,
            ),
        ),
        max_steps=1,
        max_allowed_damage=10.0,
    )
    return AirDefenseResourceAssignmentEnvV1(config=config)


def make_intercept_env() -> AirDefenseResourceAssignmentEnvV1:
    config = AirDefenseV1EnvConfig(
        protected_zones=(
            ProtectedZoneConfig(position=(0.0, 0.0), radius=5.0, value=2.0),
        ),
        defense_units=(
            DefenseUnitV1Config(
                resource_type="missile",
                position=(0.0, 0.0),
                ammo=1,
                max_range=100.0,
                base_hit_probability=1.0,
                cost=2.0,
            ),
        ),
        targets=(
            TargetV1Config(
                position=(0.0, 0.0),
                speed=0.0,
                threat=0.9,
                target_zone=0,
                payload=2.0,
            ),
        ),
        max_steps=1,
        max_allowed_damage=10.0,
    )
    return AirDefenseResourceAssignmentEnvV1(config=config)


def test_fixed_conflict_scenario_has_hand_verifiable_diagnostics() -> None:
    metrics = run_air_defense_v1_episode(
        make_conflict_leak_env(),
        FixedJointPolicy([0, 0, 1]),
        seed=0,
    )

    assert metrics.num_high_threat_targets == 1
    assert metrics.num_high_threat_leaked == 1
    assert metrics.high_threat_leak_rate == pytest.approx(1.0)
    assert metrics.zone_weighted_damage == pytest.approx(4.6)
    assert metrics.engaged_target_events == 2
    assert metrics.conflict_target_events == 1
    assert metrics.assignment_conflict_rate == pytest.approx(0.5)
    assert metrics.overkill_assignments == 1
    assert metrics.overkill_rate == pytest.approx(1.0 / 3.0)
    assert metrics.damage_reduction_per_ammo == pytest.approx(0.0)
    assert metrics.resource_cost == pytest.approx(6.0)


def test_intercepted_damage_potential_defines_resource_efficiency() -> None:
    metrics = run_air_defense_v1_episode(
        make_intercept_env(),
        FixedJointPolicy([0]),
        seed=0,
    )

    assert metrics.intercepted_damage_potential == pytest.approx(3.6)
    assert metrics.damage_reduction_per_ammo == pytest.approx(3.6)
    assert metrics.zone_weighted_damage == pytest.approx(0.0)
    assert metrics.high_threat_leak_rate == pytest.approx(0.0)
    assert metrics.resource_cost == pytest.approx(2.0)


def test_raw_episode_rows_reaggregate_to_reported_metrics() -> None:
    raw_episode_rows: list[dict[str, float | int | bool]] = []
    metrics = evaluate_air_defense_v1_policy(
        env_factory=make_conflict_leak_env,
        policy_factory=lambda seed: FixedJointPolicy([0, 0, 1]),
        episodes=2,
        seed=10,
        episode_metrics_callback=raw_episode_rows.append,
    )

    reaggregated = aggregate_air_defense_v1_episode_metrics(raw_episode_rows)

    assert len(raw_episode_rows) == 2
    assert metrics == pytest.approx(reaggregated)
    for ratio_name in (
        "success_rate",
        "intercept_rate",
        "leak_rate",
        "hit_rate_per_shot",
        "high_threat_leak_rate",
        "assignment_conflict_rate",
        "overkill_rate",
    ):
        assert 0.0 <= metrics[ratio_name] <= 1.0


def test_pre_diagnostic_episode_rows_keep_old_metrics_compatible() -> None:
    legacy_row = {
        "total_reward": -3.0,
        "steps": 2,
        "num_targets": 2,
        "num_intercepted": 1,
        "num_leaked": 1,
        "total_damage": 1.5,
        "ammo_used": 2,
        "shots": 2,
        "hits": 1,
        "invalid_actions": 0,
        "success": False,
        "decision_time_seconds": 0.002,
    }

    metrics = aggregate_air_defense_v1_episode_metrics([legacy_row])

    assert metrics["avg_reward"] == pytest.approx(-3.0)
    assert metrics["intercept_rate"] == pytest.approx(0.5)
    assert metrics["leak_rate"] == pytest.approx(0.5)
    assert metrics["avg_total_damage"] == pytest.approx(1.5)
    assert metrics["avg_zone_weighted_damage"] == pytest.approx(1.5)
    assert metrics["high_threat_leak_rate"] == pytest.approx(0.0)
    assert metrics["assignment_conflict_rate"] == pytest.approx(0.0)
    assert metrics["overkill_rate"] == pytest.approx(0.0)
