from rein_learning.baselines import (
    GreedyDamageReductionPolicy,
    HighestThreatJointPolicy,
    NearestTargetJointPolicy,
    RandomLegalJointPolicy,
    TimeToImpactJointPolicy,
    evaluate_air_defense_v1_policy,
    run_air_defense_v1_episode,
)
from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    AirDefenseV1EnvConfig,
    DefenseUnitV1Config,
    ProtectedZoneConfig,
    TargetV1Config,
)


def make_v1_policy_test_env() -> AirDefenseResourceAssignmentEnvV1:
    config = AirDefenseV1EnvConfig(
        protected_zones=(
            ProtectedZoneConfig(position=(0.0, 0.0), radius=2.0, value=1.0),
            ProtectedZoneConfig(position=(30.0, 0.0), radius=2.0, value=0.5),
        ),
        defense_units=(
            DefenseUnitV1Config(
                resource_type="missile",
                position=(0.0, 0.0),
                ammo=5,
                max_range=100.0,
                base_hit_probability=1.0,
                cost=0.0,
            ),
            DefenseUnitV1Config(
                resource_type="laser",
                position=(30.0, 0.0),
                ammo=5,
                max_range=100.0,
                base_hit_probability=1.0,
                cost=0.0,
            ),
        ),
        targets=(
            TargetV1Config(
                position=(40.0, 0.0),
                speed=0.0,
                threat=0.9,
                target_zone=0,
                payload=1.0,
            ),
            TargetV1Config(
                position=(20.0, 0.0),
                speed=0.0,
                threat=0.3,
                target_zone=1,
                payload=0.6,
            ),
        ),
        max_steps=5,
    )
    env = AirDefenseResourceAssignmentEnvV1(config=config)
    env.reset(seed=0)
    return env


def test_v1_random_legal_policy_returns_actions_allowed_by_mask() -> None:
    env = make_v1_policy_test_env()
    policy = RandomLegalJointPolicy(seed=0)

    action = policy.select_action(env)
    mask = env.action_mask()

    assert all(mask[unit_index, unit_action] == 1 for unit_index, unit_action in enumerate(action))


def test_v1_nearest_policy_returns_joint_action() -> None:
    env = make_v1_policy_test_env()

    action = NearestTargetJointPolicy().select_action(env)

    assert action.shape == (env.num_defense_units,)
    assert action[0] in {0, 1, env.noop_action}


def test_v1_highest_threat_policy_prioritizes_damage_weighted_target() -> None:
    env = make_v1_policy_test_env()

    action = HighestThreatJointPolicy().select_action(env)

    assert 0 in set(action.tolist())


def test_v1_time_to_impact_policy_returns_legal_joint_action() -> None:
    env = make_v1_policy_test_env()

    action = TimeToImpactJointPolicy().select_action(env)
    mask = env.action_mask()

    assert all(mask[unit_index, unit_action] == 1 for unit_index, unit_action in enumerate(action))


def test_v1_greedy_damage_policy_returns_legal_joint_action() -> None:
    env = make_v1_policy_test_env()

    action = GreedyDamageReductionPolicy().select_action(env)
    mask = env.action_mask()

    assert all(mask[unit_index, unit_action] == 1 for unit_index, unit_action in enumerate(action))


def test_v1_run_episode_returns_metrics() -> None:
    env = make_v1_policy_test_env()

    metrics = run_air_defense_v1_episode(env, HighestThreatJointPolicy(), seed=0)

    assert metrics.num_targets == 2
    assert metrics.steps > 0
    assert metrics.total_reward != 0.0
    assert metrics.decision_time_seconds >= 0.0
    assert 0.0 <= metrics.high_threat_leak_rate <= 1.0
    assert 0.0 <= metrics.assignment_conflict_rate <= 1.0
    assert 0.0 <= metrics.overkill_rate <= 1.0


def test_v1_evaluate_policy_returns_aggregate_metrics() -> None:
    metrics = evaluate_air_defense_v1_policy(
        env_factory=make_v1_policy_test_env,
        policy_factory=lambda seed: HighestThreatJointPolicy(),
        episodes=3,
        seed=0,
    )

    assert metrics["episodes"] == 3.0
    assert 0.0 <= metrics["intercept_rate"] <= 1.0
    assert 0.0 <= metrics["leak_rate"] <= 1.0
    assert metrics["avg_steps"] > 0.0
    assert metrics["avg_decision_time_ms"] >= 0.0
    assert metrics["avg_zone_weighted_damage"] >= 0.0
    assert metrics["avg_resource_cost"] >= 0.0
