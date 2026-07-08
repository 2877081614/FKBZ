from rein_learning.baselines import (
    GreedyExpectedBenefitPolicy,
    HighestThreatPolicy,
    NearestTargetPolicy,
    RandomLegalPolicy,
    evaluate_air_defense_policy,
    run_air_defense_episode,
)
from rein_learning.envs import (
    AirDefenseEnvConfig,
    AirDefenseResourceAssignmentEnv,
    DefenseUnitConfig,
    TargetConfig,
)


def make_policy_test_env() -> AirDefenseResourceAssignmentEnv:
    config = AirDefenseEnvConfig(
        defense_units=(
            DefenseUnitConfig(
                resource_type="missile",
                position=(0.0, 0.0),
                ammo=5,
                max_range=100.0,
                base_hit_probability=1.0,
                cost=0.0,
            ),
        ),
        targets=(
            TargetConfig(position=(40.0, 0.0), speed=0.0, threat=0.9),
            TargetConfig(position=(20.0, 0.0), speed=0.0, threat=0.3),
        ),
        max_steps=5,
    )
    env = AirDefenseResourceAssignmentEnv(config=config)
    env.reset(seed=0)
    return env


def test_nearest_target_policy_selects_closest_alive_target() -> None:
    env = make_policy_test_env()

    action = NearestTargetPolicy().select_action(env)

    assert action == 1


def test_highest_threat_policy_selects_highest_threat_target() -> None:
    env = make_policy_test_env()

    action = HighestThreatPolicy().select_action(env)

    assert action == 0


def test_greedy_expected_benefit_policy_selects_best_score() -> None:
    env = make_policy_test_env()

    action = GreedyExpectedBenefitPolicy().select_action(env)

    assert action == 0


def test_random_legal_policy_returns_action_allowed_by_mask() -> None:
    env = make_policy_test_env()
    policy = RandomLegalPolicy(seed=0)

    action = policy.select_action(env)

    assert env.action_mask()[action] == 1


def test_run_air_defense_episode_returns_metrics() -> None:
    env = make_policy_test_env()

    metrics = run_air_defense_episode(env, HighestThreatPolicy(), seed=0)

    assert metrics.num_targets == 2
    assert metrics.steps > 0
    assert metrics.total_reward != 0.0


def test_evaluate_air_defense_policy_returns_aggregate_metrics() -> None:
    metrics = evaluate_air_defense_policy(
        env_factory=make_policy_test_env,
        policy_factory=lambda seed: HighestThreatPolicy(),
        episodes=3,
        seed=0,
    )

    assert metrics["episodes"] == 3.0
    assert 0.0 <= metrics["intercept_rate"] <= 1.0
    assert 0.0 <= metrics["leak_rate"] <= 1.0
    assert metrics["avg_steps"] > 0.0
