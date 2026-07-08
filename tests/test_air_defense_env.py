import numpy as np

from rein_learning.envs import (
    AirDefenseEnvConfig,
    AirDefenseResourceAssignmentEnv,
    DefenseUnitConfig,
    TargetConfig,
)


def make_single_target_config(
    *,
    unit: DefenseUnitConfig | None = None,
    target: TargetConfig | None = None,
    asset_radius: float = 1.0,
    max_allowed_leaks: int | None = None,
) -> AirDefenseEnvConfig:
    return AirDefenseEnvConfig(
        defense_units=(
            unit
            or DefenseUnitConfig(
                resource_type="missile",
                position=(20.0, 0.0),
                ammo=1,
                max_range=100.0,
                base_hit_probability=1.0,
                cost=1.0,
                cooldown_after_fire=1,
            ),
        ),
        targets=(
            target
            or TargetConfig(
                position=(20.0, 0.0),
                speed=0.0,
                threat=1.0,
            ),
        ),
        asset_radius=asset_radius,
        max_allowed_leaks=max_allowed_leaks,
        max_steps=10,
    )


def test_reset_returns_fixed_shape_observation_and_info() -> None:
    env = AirDefenseResourceAssignmentEnv()

    obs, info = env.reset(seed=0)

    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert env.action_space.n == env.num_defense_units * env.num_targets + 1
    assert info["num_alive"] == env.num_targets
    assert info["num_intercepted"] == 0
    assert info["num_leaked"] == 0


def test_reset_is_reproducible_with_same_seed() -> None:
    env = AirDefenseResourceAssignmentEnv()

    first_obs, _ = env.reset(seed=123)
    second_obs, _ = env.reset(seed=123)

    assert np.allclose(first_obs, second_obs)


def test_successful_intercept_consumes_ammo_and_terminates() -> None:
    env = AirDefenseResourceAssignmentEnv(config=make_single_target_config())
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step(0)

    assert env.observation_space.contains(obs)
    assert reward > 0.0
    assert terminated
    assert not truncated
    assert info["hit"] is True
    assert info["hit_probability"] == 1.0
    assert info["num_intercepted"] == 1
    assert info["ammo_remaining"] == 0
    assert info["reward_breakdown"]["intercept"] == 10.0


def test_invalid_action_is_penalized() -> None:
    unit = DefenseUnitConfig(
        resource_type="missile",
        position=(20.0, 0.0),
        ammo=0,
        max_range=100.0,
        base_hit_probability=1.0,
        cost=1.0,
    )
    target = TargetConfig(position=(20.0, 0.0), speed=0.0, threat=1.0)
    env = AirDefenseResourceAssignmentEnv(
        config=make_single_target_config(unit=unit, target=target)
    )
    env.reset(seed=0)

    _, reward, terminated, truncated, info = env.step(0)

    assert reward == -5.1
    assert not terminated
    assert not truncated
    assert info["invalid_action"] is True
    assert info["reward_breakdown"]["invalid"] == -5.0


def test_target_leak_penalty_and_failure_terminal() -> None:
    target = TargetConfig(position=(6.0, 0.0), speed=2.0, threat=1.0)
    env = AirDefenseResourceAssignmentEnv(
        config=make_single_target_config(
            target=target,
            asset_radius=5.0,
            max_allowed_leaks=1,
        )
    )
    env.reset(seed=0)

    _, reward, terminated, truncated, info = env.step(env.noop_action)

    assert reward == -40.1
    assert terminated
    assert not truncated
    assert info["num_leaked"] == 1
    assert info["reward_breakdown"]["leak"] == -20.0
    assert info["reward_breakdown"]["terminal"] == -20.0


def test_action_mask_marks_noop_and_legal_actions() -> None:
    env = AirDefenseResourceAssignmentEnv(config=make_single_target_config())
    env.reset(seed=0)

    mask = env.action_mask()

    assert mask.shape == (env.action_space.n,)
    assert mask[0] == 1
    assert mask[env.noop_action] == 1
