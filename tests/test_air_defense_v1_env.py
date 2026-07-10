import numpy as np

from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    AirDefenseV1EnvConfig,
    DefenseUnitV1Config,
    ProtectedZoneConfig,
    TargetV1Config,
)


def make_single_target_v1_config(
    *,
    unit: DefenseUnitV1Config | None = None,
    target: TargetV1Config | None = None,
    zone: ProtectedZoneConfig | None = None,
    max_allowed_damage: float = 2.0,
) -> AirDefenseV1EnvConfig:
    return AirDefenseV1EnvConfig(
        protected_zones=(
            zone
            or ProtectedZoneConfig(
                position=(0.0, 0.0),
                radius=2.0,
                value=1.0,
            ),
        ),
        defense_units=(
            unit
            or DefenseUnitV1Config(
                resource_type="missile",
                position=(10.0, 0.0),
                ammo=1,
                max_range=100.0,
                base_hit_probability=1.0,
                cost=1.0,
            ),
        ),
        targets=(
            target
            or TargetV1Config(
                position=(10.0, 0.0),
                speed=0.0,
                threat=1.0,
                target_zone=0,
                payload=1.0,
            ),
        ),
        max_steps=10,
        max_allowed_damage=max_allowed_damage,
    )


def test_v1_reset_returns_fixed_shape_observation_and_info() -> None:
    env = AirDefenseResourceAssignmentEnvV1()

    obs, info = env.reset(seed=0)

    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert env.action_space.nvec.tolist() == [env.num_targets + 1] * env.num_defense_units
    assert info["num_alive"] == env.num_targets
    assert info["total_damage"] == 0.0


def test_v1_reset_is_reproducible_with_same_seed() -> None:
    env = AirDefenseResourceAssignmentEnvV1()

    first_obs, _ = env.reset(seed=123)
    second_obs, _ = env.reset(seed=123)

    assert np.allclose(first_obs, second_obs)


def test_v1_action_mask_has_one_row_per_defense_unit() -> None:
    env = AirDefenseResourceAssignmentEnvV1(
        config=make_single_target_v1_config()
    )
    env.reset(seed=0)

    mask = env.action_mask()

    assert mask.shape == (env.num_defense_units, env.num_targets + 1)
    assert mask[0, 0] == 1
    assert mask[0, env.noop_action] == 1
    assert env.action_masks().shape == (env.num_defense_units * (env.num_targets + 1),)


def test_v1_successful_intercept_uses_joint_action_and_terminates() -> None:
    env = AirDefenseResourceAssignmentEnvV1(
        config=make_single_target_v1_config()
    )
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step(np.asarray([0]))

    assert env.observation_space.contains(obs)
    assert reward > 0.0
    assert terminated
    assert not truncated
    assert info["num_intercepted"] == 1
    assert info["num_leaked"] == 0
    assert info["shots"] == 1
    assert info["hits"] == 1
    assert info["ammo_remaining"] == 0
    assert info["reward_breakdown"]["intercept"] == 8.0


def test_v1_invalid_joint_action_is_penalized() -> None:
    unit = DefenseUnitV1Config(
        resource_type="missile",
        position=(10.0, 0.0),
        ammo=0,
        max_range=100.0,
        base_hit_probability=1.0,
        cost=1.0,
    )
    env = AirDefenseResourceAssignmentEnvV1(
        config=make_single_target_v1_config(unit=unit)
    )
    env.reset(seed=0)

    _, reward, terminated, truncated, info = env.step(np.asarray([0]))

    assert reward == -5.1
    assert not terminated
    assert not truncated
    assert info["invalid_actions"] == 1
    assert info["reward_breakdown"]["invalid"] == -5.0


def test_v1_target_leak_creates_zone_damage_and_failure() -> None:
    target = TargetV1Config(
        position=(3.0, 0.0),
        speed=2.0,
        threat=1.0,
        target_zone=0,
        payload=1.0,
    )
    env = AirDefenseResourceAssignmentEnvV1(
        config=make_single_target_v1_config(
            target=target,
            max_allowed_damage=0.5,
        )
    )
    env.reset(seed=0)

    _, reward, terminated, truncated, info = env.step(np.asarray([env.noop_action]))

    assert terminated
    assert not truncated
    assert info["num_leaked"] == 1
    assert info["total_damage"] == 1.0
    assert info["zone_damage"] == [1.0]
    assert info["reward_breakdown"]["damage"] == -30.0
    assert info["reward_breakdown"]["terminal"] == -25.0
    assert reward == -55.1


def test_v1_duplicate_target_assignment_adds_conflict_penalty() -> None:
    config = AirDefenseV1EnvConfig(
        protected_zones=(
            ProtectedZoneConfig(position=(0.0, 0.0), radius=2.0, value=1.0),
        ),
        defense_units=(
            DefenseUnitV1Config(
                resource_type="missile",
                position=(10.0, 0.0),
                ammo=1,
                max_range=100.0,
                base_hit_probability=0.0,
                cost=0.0,
            ),
            DefenseUnitV1Config(
                resource_type="missile",
                position=(11.0, 0.0),
                ammo=1,
                max_range=100.0,
                base_hit_probability=0.0,
                cost=0.0,
            ),
        ),
        targets=(
            TargetV1Config(
                position=(10.0, 0.0),
                speed=0.0,
                threat=1.0,
                target_zone=0,
                payload=1.0,
            ),
        ),
        max_steps=3,
    )
    env = AirDefenseResourceAssignmentEnvV1(config=config)
    env.reset(seed=0)

    _, _, _, _, info = env.step(np.asarray([0, 0]))

    assert info["shots"] == 2
    assert info["reward_breakdown"]["conflict"] == -1.0
    assert info["reward_breakdown"]["overkill"] == -0.5
