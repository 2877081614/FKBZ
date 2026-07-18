import pytest
from gymnasium import spaces

pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")
from sb3_contrib import MaskablePPO

from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    AirDefenseV1EnvConfig,
    ConflictFreeJointActionWrapper,
    DefenseUnitV1Config,
    ProtectedZoneConfig,
    TargetV1Config,
)
from rein_learning.trainers.air_defense_v1_ppo import (
    AirDefenseV1PPOConfig,
    evaluate_air_defense_v1_model,
    train_autoregressive_maskable_ppo,
    train_conflict_free_maskable_ppo,
    train_maskable_ppo,
    train_ppo,
)
from rein_learning.algorithms.policy_gradient import AutoregressiveMaskablePPO


def make_tiny_training_config() -> AirDefenseV1EnvConfig:
    return AirDefenseV1EnvConfig(
        protected_zones=(
            ProtectedZoneConfig(
                position=(0.0, 0.0),
                radius=2.0,
                value=1.0,
            ),
        ),
        defense_units=(
            DefenseUnitV1Config(
                resource_type="missile",
                position=(10.0, 0.0),
                ammo=2,
                max_range=100.0,
                base_hit_probability=1.0,
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


def make_tiny_ppo_config() -> AirDefenseV1PPOConfig:
    return AirDefenseV1PPOConfig(
        total_timesteps=16,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        net_arch=(16,),
        seed=0,
        verbose=0,
        device="cpu",
    )


def test_train_ppo_smoke_evaluates_with_v1_metrics() -> None:
    model = train_ppo(
        env_config=make_tiny_training_config(),
        train_config=make_tiny_ppo_config(),
    )

    metrics = evaluate_air_defense_v1_model(
        model,
        env_config=make_tiny_training_config(),
        episodes=2,
        seed=10,
        use_action_masks=False,
    )

    assert metrics["episodes"] == 2.0
    assert metrics["avg_steps"] > 0.0
    assert 0.0 <= metrics["intercept_rate"] <= 1.0


def test_train_maskable_ppo_smoke_uses_action_masks() -> None:
    model = train_maskable_ppo(
        env_config=make_tiny_training_config(),
        train_config=make_tiny_ppo_config(),
    )

    metrics = evaluate_air_defense_v1_model(
        model,
        env_config=make_tiny_training_config(),
        episodes=2,
        seed=20,
        use_action_masks=True,
    )

    assert metrics["episodes"] == 2.0
    assert metrics["avg_steps"] > 0.0
    assert metrics["avg_invalid_actions"] == 0.0
    assert metrics["avg_decision_time_ms"] >= 0.0
    assert 0.0 <= metrics["high_threat_leak_rate"] <= 1.0
    assert 0.0 <= metrics["assignment_conflict_rate"] <= 1.0
    assert 0.0 <= metrics["overkill_rate"] <= 1.0


def test_train_conflict_free_maskable_ppo_uses_discrete_joint_masks() -> None:
    env_config = make_tiny_training_config()
    model = train_conflict_free_maskable_ppo(
        env_config=env_config,
        train_config=make_tiny_ppo_config(),
    )

    assert isinstance(model.action_space, spaces.Discrete)
    metrics = evaluate_air_defense_v1_model(
        model,
        env_factory=lambda: ConflictFreeJointActionWrapper(
            AirDefenseResourceAssignmentEnvV1(config=env_config)
        ),
        episodes=2,
        seed=30,
        use_action_masks=True,
    )

    assert metrics["avg_invalid_actions"] == 0.0
    assert metrics["assignment_conflict_rate"] == 0.0
    assert metrics["overkill_rate"] == 0.0


def test_train_autoregressive_maskable_ppo_reuses_multidiscrete_environment() -> None:
    env_config = make_tiny_training_config()
    model = train_autoregressive_maskable_ppo(
        env_config=env_config,
        train_config=make_tiny_ppo_config(),
    )

    assert isinstance(model, AutoregressiveMaskablePPO)
    assert isinstance(model.action_space, spaces.MultiDiscrete)
    metrics = evaluate_air_defense_v1_model(
        model,
        env_config=env_config,
        episodes=2,
        seed=35,
        use_action_masks=True,
    )

    assert metrics["avg_invalid_actions"] == 0.0
    assert metrics["assignment_conflict_rate"] == 0.0
    assert metrics["overkill_rate"] == 0.0


def test_autoregressive_model_save_load_preserves_generator_signature(tmp_path) -> None:
    env_config = make_tiny_training_config()
    save_path = tmp_path / "autoregressive_model"
    model = train_autoregressive_maskable_ppo(
        env_config=env_config,
        train_config=make_tiny_ppo_config(),
        save_path=save_path,
    )

    env = AirDefenseResourceAssignmentEnvV1(config=env_config)
    loaded = AutoregressiveMaskablePPO.load(save_path, env=env)
    assert loaded.action_generator_signature == model.action_generator_signature
    action, _ = loaded.predict(
        env.reset(seed=37)[0],
        deterministic=True,
        action_masks=env.action_masks(),
    )
    assert env.action_space.contains(action)
    env.close()


def test_autoregressive_model_preserves_nondefault_unit_order(tmp_path) -> None:
    env_config = make_tiny_training_config()
    save_path = tmp_path / "autoregressive_order_model"
    model = train_autoregressive_maskable_ppo(
        env_config=env_config,
        train_config=make_tiny_ppo_config(),
        save_path=save_path,
        unit_order=(0,),
    )

    env = AirDefenseResourceAssignmentEnvV1(config=env_config)
    loaded = AutoregressiveMaskablePPO.load(save_path, env=env)
    assert model.policy.unit_order == (0,)
    assert loaded.policy.unit_order == (0,)
    assert loaded.action_generator_signature["unit_order"] == [0]
    env.close()


def test_autoregressive_loader_rejects_independent_maskable_archive(tmp_path) -> None:
    env_config = make_tiny_training_config()
    save_path = tmp_path / "independent_maskable_model"
    train_maskable_ppo(
        env_config=env_config,
        train_config=make_tiny_ppo_config(),
        save_path=save_path,
    )

    env = AirDefenseResourceAssignmentEnvV1(config=env_config)
    with pytest.raises(ValueError, match="requires.*Autoregressive"):
        AutoregressiveMaskablePPO.load(save_path, env=env)
    env.close()


def test_conflict_free_model_load_requires_matching_discrete_space(tmp_path) -> None:
    env_config = make_tiny_training_config()
    save_path = tmp_path / "conflict_free_model"
    train_conflict_free_maskable_ppo(
        env_config=env_config,
        train_config=make_tiny_ppo_config(),
        save_path=save_path,
    )

    wrapped_env = ConflictFreeJointActionWrapper(
        AirDefenseResourceAssignmentEnvV1(config=env_config)
    )
    loaded_model = MaskablePPO.load(save_path, env=wrapped_env)
    assert isinstance(loaded_model.action_space, spaces.Discrete)
    wrapped_env.close()

    base_env = AirDefenseResourceAssignmentEnvV1(config=env_config)
    with pytest.raises(ValueError, match="Action spaces do not match"):
        MaskablePPO.load(save_path, env=base_env)
    base_env.close()
