import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")

from rein_learning.envs import (
    AirDefenseV1EnvConfig,
    DefenseUnitV1Config,
    ProtectedZoneConfig,
    TargetV1Config,
)
from rein_learning.trainers.air_defense_v1_ppo import (
    AirDefenseV1PPOConfig,
    evaluate_air_defense_v1_model,
    train_maskable_ppo,
    train_ppo,
)


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
