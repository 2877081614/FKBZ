from __future__ import annotations

from dataclasses import dataclass


RESOURCE_TYPE_IDS = {
    "missile": 0,
    "laser": 1,
}


@dataclass(frozen=True)
class DefenseUnitConfig:
    resource_type: str
    position: tuple[float, float]
    ammo: int
    max_range: float
    base_hit_probability: float
    cost: float
    cooldown_after_fire: int = 0


@dataclass(frozen=True)
class TargetConfig:
    position: tuple[float, float]
    speed: float
    threat: float
    evasion: float = 0.0


@dataclass(frozen=True)
class AirDefenseEnvConfig:
    defense_units: tuple[DefenseUnitConfig, ...] = (
        DefenseUnitConfig(
            resource_type="missile",
            position=(-10.0, 0.0),
            ammo=3,
            max_range=80.0,
            base_hit_probability=0.85,
            cost=2.0,
            cooldown_after_fire=1,
        ),
        DefenseUnitConfig(
            resource_type="missile",
            position=(10.0, 0.0),
            ammo=3,
            max_range=80.0,
            base_hit_probability=0.85,
            cost=2.0,
            cooldown_after_fire=1,
        ),
        DefenseUnitConfig(
            resource_type="laser",
            position=(0.0, 10.0),
            ammo=10,
            max_range=50.0,
            base_hit_probability=0.65,
            cost=0.5,
            cooldown_after_fire=0,
        ),
    )
    targets: tuple[TargetConfig, ...] = ()
    num_random_targets: int = 5
    map_size: float = 100.0
    asset_position: tuple[float, float] = (0.0, 0.0)
    asset_radius: float = 5.0
    target_spawn_min_distance: float = 60.0
    target_spawn_max_distance: float = 100.0
    target_min_speed: float = 1.0
    target_max_speed: float = 3.0
    target_min_threat: float = 0.5
    target_max_threat: float = 1.0
    dt: float = 1.0
    max_steps: int = 50
    max_allowed_leaks: int | None = None
    intercept_reward_weight: float = 10.0
    leak_penalty_weight: float = 20.0
    invalid_action_penalty: float = -5.0
    time_penalty: float = -0.1
    success_bonus: float = 20.0
    failure_penalty: float = -20.0
