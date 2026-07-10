from __future__ import annotations

from dataclasses import dataclass


RESOURCE_TYPE_IDS = {
    "missile": 0,
    "laser": 1,
}

ZONE_TYPE_IDS = {
    "command": 0,
    "radar": 1,
    "logistics": 2,
}

TARGET_CLASS_IDS = {
    "small_uav": 0,
    "loitering_munition": 1,
    "decoy": 2,
}


@dataclass(frozen=True)
class ProtectedZoneConfig:
    position: tuple[float, float]
    radius: float
    value: float
    priority: float = 1.0
    zone_type: str = "command"


@dataclass(frozen=True)
class DefenseUnitV1Config:
    resource_type: str
    position: tuple[float, float]
    ammo: int
    max_range: float
    base_hit_probability: float
    cost: float
    cooldown_after_fire: int = 0
    energy: float = 1.0


@dataclass(frozen=True)
class TargetV1Config:
    position: tuple[float, float]
    speed: float
    threat: float
    target_zone: int
    payload: float = 1.0
    evasion: float = 0.0
    target_class: str = "small_uav"


@dataclass(frozen=True)
class AirDefenseV1EnvConfig:
    protected_zones: tuple[ProtectedZoneConfig, ...] = (
        ProtectedZoneConfig(
            position=(0.0, 0.0),
            radius=5.0,
            value=1.0,
            priority=1.0,
            zone_type="command",
        ),
        ProtectedZoneConfig(
            position=(25.0, -10.0),
            radius=4.0,
            value=0.8,
            priority=0.8,
            zone_type="radar",
        ),
    )
    defense_units: tuple[DefenseUnitV1Config, ...] = (
        DefenseUnitV1Config(
            resource_type="missile",
            position=(-12.0, 0.0),
            ammo=3,
            max_range=85.0,
            base_hit_probability=0.88,
            cost=2.0,
            cooldown_after_fire=1,
        ),
        DefenseUnitV1Config(
            resource_type="missile",
            position=(12.0, 0.0),
            ammo=3,
            max_range=85.0,
            base_hit_probability=0.88,
            cost=2.0,
            cooldown_after_fire=1,
        ),
        DefenseUnitV1Config(
            resource_type="laser",
            position=(3.0, 12.0),
            ammo=10,
            max_range=55.0,
            base_hit_probability=0.68,
            cost=0.5,
            cooldown_after_fire=0,
        ),
    )
    targets: tuple[TargetV1Config, ...] = ()
    num_random_targets: int = 5
    map_size: float = 100.0
    target_spawn_min_distance: float = 60.0
    target_spawn_max_distance: float = 100.0
    target_min_speed: float = 1.0
    target_max_speed: float = 3.0
    target_min_threat: float = 0.5
    target_max_threat: float = 1.0
    target_min_payload: float = 0.6
    target_max_payload: float = 1.5
    dt: float = 1.0
    max_steps: int = 50
    max_allowed_damage: float = 2.5
    success_damage_threshold: float = 0.0
    intercept_reward_weight: float = 8.0
    damage_penalty_weight: float = 30.0
    invalid_action_penalty: float = -5.0
    time_penalty: float = -0.1
    assignment_conflict_penalty: float = 1.0
    overkill_penalty: float = 0.5
    success_bonus: float = 25.0
    failure_penalty: float = -25.0
