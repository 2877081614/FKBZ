from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .config import (
    AirDefenseV1EnvConfig,
    DefenseUnitV1Config,
    ProtectedZoneConfig,
)


ScenarioKind = Literal["baseline", "difficulty", "pressure"]

AIR_DEFENSE_V1_DEFAULT_SCENARIO = "medium"
AIR_DEFENSE_V1_DIFFICULTY_SCENARIOS = ("easy", "medium", "hard")
AIR_DEFENSE_V1_PRESSURE_SCENARIOS = (
    "time_pressure",
    "resource_pressure",
    "intercept_uncertainty",
    "damage_pressure",
    "heterogeneity_pressure",
)


@dataclass(frozen=True)
class AirDefenseV1ScenarioProfile:
    """Named, immutable v1.0 scenario configuration and its design intent."""

    name: str
    kind: ScenarioKind
    description: str
    pressure_axes: tuple[str, ...]
    changed_fields: tuple[str, ...]
    config: AirDefenseV1EnvConfig


def _replace_units(
    config: AirDefenseV1EnvConfig,
    *unit_updates: dict[str, float | int],
) -> tuple[DefenseUnitV1Config, ...]:
    if len(unit_updates) != len(config.defense_units):
        raise ValueError("unit_updates must match the number of defense units")
    return tuple(
        replace(unit, **updates)
        for unit, updates in zip(config.defense_units, unit_updates)
    )


# Keep the formal v1.0 benchmark independent from future dataclass default changes.
_MEDIUM_CONFIG = AirDefenseV1EnvConfig(
    protected_zones=(
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
    ),
    defense_units=(
        DefenseUnitV1Config(
            resource_type="missile",
            position=(-12.0, 0.0),
            ammo=3,
            max_range=85.0,
            base_hit_probability=0.88,
            cost=2.0,
            cooldown_after_fire=1,
            energy=1.0,
        ),
        DefenseUnitV1Config(
            resource_type="missile",
            position=(12.0, 0.0),
            ammo=3,
            max_range=85.0,
            base_hit_probability=0.88,
            cost=2.0,
            cooldown_after_fire=1,
            energy=1.0,
        ),
        DefenseUnitV1Config(
            resource_type="laser",
            position=(3.0, 12.0),
            ammo=10,
            max_range=55.0,
            base_hit_probability=0.68,
            cost=0.5,
            cooldown_after_fire=0,
            energy=1.0,
        ),
    ),
    targets=(),
    num_random_targets=5,
    map_size=100.0,
    target_spawn_min_distance=60.0,
    target_spawn_max_distance=100.0,
    target_min_speed=1.0,
    target_max_speed=3.0,
    target_min_threat=0.5,
    target_max_threat=1.0,
    target_min_payload=0.6,
    target_max_payload=1.5,
    dt=1.0,
    max_steps=50,
    max_allowed_damage=2.5,
    success_damage_threshold=0.0,
    intercept_reward_weight=8.0,
    damage_penalty_weight=30.0,
    invalid_action_penalty=-5.0,
    time_penalty=-0.1,
    assignment_conflict_penalty=1.0,
    overkill_penalty=0.5,
    success_bonus=25.0,
    failure_penalty=-25.0,
)

_EASY_CONFIG = replace(
    _MEDIUM_CONFIG,
    defense_units=_replace_units(
        _MEDIUM_CONFIG,
        {"ammo": 4, "base_hit_probability": 0.93},
        {"ammo": 4, "base_hit_probability": 0.93},
        {"ammo": 12, "base_hit_probability": 0.75},
    ),
    target_min_speed=0.8,
    target_max_speed=2.4,
    target_min_threat=0.4,
    target_max_threat=0.85,
    target_min_payload=0.5,
    target_max_payload=1.2,
)

_HARD_CONFIG = replace(
    _MEDIUM_CONFIG,
    defense_units=_replace_units(
        _MEDIUM_CONFIG,
        {"ammo": 2, "base_hit_probability": 0.80},
        {"ammo": 2, "base_hit_probability": 0.80},
        {"ammo": 7, "base_hit_probability": 0.58},
    ),
    target_min_speed=1.8,
    target_max_speed=3.5,
    target_min_threat=0.7,
    target_max_threat=1.1,
    target_min_payload=0.9,
    target_max_payload=1.8,
)

_TIME_PRESSURE_CONFIG = replace(
    _MEDIUM_CONFIG,
    target_min_speed=2.0,
    target_max_speed=3.5,
)

_RESOURCE_PRESSURE_CONFIG = replace(
    _MEDIUM_CONFIG,
    defense_units=_replace_units(
        _MEDIUM_CONFIG,
        {"ammo": 2},
        {"ammo": 2},
        {"ammo": 6},
    ),
)

_INTERCEPT_UNCERTAINTY_CONFIG = replace(
    _MEDIUM_CONFIG,
    defense_units=_replace_units(
        _MEDIUM_CONFIG,
        {"base_hit_probability": 0.72},
        {"base_hit_probability": 0.72},
        {"base_hit_probability": 0.52},
    ),
)

_DAMAGE_PRESSURE_CONFIG = replace(
    _MEDIUM_CONFIG,
    target_min_threat=0.8,
    target_max_threat=1.2,
    target_min_payload=1.0,
    target_max_payload=1.8,
)

_HETEROGENEITY_PRESSURE_CONFIG = replace(
    _MEDIUM_CONFIG,
    defense_units=_replace_units(
        _MEDIUM_CONFIG,
        {
            "max_range": 92.0,
            "base_hit_probability": 0.94,
            "cost": 2.8,
        },
        {
            "max_range": 72.0,
            "base_hit_probability": 0.78,
            "cost": 1.5,
        },
        {
            "max_range": 45.0,
            "base_hit_probability": 0.50,
            "cost": 0.25,
        },
    ),
)


_SCENARIO_PROFILES = {
    "easy": AirDefenseV1ScenarioProfile(
        name="easy",
        kind="difficulty",
        description="Lower target pressure with more ammunition and higher hit probability.",
        pressure_axes=("time", "resources", "intercept", "damage"),
        changed_fields=(
            "defense_units",
            "target_min_speed",
            "target_max_speed",
            "target_min_threat",
            "target_max_threat",
            "target_min_payload",
            "target_max_payload",
        ),
        config=_EASY_CONFIG,
    ),
    "medium": AirDefenseV1ScenarioProfile(
        name="medium",
        kind="baseline",
        description="Frozen AirDefenseResourceAssignmentEnv v1.0 formal benchmark.",
        pressure_axes=(),
        changed_fields=(),
        config=_MEDIUM_CONFIG,
    ),
    "hard": AirDefenseV1ScenarioProfile(
        name="hard",
        kind="difficulty",
        description="Higher target pressure with scarcer and less reliable resources.",
        pressure_axes=("time", "resources", "intercept", "damage"),
        changed_fields=(
            "defense_units",
            "target_min_speed",
            "target_max_speed",
            "target_min_threat",
            "target_max_threat",
            "target_min_payload",
            "target_max_payload",
        ),
        config=_HARD_CONFIG,
    ),
    "time_pressure": AirDefenseV1ScenarioProfile(
        name="time_pressure",
        kind="pressure",
        description="Medium scenario with only target speed pressure increased.",
        pressure_axes=("time",),
        changed_fields=("target_min_speed", "target_max_speed"),
        config=_TIME_PRESSURE_CONFIG,
    ),
    "resource_pressure": AirDefenseV1ScenarioProfile(
        name="resource_pressure",
        kind="pressure",
        description="Medium scenario with only ammunition availability reduced.",
        pressure_axes=("resources",),
        changed_fields=("defense_units",),
        config=_RESOURCE_PRESSURE_CONFIG,
    ),
    "intercept_uncertainty": AirDefenseV1ScenarioProfile(
        name="intercept_uncertainty",
        kind="pressure",
        description="Medium scenario with only defense-unit hit probabilities reduced.",
        pressure_axes=("intercept",),
        changed_fields=("defense_units",),
        config=_INTERCEPT_UNCERTAINTY_CONFIG,
    ),
    "damage_pressure": AirDefenseV1ScenarioProfile(
        name="damage_pressure",
        kind="pressure",
        description="Medium scenario with only target threat and payload pressure increased.",
        pressure_axes=("damage",),
        changed_fields=(
            "target_min_threat",
            "target_max_threat",
            "target_min_payload",
            "target_max_payload",
        ),
        config=_DAMAGE_PRESSURE_CONFIG,
    ),
    "heterogeneity_pressure": AirDefenseV1ScenarioProfile(
        name="heterogeneity_pressure",
        kind="pressure",
        description="Medium scenario with wider defense-unit range, reliability, and cost gaps.",
        pressure_axes=("heterogeneity",),
        changed_fields=("defense_units",),
        config=_HETEROGENEITY_PRESSURE_CONFIG,
    ),
}

AIR_DEFENSE_V1_SCENARIO_NAMES = tuple(_SCENARIO_PROFILES)

_SCENARIO_ALIASES = {
    "default": AIR_DEFENSE_V1_DEFAULT_SCENARIO,
    "v1_default": AIR_DEFENSE_V1_DEFAULT_SCENARIO,
}


def default_air_defense_v1_config() -> AirDefenseV1EnvConfig:
    """Return the frozen configuration used by the formal v1.0 benchmark."""

    return _MEDIUM_CONFIG


def get_air_defense_v1_scenario_profile(
    name: str,
) -> AirDefenseV1ScenarioProfile:
    """Return scenario metadata and configuration for a canonical name or alias."""

    normalized_name = _normalize_scenario_name(name)
    canonical_name = _SCENARIO_ALIASES.get(normalized_name, normalized_name)
    try:
        return _SCENARIO_PROFILES[canonical_name]
    except KeyError as exc:
        available = ", ".join(AIR_DEFENSE_V1_SCENARIO_NAMES)
        raise ValueError(
            f"Unknown AirDefense v1 scenario {name!r}. Available: {available}"
        ) from exc


def get_air_defense_v1_scenario(name: str) -> AirDefenseV1EnvConfig:
    """Return an immutable environment configuration for a named scenario."""

    return get_air_defense_v1_scenario_profile(name).config


def list_air_defense_v1_scenarios(
    kind: ScenarioKind | None = None,
) -> tuple[str, ...]:
    """List canonical scenario names, optionally filtered by profile kind."""

    if kind is None:
        return AIR_DEFENSE_V1_SCENARIO_NAMES
    if kind not in ("baseline", "difficulty", "pressure"):
        raise ValueError(f"Unsupported scenario kind: {kind!r}")
    return tuple(
        name
        for name, profile in _SCENARIO_PROFILES.items()
        if profile.kind == kind
    )


def _normalize_scenario_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Scenario name must be a non-empty string")
    return name.strip().lower().replace("-", "_")
