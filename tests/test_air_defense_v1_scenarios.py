from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from rein_learning.envs import (
    AIR_DEFENSE_V1_DIFFICULTY_SCENARIOS,
    AIR_DEFENSE_V1_PRESSURE_SCENARIOS,
    AIR_DEFENSE_V1_SCENARIO_NAMES,
    AirDefenseResourceAssignmentEnvV1,
    AirDefenseV1EnvConfig,
    default_air_defense_v1_config,
    get_air_defense_v1_scenario,
    get_air_defense_v1_scenario_profile,
    list_air_defense_v1_scenarios,
)


EXPECTED_PRESSURE_CHANGED_FIELDS = {
    "time_pressure": {"target_min_speed", "target_max_speed"},
    "resource_pressure": {"defense_units"},
    "intercept_uncertainty": {"defense_units"},
    "damage_pressure": {
        "target_min_threat",
        "target_max_threat",
        "target_min_payload",
        "target_max_payload",
    },
    "heterogeneity_pressure": {"defense_units"},
}


def _changed_top_level_fields(
    baseline: AirDefenseV1EnvConfig,
    candidate: AirDefenseV1EnvConfig,
) -> set[str]:
    baseline_values = asdict(baseline)
    candidate_values = asdict(candidate)
    return {
        field_name
        for field_name, baseline_value in baseline_values.items()
        if candidate_values[field_name] != baseline_value
    }


def test_medium_scenario_is_the_frozen_v1_default_config() -> None:
    default_config = AirDefenseV1EnvConfig()

    assert default_air_defense_v1_config() == default_config
    assert get_air_defense_v1_scenario("medium") == default_config
    assert get_air_defense_v1_scenario("default") == default_config
    assert get_air_defense_v1_scenario("v1-default") == default_config
    assert AirDefenseResourceAssignmentEnvV1().config == default_config


def test_scenario_registry_lists_canonical_names_by_kind() -> None:
    assert list_air_defense_v1_scenarios() == AIR_DEFENSE_V1_SCENARIO_NAMES
    assert list_air_defense_v1_scenarios("difficulty") == (
        "easy",
        "hard",
    )
    assert list_air_defense_v1_scenarios("baseline") == ("medium",)
    assert list_air_defense_v1_scenarios("pressure") == (
        AIR_DEFENSE_V1_PRESSURE_SCENARIOS
    )
    assert AIR_DEFENSE_V1_DIFFICULTY_SCENARIOS == ("easy", "medium", "hard")


def test_unknown_or_empty_scenario_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown AirDefense v1 scenario"):
        get_air_defense_v1_scenario("unknown")
    with pytest.raises(ValueError, match="non-empty"):
        get_air_defense_v1_scenario("  ")
    with pytest.raises(ValueError, match="Unsupported scenario kind"):
        list_air_defense_v1_scenarios("other")  # type: ignore[arg-type]


@pytest.mark.parametrize("scenario_name", AIR_DEFENSE_V1_SCENARIO_NAMES)
def test_all_scenarios_keep_the_frozen_space_dimensions(scenario_name: str) -> None:
    baseline_env = AirDefenseResourceAssignmentEnvV1(
        config=get_air_defense_v1_scenario("medium")
    )
    scenario_env = AirDefenseResourceAssignmentEnvV1(
        config=get_air_defense_v1_scenario(scenario_name)
    )

    assert scenario_env.num_zones == baseline_env.num_zones == 2
    assert scenario_env.num_defense_units == baseline_env.num_defense_units == 3
    assert scenario_env.num_targets == baseline_env.num_targets == 5
    assert scenario_env.observation_space.shape == baseline_env.observation_space.shape
    assert np.array_equal(scenario_env.action_space.nvec, baseline_env.action_space.nvec)


@pytest.mark.parametrize("scenario_name", AIR_DEFENSE_V1_SCENARIO_NAMES)
def test_all_scenarios_are_reproducible_with_the_same_seed(
    scenario_name: str,
) -> None:
    env = AirDefenseResourceAssignmentEnvV1(
        config=get_air_defense_v1_scenario(scenario_name)
    )

    first_obs, first_info = env.reset(seed=1234)
    second_obs, second_info = env.reset(seed=1234)

    assert np.array_equal(first_obs, second_obs)
    assert first_info == second_info


@pytest.mark.parametrize("scenario_name", AIR_DEFENSE_V1_SCENARIO_NAMES)
def test_all_scenarios_can_run_to_a_terminal_or_truncated_state(
    scenario_name: str,
) -> None:
    env = AirDefenseResourceAssignmentEnvV1(
        config=get_air_defense_v1_scenario(scenario_name)
    )
    env.reset(seed=7)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = np.full(
            env.num_defense_units,
            env.noop_action,
            dtype=np.int64,
        )
        _, _, terminated, truncated, _ = env.step(action)

    assert terminated or truncated
    assert env.current_step <= env.config.max_steps


@pytest.mark.parametrize(
    ("scenario_name", "expected_changed_fields"),
    EXPECTED_PRESSURE_CHANGED_FIELDS.items(),
)
def test_single_axis_pressure_scenarios_only_change_declared_fields(
    scenario_name: str,
    expected_changed_fields: set[str],
) -> None:
    baseline = get_air_defense_v1_scenario("medium")
    profile = get_air_defense_v1_scenario_profile(scenario_name)

    assert profile.kind == "pressure"
    assert len(profile.pressure_axes) == 1
    assert set(profile.changed_fields) == expected_changed_fields
    assert _changed_top_level_fields(baseline, profile.config) == (
        expected_changed_fields
    )


def test_easy_and_hard_profiles_move_pressure_in_expected_directions() -> None:
    easy = get_air_defense_v1_scenario("easy")
    medium = get_air_defense_v1_scenario("medium")
    hard = get_air_defense_v1_scenario("hard")

    assert sum(unit.ammo for unit in easy.defense_units) > sum(
        unit.ammo for unit in medium.defense_units
    )
    assert sum(unit.ammo for unit in hard.defense_units) < sum(
        unit.ammo for unit in medium.defense_units
    )
    assert easy.target_max_speed < medium.target_max_speed < hard.target_max_speed
    assert easy.target_max_payload < medium.target_max_payload < hard.target_max_payload
    assert easy.target_max_threat < medium.target_max_threat < hard.target_max_threat
    for easy_unit, medium_unit, hard_unit in zip(
        easy.defense_units,
        medium.defense_units,
        hard.defense_units,
    ):
        assert easy_unit.base_hit_probability > medium_unit.base_hit_probability
        assert hard_unit.base_hit_probability < medium_unit.base_hit_probability
