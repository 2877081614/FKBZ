from itertools import product

import numpy as np
import pytest

import rein_learning.baselines.air_defense_v1 as baseline_module
from rein_learning.baselines import (
    GreedyDamageReductionPolicy,
    HungarianDamageReductionPolicy,
    build_expected_damage_reduction_matrix,
    expected_damage_reduction_score,
)
from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    AirDefenseV1EnvConfig,
    DefenseUnitV1Config,
    ProtectedZoneConfig,
    TargetV1Config,
)


class _MatrixEnv:
    num_defense_units = 3
    num_targets = 2
    noop_action = 2


def make_assignment_test_env() -> AirDefenseResourceAssignmentEnvV1:
    config = AirDefenseV1EnvConfig(
        protected_zones=(
            ProtectedZoneConfig(position=(0.0, 0.0), radius=2.0, value=1.0),
        ),
        defense_units=(
            DefenseUnitV1Config(
                resource_type="missile",
                position=(0.0, 0.0),
                ammo=3,
                max_range=100.0,
                base_hit_probability=0.9,
                cost=1.0,
            ),
            DefenseUnitV1Config(
                resource_type="laser",
                position=(25.0, 0.0),
                ammo=3,
                max_range=20.0,
                base_hit_probability=0.8,
                cost=0.5,
            ),
            DefenseUnitV1Config(
                resource_type="missile",
                position=(5.0, 0.0),
                ammo=3,
                max_range=100.0,
                base_hit_probability=0.7,
                cost=1.5,
            ),
        ),
        targets=(
            TargetV1Config(
                position=(10.0, 0.0),
                speed=0.0,
                threat=0.9,
                target_zone=0,
                payload=1.0,
            ),
            TargetV1Config(
                position=(80.0, 0.0),
                speed=0.0,
                threat=0.7,
                target_zone=0,
                payload=0.8,
            ),
        ),
        max_steps=3,
    )
    env = AirDefenseResourceAssignmentEnvV1(config=config)
    env.reset(seed=0)
    return env


def _action_objective(
    env: AirDefenseResourceAssignmentEnvV1,
    action: np.ndarray,
) -> float:
    return float(
        sum(
            expected_damage_reduction_score(env, unit_index, int(target_index))
            for unit_index, target_index in enumerate(action)
            if target_index != env.noop_action
        )
    )


def _brute_force_objective(env: AirDefenseResourceAssignmentEnvV1) -> float:
    best_objective = 0.0
    action_choices = range(env.num_targets + 1)
    for candidate in product(action_choices, repeat=env.num_defense_units):
        assigned_targets = [
            target_index
            for target_index in candidate
            if target_index != env.noop_action
        ]
        if len(assigned_targets) != len(set(assigned_targets)):
            continue
        if any(
            target_index != env.noop_action
            and not env.is_unit_target_action_legal(unit_index, target_index)
            for unit_index, target_index in enumerate(candidate)
        ):
            continue
        objective = _action_objective(env, np.asarray(candidate, dtype=np.int64))
        best_objective = max(best_objective, objective)
    return best_objective


def test_hungarian_solver_finds_global_assignment_with_independent_noops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_matrix = np.asarray(
        [
            [10.0, 9.0],
            [9.0, -np.inf],
            [8.0, 7.0],
        ]
    )
    monkeypatch.setattr(
        baseline_module,
        "build_expected_damage_reduction_matrix",
        lambda env: score_matrix,
    )

    action = HungarianDamageReductionPolicy().select_action(_MatrixEnv())

    assert action.tolist() == [1, 0, 2]


def test_hungarian_solver_keeps_resources_for_nonpositive_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_matrix = np.asarray(
        [
            [-1.0, 0.0],
            [-2.0, -3.0],
            [0.0, -4.0],
        ]
    )
    monkeypatch.setattr(
        baseline_module,
        "build_expected_damage_reduction_matrix",
        lambda env: score_matrix,
    )

    action = HungarianDamageReductionPolicy().select_action(_MatrixEnv())

    assert action.tolist() == [2, 2, 2]


def test_expected_damage_matrix_marks_illegal_assignments() -> None:
    env = make_assignment_test_env()

    score_matrix = build_expected_damage_reduction_matrix(env)

    assert score_matrix.shape == (3, 2)
    assert np.isfinite(score_matrix[0, 0])
    assert np.isfinite(score_matrix[0, 1])
    assert np.isfinite(score_matrix[1, 0])
    assert score_matrix[1, 1] == -np.inf


def test_hungarian_policy_is_legal_one_to_one_and_deterministic() -> None:
    env = make_assignment_test_env()
    policy = HungarianDamageReductionPolicy()

    first_action = policy.select_action(env)
    second_action = policy.select_action(env)
    assigned_targets = first_action[first_action != env.noop_action]

    assert np.array_equal(first_action, second_action)
    assert len(assigned_targets) == len(set(assigned_targets.tolist()))
    assert all(
        target_index == env.noop_action
        or env.is_unit_target_action_legal(unit_index, int(target_index))
        for unit_index, target_index in enumerate(first_action)
    )


def test_hungarian_objective_matches_brute_force_and_beats_greedy() -> None:
    env = make_assignment_test_env()
    hungarian_action = HungarianDamageReductionPolicy().select_action(env)
    greedy_action = GreedyDamageReductionPolicy().select_action(env)

    hungarian_objective = _action_objective(env, hungarian_action)
    greedy_objective = _action_objective(env, greedy_action)

    assert hungarian_objective == pytest.approx(_brute_force_objective(env))
    assert hungarian_objective >= greedy_objective
