from __future__ import annotations

import numpy as np
import pytest

from rein_learning.common.dynamic_support_distance import (
    DynamicSupportNotApplicableError,
    EmptySupportUnionError,
    IllegalCurrentActionError,
    IllegalPrefixError,
    dynamic_support_cost_matrix,
    dynamic_support_jaccard,
    dynamic_support_policy_distance,
    enumerate_feasible_suffixes,
    jaccard_distance,
    old_policy_structural_risk,
    suffix_count,
)
from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    ConflictFreeJointActionCodec,
    get_air_defense_v1_scenario,
)


ALL_LEGAL_3X2 = np.ones((3, 3), dtype=bool)
NOOP = 2
ORDERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


def _codec_suffixes(
    base_mask: np.ndarray,
    prefix: tuple[int, ...],
    unit_order: tuple[int, ...],
) -> set[tuple[int, ...]]:
    codec = ConflictFreeJointActionCodec(
        num_units=base_mask.shape[0],
        num_targets=base_mask.shape[1] - 1,
    )
    expected: set[tuple[int, ...]] = set()
    for joint in codec.joint_actions:
        if not all(
            base_mask[unit_index, action]
            for unit_index, action in enumerate(joint)
        ):
            continue
        ordered = tuple(joint[unit_index] for unit_index in unit_order)
        if ordered[: len(prefix)] == prefix:
            expected.add(ordered[len(prefix) :])
    return expected


def _legal_joint_actions(
    env: AirDefenseResourceAssignmentEnvV1,
) -> list[tuple[int, ...]]:
    base_mask = env.action_mask().astype(bool)
    codec = ConflictFreeJointActionCodec(
        num_units=env.num_defense_units,
        num_targets=env.num_targets,
    )
    return [
        joint
        for joint in codec.joint_actions
        if all(base_mask[unit_index, action] for unit_index, action in enumerate(joint))
    ]


def test_hand_computed_suffixes_capture_noop_preservation_and_target_occupancy() -> None:
    noop_suffixes = enumerate_feasible_suffixes(
        ALL_LEGAL_3X2,
        prefix=(NOOP,),
        unit_order=(0, 1, 2),
    )
    engage_suffixes = enumerate_feasible_suffixes(
        ALL_LEGAL_3X2,
        prefix=(0,),
        unit_order=(0, 1, 2),
    )

    assert set(noop_suffixes) == {
        (0, 1),
        (0, NOOP),
        (1, 0),
        (1, NOOP),
        (NOOP, 0),
        (NOOP, 1),
        (NOOP, NOOP),
    }
    assert set(engage_suffixes) == {
        (1, NOOP),
        (NOOP, 1),
        (NOOP, NOOP),
    }
    assert dynamic_support_jaccard(
        ALL_LEGAL_3X2,
        prefix=(),
        action_a=NOOP,
        action_b=0,
        unit_order=(0, 1, 2),
    ) == pytest.approx(4.0 / 7.0)
    assert suffix_count(
        ALL_LEGAL_3X2,
        prefix=(),
        action=0,
        unit_order=(0, 1, 2),
    ) == 3


def test_unavailable_unit_and_unreachable_targets_reduce_to_legal_noop_suffixes() -> None:
    unavailable = ALL_LEGAL_3X2.copy()
    unavailable[1] = (False, False, True)
    suffixes = enumerate_feasible_suffixes(
        unavailable,
        prefix=(0,),
        unit_order=(0, 1, 2),
    )
    assert suffixes == ((NOOP, 1), (NOOP, NOOP))

    no_targets = np.zeros((3, 3), dtype=bool)
    no_targets[:, NOOP] = True
    assert enumerate_feasible_suffixes(
        no_targets,
        prefix=(NOOP,),
        unit_order=(0, 1, 2),
    ) == ((NOOP, NOOP),)


def test_resource_heterogeneity_and_unit_order_use_real_prefix_semantics() -> None:
    mask = np.asarray(
        [
            [True, False, True],
            [True, True, True],
            [False, True, True],
        ],
        dtype=bool,
    )
    for order in ORDERS:
        prefix = (next(action for action in range(3) if mask[order[0], action]),)
        observed = set(enumerate_feasible_suffixes(mask, prefix, order))
        expected = _codec_suffixes(mask, prefix, order)
        assert observed == expected


def test_environment_adapter_tracks_ammo_cooldown_alive_range_and_occupancy() -> None:
    env = AirDefenseResourceAssignmentEnvV1(
        config=get_air_defense_v1_scenario("heterogeneity_pressure")
    )
    env.reset(seed=31)
    env.defense_units[0].ammo = 0
    env.defense_units[1].cooldown = 1
    env.targets[0].status = "intercepted"

    base_mask = env.action_mask().astype(bool)
    assert np.array_equal(
        np.asarray(env.action_mask(), dtype=bool),
        base_mask,
    )
    assert not base_mask[0, :-1].any()
    assert not base_mask[1, :-1].any()
    assert not base_mask[:, 0].any()
    for order in ORDERS:
        expected = _codec_suffixes(base_mask, (), order)
        observed = set(enumerate_feasible_suffixes(env, (), order))
        assert observed == expected
    env.close()


@pytest.mark.parametrize(
    ("scenario", "seed"),
    [
        ("medium", 101),
        ("time_pressure", 102),
        ("heterogeneity_pressure", 103),
    ],
)
def test_exact_enumerator_matches_discrete_joint_codec_on_randomized_states(
    scenario: str,
    seed: int,
) -> None:
    env = AirDefenseResourceAssignmentEnvV1(
        config=get_air_defense_v1_scenario(scenario)
    )
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    for _ in range(3):
        base_mask = env.action_mask().astype(bool)
        legal_joint = _legal_joint_actions(env)
        assert legal_joint
        factual_joint = legal_joint[int(rng.integers(len(legal_joint)))]
        for order in ORDERS:
            ordered_joint = tuple(factual_joint[unit] for unit in order)
            for prefix_length in range(env.num_defense_units + 1):
                prefix = ordered_joint[:prefix_length]
                expected = _codec_suffixes(base_mask, prefix, order)
                first = enumerate_feasible_suffixes(env, prefix, order)
                second = enumerate_feasible_suffixes(env, prefix, order)
                assert first == second
                assert set(first) == expected
                assert len(first) == len(set(first))
        _, _, terminated, truncated, _ = env.step(factual_joint)
        if terminated or truncated:
            break
    env.close()


def test_cost_matrix_is_deterministic_symmetric_bounded_and_zero_diagonal() -> None:
    first = dynamic_support_cost_matrix(ALL_LEGAL_3X2, (), (0, 1, 2))
    second = dynamic_support_cost_matrix(ALL_LEGAL_3X2, (), (0, 1, 2))

    assert first.actions == (0, 1, NOOP)
    assert first.suffix_counts == (3, 3, 7)
    assert np.array_equal(first.costs, second.costs)
    assert np.all(first.costs >= 0.0)
    assert np.all(first.costs <= 1.0)
    assert np.array_equal(first.costs, first.costs.T)
    assert np.array_equal(np.diag(first.costs), np.zeros(3))
    with pytest.raises(ValueError):
        first.costs[0, 1] = 0.0


def test_generic_jaccard_properties_and_empty_union_rule() -> None:
    assert jaccard_distance({(0,), (1,)}, {(0,), (1,)}) == 0.0
    assert jaccard_distance({(0,)}, {(1,)}) == 1.0
    assert jaccard_distance({(0,), (1,)}, {(1,), (2,)}) == pytest.approx(2 / 3)
    assert jaccard_distance({(1,), (2,)}, {(0,), (1,)}) == pytest.approx(2 / 3)
    with pytest.raises(EmptySupportUnionError, match="empty union"):
        jaccard_distance(set(), set())


def test_last_position_illegal_actions_and_bad_prefixes_are_explicit() -> None:
    with pytest.raises(DynamicSupportNotApplicableError, match="not_applicable"):
        dynamic_support_jaccard(
            ALL_LEGAL_3X2,
            prefix=(NOOP, NOOP),
            action_a=0,
            action_b=1,
            unit_order=(0, 1, 2),
        )
    illegal = ALL_LEGAL_3X2.copy()
    illegal[0, 0] = False
    with pytest.raises(IllegalCurrentActionError, match="illegal"):
        dynamic_support_jaccard(illegal, (), 0, NOOP, (0, 1, 2))
    with pytest.raises(IllegalPrefixError, match="more than once"):
        enumerate_feasible_suffixes(
            ALL_LEGAL_3X2,
            prefix=(0, 0),
            unit_order=(0, 1, 2),
        )


def test_structural_risk_and_policy_distance_follow_frozen_formula() -> None:
    costs = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    old = np.asarray([0.25, 0.75])
    new = np.asarray([0.75, 0.25])
    risk = old_policy_structural_risk(costs, old)

    assert np.allclose(risk, [0.75, 0.25])
    assert dynamic_support_policy_distance(old, old, risk) == 0.0
    assert dynamic_support_policy_distance(new, old, risk) == pytest.approx(0.25)
    assert 0.0 <= dynamic_support_policy_distance(new, old, risk) <= 1.0


@pytest.mark.parametrize(
    ("costs", "probabilities"),
    [
        (np.asarray([[0.0, 1.1], [1.0, 0.0]]), [0.5, 0.5]),
        (np.asarray([[0.0, 1.0], [1.0, 0.0]]), [0.6, 0.6]),
        (np.asarray([[0.0, 1.0], [1.0, 0.0]]), [-0.1, 1.1]),
    ],
)
def test_structural_risk_rejects_invalid_inputs(
    costs: np.ndarray,
    probabilities: list[float],
) -> None:
    with pytest.raises(ValueError):
        old_policy_structural_risk(costs, probabilities)

