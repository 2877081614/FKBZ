from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rein_learning.algorithms.policy_gradient import (
    BoundaryProbedCounterfactualEngagementPPO,
    FactorizedEngagementActorCriticPolicy,
    FactorizedEngagementMaskablePPO,
)
from rein_learning.common import (
    paired_direction_gate,
    select_boundary_candidates,
    select_random_candidates,
)
from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    get_air_defense_v1_scenario,
)


def _small_model(
    algorithm_class: type,
    *,
    seed: int = 3,
    **kwargs: object,
):
    env = AirDefenseResourceAssignmentEnvV1(
        config=get_air_defense_v1_scenario("time_pressure")
    )
    model = algorithm_class(
        FactorizedEngagementActorCriticPolicy,
        env,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        learning_rate=3e-4,
        seed=seed,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": [32], "unit_order": (0, 1, 2)},
        **kwargs,
    )
    return env, model


def test_state_snapshot_restores_observation_mask_and_rng() -> None:
    env = AirDefenseResourceAssignmentEnvV1(
        config=get_air_defense_v1_scenario("time_pressure")
    )
    observation, _ = env.reset(seed=19)
    snapshot = env.snapshot_state()
    mask = env.action_masks().copy()
    expected_random = float(env.np_random.random())

    env.step(np.full(env.num_defense_units, env.noop_action, dtype=np.int64))
    env.restore_state(snapshot)
    restored_observation = env._get_observation()
    restored_random = float(env.np_random.random())

    np.testing.assert_array_equal(restored_observation, observation)
    np.testing.assert_array_equal(env.action_masks(), mask)
    assert restored_random == expected_random


def test_hit_random_tape_makes_replayed_branch_exact() -> None:
    env = AirDefenseResourceAssignmentEnvV1(
        config=get_air_defense_v1_scenario("time_pressure")
    )
    env.reset(seed=23)
    snapshot = env.snapshot_state()
    tape = np.random.default_rng(7).random(
        (env.config.max_steps, env.num_targets)
    )
    action = np.full(env.num_defense_units, env.noop_action, dtype=np.int64)
    for unit_index in range(env.num_defense_units):
        legal = np.flatnonzero(env.action_mask()[unit_index, : env.num_targets])
        if legal.size:
            action[unit_index] = int(legal[0])
            break

    env.set_hit_random_tape(tape)
    first = env.step(action)
    env.restore_state(snapshot)
    env.set_hit_random_tape(tape)
    second = env.step(action)

    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:4] == second[1:4]
    assert first[4]["reward_breakdown"] == second[4]["reward_breakdown"]


def test_boundary_selection_is_ranked_and_radius_limited() -> None:
    probabilities = np.asarray([[0.49, 0.80], [0.55, 0.10]])
    actionable = np.ones_like(probabilities, dtype=bool)
    selected = select_boundary_candidates(
        probabilities,
        actionable,
        margin_radius=0.5,
        max_contexts=2,
    )
    assert [(item.rollout_step, item.unit_index) for item in selected] == [
        (0, 0),
        (1, 0),
    ]


def test_random_selection_is_seeded_and_budget_limited() -> None:
    probabilities = np.full((4, 3), 0.5)
    actionable = np.ones_like(probabilities, dtype=bool)
    first = select_random_candidates(
        probabilities,
        actionable,
        max_contexts=4,
        seed=11,
    )
    second = select_random_candidates(
        probabilities,
        actionable,
        max_contexts=4,
        seed=11,
    )
    assert first == second
    assert len(first) == 4


@pytest.mark.parametrize(
    ("deltas", "expected_direction", "expected_accepted"),
    [
        ([2.0] * 7 + [-0.1], 1, True),
        ([-2.0] * 7 + [0.1], -1, True),
        ([2.0] * 6 + [-0.1] * 2, 1, False),
        ([0.2] * 8, 1, False),
    ],
)
def test_paired_direction_gate(
    deltas: list[float],
    expected_direction: int,
    expected_accepted: bool,
) -> None:
    direction, accepted, _, _ = paired_direction_gate(
        deltas,
        minimum_sign_agreement=1,
        minimum_return_effect=1.0,
        minimum_informative_repeats=2,
        maximum_opposite_repeats=1,
    )
    assert direction == expected_direction
    assert accepted is expected_accepted


def test_sparse_gate_treats_zero_as_neutral_but_requires_two_effects() -> None:
    _, one_accepted, _, _ = paired_direction_gate(
        [3.0] + [0.0] * 7,
        minimum_sign_agreement=1,
        minimum_return_effect=0.1,
        minimum_informative_repeats=2,
        maximum_opposite_repeats=1,
    )
    _, two_accepted, _, _ = paired_direction_gate(
        [3.0, 2.0] + [0.0] * 6,
        minimum_sign_agreement=1,
        minimum_return_effect=0.1,
        minimum_informative_repeats=2,
        maximum_opposite_repeats=1,
    )
    assert one_accepted is False
    assert two_accepted is True


def test_ranking_loss_pushes_margin_in_label_direction() -> None:
    margin = torch.tensor(0.0, requires_grad=True)
    positive_loss = F.softplus(-margin)
    positive_loss.backward()
    assert margin.grad is not None and float(margin.grad) < 0.0

    margin = torch.tensor(0.0, requires_grad=True)
    negative_loss = F.softplus(margin)
    negative_loss.backward()
    assert margin.grad is not None and float(margin.grad) > 0.0


def test_zero_probe_budget_is_exact_factorized_fallback() -> None:
    _, baseline = _small_model(FactorizedEngagementMaskablePPO, seed=31)
    baseline.learn(total_timesteps=32)
    baseline_state = deepcopy(baseline.policy.state_dict())

    _, candidate = _small_model(
        BoundaryProbedCounterfactualEngagementPPO,
        seed=31,
        counterfactual_loss_coef=0.05,
        probe_max_contexts=0,
    )
    candidate.learn(total_timesteps=32)

    maximum_difference = max(
        float(torch.max(torch.abs(baseline_state[key] - value)).item())
        for key, value in candidate.policy.state_dict().items()
    )
    assert maximum_difference <= 1e-6
    assert candidate.bpce_extra_transitions == 0


def test_all_rejected_labels_are_exact_factorized_fallback() -> None:
    _, baseline = _small_model(FactorizedEngagementMaskablePPO, seed=41)
    baseline.learn(total_timesteps=32)
    baseline_state = deepcopy(baseline.policy.state_dict())

    _, candidate = _small_model(
        BoundaryProbedCounterfactualEngagementPPO,
        seed=41,
        counterfactual_loss_coef=0.05,
        probe_interval=1,
        probe_max_contexts=1,
        probe_repeats=2,
        probe_margin_radius=10.0,
        probe_minimum_sign_agreement=1,
        probe_minimum_informative_repeats=2,
        probe_maximum_opposite_repeats=0,
        probe_minimum_return_effect=1e9,
    )
    candidate.learn(total_timesteps=32)

    maximum_difference = max(
        float(torch.max(torch.abs(baseline_state[key] - value)).item())
        for key, value in candidate.policy.state_dict().items()
    )
    assert maximum_difference <= 1e-6
    assert candidate.last_bpce_probe_diagnostics["accepted_count"] == 0.0


def test_bpce_smoke_generates_paired_probe_diagnostics() -> None:
    _, model = _small_model(
        BoundaryProbedCounterfactualEngagementPPO,
        seed=37,
        counterfactual_loss_coef=0.05,
        probe_interval=1,
        probe_max_contexts=1,
        probe_repeats=2,
        probe_margin_radius=10.0,
        probe_minimum_sign_agreement=1,
        probe_minimum_return_effect=0.0,
    )
    model.learn(total_timesteps=32)
    diagnostics = model.last_bpce_probe_diagnostics
    assert diagnostics["selected_count"] == 1.0
    assert diagnostics["extra_transitions"] > 0.0
    assert model.bpce_extra_transitions == diagnostics["extra_transitions"]


def test_bpce_model_save_and_load(tmp_path) -> None:
    env, model = _small_model(
        BoundaryProbedCounterfactualEngagementPPO,
        seed=43,
        probe_max_contexts=0,
    )
    model.learn(total_timesteps=32)
    path = tmp_path / "bpce_model"
    model.save(path)

    loaded = BoundaryProbedCounterfactualEngagementPPO.load(
        path,
        env=env,
    )
    assert isinstance(loaded.policy, FactorizedEngagementActorCriticPolicy)
    assert loaded.probe_config.max_contexts == 0
