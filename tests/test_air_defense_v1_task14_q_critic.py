import numpy as np
import pytest
import torch

from rein_learning.common import (
    engagement_sign_accuracy,
    grouped_state_split,
    pairwise_ranking_accuracy,
    regression_metrics,
    top_action_accuracy,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    FactorizedEngagementAutoregressiveDistribution,
    MaskedActionQCritic,
    MaskedActionQCriticConfig,
)


def test_masked_action_q_critic_forward_and_ablation_signatures() -> None:
    env = AirDefenseResourceAssignmentEnvV1()
    observation, _ = env.reset(seed=1)
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space, env.action_space
    )
    legal_mask = env.action_mask()[0]
    candidate = int(np.flatnonzero(legal_mask)[0])
    tensors = (
        torch.as_tensor(observation).unsqueeze(0),
        torch.tensor([0]),
        torch.tensor([candidate]),
        torch.zeros((1, env.num_targets)),
        torch.as_tensor(legal_mask).unsqueeze(0),
    )

    full = MaskedActionQCritic(layout)
    ablated = MaskedActionQCritic(
        layout,
        MaskedActionQCriticConfig(
            include_entity_features=False,
            include_prefix_occupancy=False,
            include_legal_mask=False,
        ),
    )

    assert full(*tensors).shape == (1,)
    assert ablated(*tensors).shape == (1,)
    assert full.parameter_count() > ablated.parameter_count()
    assert full.signature()["type"] == "masked_action_q_critic_mlp"
    env.close()


def test_masked_action_q_critic_rejects_illegal_candidate() -> None:
    env = AirDefenseResourceAssignmentEnvV1()
    observation, _ = env.reset(seed=2)
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space, env.action_space
    )
    model = MaskedActionQCritic(layout)
    legal_mask = torch.zeros((1, env.num_unit_actions))
    legal_mask[0, env.noop_action] = 1.0

    with pytest.raises(ValueError, match="legal"):
        model(
            torch.as_tensor(observation).unsqueeze(0),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.zeros((1, env.num_targets)),
            legal_mask,
        )
    env.close()


def test_factorized_fixed_prefix_resamples_suffix_without_duplicate() -> None:
    distribution = FactorizedEngagementAutoregressiveDistribution(
        torch.zeros((1, 3, 3)),
        torch.full((1, 3), 20.0),
        (4, 4, 4),
        torch.ones((1, 3, 4), dtype=torch.bool),
    )

    torch.manual_seed(3)
    evaluation = distribution.sample_with_fixed_actions(
        torch.tensor([[1, -1, -1]])
    )

    assert evaluation.actions[0, 0].item() == 1
    selected_targets = [
        action
        for action in evaluation.actions[0].tolist()
        if action != distribution.noop_action
    ]
    assert len(selected_targets) == len(set(selected_targets))
    with pytest.raises(ValueError, match="illegal"):
        distribution.sample_with_fixed_actions(torch.tensor([[1, 1, -1]]))


def test_grouped_state_split_has_no_leakage_and_preserves_strata() -> None:
    state_ids = np.repeat([f"state-{index}" for index in range(12)], 3)
    strata = np.repeat(
        ["a"] * 6 + ["b"] * 6,
        3,
    )

    split = grouped_state_split(state_ids, strata=strata, seed=14)

    for state_id in np.unique(state_ids):
        assert len(np.unique(split[state_ids == state_id])) == 1
    for stratum in ("a", "b"):
        selected = split[strata == stratum]
        assert set(selected.tolist()) == {"train", "validation", "test"}


def test_q_critic_metrics_handle_ranking_and_uncertainty() -> None:
    labels = [0.0, 2.0, 1.0, 4.0]
    predictions = [0.1, 1.8, 1.2, 3.5]
    groups = ["a", "a", "b", "b"]

    regression = regression_metrics(labels, predictions)
    ranking = pairwise_ranking_accuracy(labels, predictions, groups)
    top = top_action_accuracy(labels, predictions, groups)
    uncertain = pairwise_ranking_accuracy(
        labels,
        predictions,
        groups,
        standard_errors=[2.0, 2.0, 2.0, 2.0],
    )

    assert regression["mae"] == pytest.approx(0.25)
    assert ranking == {"count": 2, "accuracy": 1.0}
    assert top == {"count": 2, "accuracy": 1.0}
    assert uncertain["count"] == 0
    assert np.isnan(uncertain["accuracy"])


def test_engagement_sign_accuracy_uses_conditional_target_mixture() -> None:
    result = engagement_sign_accuracy(
        labels=[0.0, 2.0, 4.0, 3.0, 1.0, 2.0],
        predictions=[0.2, 2.2, 3.8, 2.8, 1.2, 2.1],
        group_ids=["a", "a", "a", "b", "b", "b"],
        candidate_actions=[2, 0, 1, 2, 0, 1],
        conditional_target_probabilities=[0.0, 0.25, 0.75, 0.0, 0.5, 0.5],
        noop_action=2,
    )

    assert result == {"count": 2, "accuracy": 1.0}


def test_common_random_returns_use_paired_standard_error() -> None:
    labels = [10.0, 12.0]
    predictions = [9.0, 11.0]
    samples = np.asarray(
        [
            [-90.0, 110.0, -90.0, 110.0],
            [-88.0, 112.0, -88.0, 112.0],
        ]
    )

    independent = pairwise_ranking_accuracy(
        labels,
        predictions,
        ["state", "state"],
        standard_errors=[50.0, 50.0],
    )
    paired = pairwise_ranking_accuracy(
        labels,
        predictions,
        ["state", "state"],
        standard_errors=[50.0, 50.0],
        return_samples=samples,
    )

    assert independent["count"] == 0
    assert paired == {"count": 1, "accuracy": 1.0}
