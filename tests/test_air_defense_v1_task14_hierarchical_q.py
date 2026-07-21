import numpy as np
import pytest
import torch

from rein_learning.common import (
    build_hierarchical_q_data,
    engagement_sign_metrics,
    hierarchical_q_metrics,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    HierarchicalMaskedQCritic,
)


def test_hierarchical_q_critic_outputs_separate_levels() -> None:
    env = AirDefenseResourceAssignmentEnvV1()
    observation, _ = env.reset(seed=4)
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space, env.action_space
    )
    model = HierarchicalMaskedQCritic(layout)
    legal_mask = env.action_mask()[0]
    target = int(np.flatnonzero(legal_mask[: env.num_targets])[0])
    engagement = model.forward_engagement(
        torch.as_tensor(observation).unsqueeze(0),
        torch.tensor([0]),
        torch.zeros((1, env.num_targets)),
        torch.as_tensor(legal_mask).unsqueeze(0),
    )
    target_q = model.forward_target(
        torch.as_tensor(observation).unsqueeze(0),
        torch.tensor([0]),
        torch.tensor([target]),
        torch.zeros((1, env.num_targets)),
        torch.as_tensor(legal_mask).unsqueeze(0),
    )

    assert engagement.shape == (1, 2)
    assert target_q.shape == (1,)
    assert model.signature()["type"] == "hierarchical_masked_q_critic_mlp"
    with pytest.raises(ValueError, match="no-op"):
        model.forward_target(
            torch.as_tensor(observation).unsqueeze(0),
            torch.tensor([0]),
            torch.tensor([env.noop_action]),
            torch.zeros((1, env.num_targets)),
            torch.as_tensor(legal_mask).unsqueeze(0),
        )
    env.close()


def _synthetic_action_dataset() -> dict[str, np.ndarray]:
    return {
        "state_ids": np.asarray(["s", "s", "s"]),
        "unit_indices": np.asarray([0, 0, 0]),
        "candidate_actions": np.asarray([0, 1, 2]),
        "conditional_target_probabilities": np.asarray([0.25, 0.75, 0.0]),
        "q_labels": np.asarray([2.0, 6.0, 1.0]),
        "return_samples": np.asarray(
            [
                [1.0, 3.0, 1.0, 3.0],
                [5.0, 7.0, 5.0, 7.0],
                [0.0, 2.0, 0.0, 2.0],
            ]
        ),
        "scenarios": np.asarray(["medium", "medium", "medium"]),
        "source_seeds": np.asarray([8, 8, 8]),
    }


def test_hierarchical_data_builds_policy_weighted_engagement_value() -> None:
    dataset = _synthetic_action_dataset()
    hierarchy = build_hierarchical_q_data(dataset, [0, 1, 2], noop_action=2)

    assert hierarchy["context_indices"].tolist() == [2]
    np.testing.assert_allclose(hierarchy["engagement_labels"], [[1.0, 5.0]])
    assert hierarchy["engagement_return_samples"][0, 1].tolist() == pytest.approx(
        [4.0, 6.0, 4.0, 6.0]
    )
    assert hierarchy["target_indices"].tolist() == [0, 1]


def test_engagement_sign_metrics_uses_paired_uncertainty() -> None:
    labels = np.asarray([[1.0, 3.0], [4.0, 2.0]])
    predictions = np.asarray([[0.0, 2.0], [5.0, 1.0]])
    samples = np.asarray(
        [
            [[0.0, 2.0, 0.0, 2.0], [2.0, 4.0, 2.0, 4.0]],
            [[3.0, 5.0, 3.0, 5.0], [1.0, 3.0, 1.0, 3.0]],
        ]
    )

    assert engagement_sign_metrics(labels, predictions, samples) == {
        "count": 2,
        "accuracy": 1.0,
    }


def test_hierarchical_metrics_report_both_levels() -> None:
    dataset = _synthetic_action_dataset()
    hierarchy = build_hierarchical_q_data(dataset, [0, 1, 2], noop_action=2)
    metrics = hierarchical_q_metrics(
        hierarchy=hierarchy,
        dataset=dataset,
        engagement_predictions=np.asarray([[1.0, 5.0]]),
        target_predictions=np.asarray([2.0, 6.0]),
    )

    assert metrics["engagement_mae"] == pytest.approx(0.0)
    assert metrics["engagement_sign_accuracy"] == pytest.approx(1.0)
    assert metrics["target_ranking_accuracy"] == pytest.approx(1.0)


def test_hierarchical_data_skips_noop_only_groups() -> None:
    dataset = _synthetic_action_dataset()
    dataset = {
        key: np.concatenate((value, value[-1:]))
        for key, value in dataset.items()
    }
    dataset["state_ids"][-1] = "noop-only"
    dataset["candidate_actions"][-1] = 2

    hierarchy = build_hierarchical_q_data(dataset, [0, 1, 2, 3], noop_action=2)

    assert hierarchy["group_ids"].tolist() == ["s/unit0"]
