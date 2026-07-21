import numpy as np
import pytest
import torch

from rein_learning.common import (
    binary_calibration_metrics,
    engagement_threshold_grid,
    hierarchical_counterfactual_advantages,
    one_step_td_error,
)
from rein_learning.models import (
    AutoregressiveMaskedMultiCategorical,
    FactorizedEngagementAutoregressiveDistribution,
)


def test_threshold_grid_is_inclusive_and_stable() -> None:
    thresholds = engagement_threshold_grid()

    assert len(thresholds) == 17
    assert thresholds[0] == pytest.approx(0.10)
    assert thresholds[-1] == pytest.approx(0.90)
    assert np.allclose(np.diff(thresholds), 0.05)


def test_binary_calibration_metrics_include_empty_bins() -> None:
    metrics = binary_calibration_metrics([0.0, 0.2, 0.8, 1.0], [0, 0, 1, 1])

    assert metrics["brier_score"] == pytest.approx(0.02)
    assert metrics["ece"] == pytest.approx(0.10)
    assert len(metrics["bins"]) == 10
    assert sum(row["count"] for row in metrics["bins"]) == 4


def test_hierarchical_counterfactual_advantages_are_centered_and_additive() -> None:
    result = hierarchical_counterfactual_advantages(
        q_noop=np.array([[0.0]]),
        q_targets=np.array([[[2.0, 4.0]]]),
        engage_probabilities=np.array([[0.6]]),
        target_probabilities=np.array([[[0.25, 0.75]]]),
        legal_target_mask=np.array([[[True, True]]]),
        selected_actions=np.array([[1]]),
    )

    assert result["q_engage"].item() == pytest.approx(3.5)
    assert result["counterfactual_baseline"].item() == pytest.approx(2.1)
    engagement_expectation = (
        0.6 * result["engagement_advantage_engage"].item()
        + 0.4 * result["engagement_advantage_noop"].item()
    )
    target_expectation = np.sum(
        np.array([0.25, 0.75]) * result["target_advantages"][0, 0]
    )
    assert engagement_expectation == pytest.approx(0.0)
    assert target_expectation == pytest.approx(0.0)
    assert result["selected_total_advantage"].item() == pytest.approx(1.9)
    assert result["selected_total_advantage"].item() == pytest.approx(
        4.0 - result["counterfactual_baseline"].item()
    )


def test_hierarchical_counterfactual_advantages_renormalize_legal_targets() -> None:
    result = hierarchical_counterfactual_advantages(
        q_noop=np.array([0.0]),
        q_targets=np.array([[2.0, 100.0, 4.0]]),
        engage_probabilities=np.array([0.5]),
        target_probabilities=np.array([[0.2, 0.7, 0.1]]),
        legal_target_mask=np.array([[True, False, True]]),
    )

    assert result["normalized_target_probabilities"][0].tolist() == pytest.approx(
        [2.0 / 3.0, 0.0, 1.0 / 3.0]
    )
    assert result["q_engage"].item() == pytest.approx(8.0 / 3.0)
    assert np.isnan(result["target_advantages"][0, 1])


def test_one_step_td_error_disables_terminal_bootstrap() -> None:
    residual = one_step_td_error(
        rewards=np.array([1.0, 1.0]),
        values=np.array([0.5, 0.5]),
        next_values=np.array([2.0, 2.0]),
        terminated=np.array([False, True]),
        gamma=0.9,
    )

    assert residual.tolist() == pytest.approx([2.3, 0.5])


def test_threshold_sampling_and_hierarchical_log_probs_preserve_prefix_mask() -> None:
    distribution = FactorizedEngagementAutoregressiveDistribution(
        torch.zeros((1, 2, 2)),
        torch.tensor([[np.log(1.5), np.log(1.5)]]),
        (3, 3),
        torch.ones((1, 2, 3), dtype=torch.bool),
    )

    low_threshold = distribution.sample_with_engagement_threshold(0.5)
    high_threshold = distribution.sample_with_engagement_threshold(0.7)
    diagnostics = distribution.hierarchical_diagnostics(low_threshold.actions)

    assert low_threshold.actions.tolist() == [[0, 1]]
    assert high_threshold.actions.tolist() == [[2, 2]]
    assert diagnostics["selected_engage"].tolist() == [[1.0, 1.0]]
    reconstructed = (
        diagnostics["engagement_log_prob"] + diagnostics["target_log_prob"]
    ).sum(dim=1)
    assert torch.allclose(reconstructed, low_threshold.log_prob)


def test_threshold_sampling_rejects_out_of_range_threshold() -> None:
    distribution = FactorizedEngagementAutoregressiveDistribution(
        torch.zeros((1, 1, 1)),
        torch.zeros((1, 1)),
        (2,),
        torch.ones((1, 1, 2), dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="threshold"):
        distribution.sample_with_engagement_threshold(1.1)


def test_categorical_threshold_sampling_uses_aggregate_engagement_mass() -> None:
    distribution = AutoregressiveMaskedMultiCategorical(
        torch.tensor([[0.0, 0.0, np.log(4.0) / 2.0]]),
        (3,),
        torch.ones((1, 3), dtype=torch.bool),
    )

    low_threshold = distribution.sample_with_engagement_threshold(0.5)
    high_threshold = distribution.sample_with_engagement_threshold(0.7)
    diagnostics = distribution.diagnostics(actions=low_threshold.actions)

    assert low_threshold.actions.item() == 0
    assert high_threshold.actions.item() == 2
    reconstructed = (
        diagnostics["engagement_log_prob"] + diagnostics["target_log_prob"]
    ).sum(dim=1)
    assert torch.allclose(reconstructed, low_threshold.log_prob)
