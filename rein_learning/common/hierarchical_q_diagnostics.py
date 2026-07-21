from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .q_critic_diagnostics import (
    pairwise_ranking_accuracy,
    regression_metrics,
    top_action_accuracy,
)
from .q_critic_training import action_group_ids


def build_hierarchical_q_data(
    dataset: dict[str, np.ndarray], indices: Sequence[int], *, noop_action: int
) -> dict[str, np.ndarray]:
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("indices must be non-empty and one-dimensional")
    groups = action_group_ids(
        dataset["state_ids"][selected], dataset["unit_indices"][selected]
    )
    context_indices: list[int] = []
    group_names: list[str] = []
    engagement_labels: list[np.ndarray] = []
    engagement_samples: list[np.ndarray] = []
    scenarios: list[str] = []
    source_seeds: list[int] = []
    target_indices: list[int] = []
    target_groups: list[str] = []
    target_scenarios: list[str] = []

    for group in np.unique(groups):
        rows = selected[np.flatnonzero(groups == group)]
        actions = dataset["candidate_actions"][rows]
        noop_rows = rows[actions == noop_action]
        targets = rows[actions != noop_action]
        if len(noop_rows) != 1:
            raise ValueError("Every hierarchy group needs exactly one no-op")
        if len(targets) == 0:
            continue
        probabilities = dataset["conditional_target_probabilities"][targets].astype(
            np.float64
        )
        probability_sum = float(np.sum(probabilities))
        if probability_sum <= 0.0:
            raise ValueError("Conditional target probabilities must have positive mass")
        probabilities /= probability_sum
        if not np.isclose(np.sum(probabilities), 1.0, atol=1e-6):
            raise ValueError("Conditional target probabilities must sum to one")

        noop_row = int(noop_rows[0])
        target_returns = dataset["return_samples"][targets].astype(np.float64)
        noop_returns = dataset["return_samples"][noop_row].astype(np.float64)
        valid_columns = np.isfinite(noop_returns) & np.all(
            np.isfinite(target_returns), axis=0
        )
        if int(np.sum(valid_columns)) < 2:
            raise ValueError("Hierarchy groups need at least two paired rollouts")
        weighted_returns = np.full(noop_returns.shape, np.nan, dtype=np.float64)
        weighted_returns[valid_columns] = np.sum(
            probabilities[:, None] * target_returns[:, valid_columns], axis=0
        )
        context_indices.append(noop_row)
        group_names.append(str(group))
        engagement_labels.append(
            np.asarray(
                (
                    dataset["q_labels"][noop_row],
                    float(np.sum(probabilities * dataset["q_labels"][targets])),
                ),
                dtype=np.float32,
            )
        )
        engagement_samples.append(
            np.stack((noop_returns, weighted_returns)).astype(np.float32)
        )
        scenarios.append(str(dataset["scenarios"][noop_row]))
        source_seeds.append(int(dataset["source_seeds"][noop_row]))
        target_indices.extend(targets.tolist())
        target_groups.extend([str(group)] * len(targets))
        target_scenarios.extend([str(dataset["scenarios"][noop_row])] * len(targets))

    if not context_indices:
        raise ValueError("No actionable hierarchy groups were found")
    return {
        "context_indices": np.asarray(context_indices, dtype=np.int64),
        "group_ids": np.asarray(group_names),
        "engagement_labels": np.stack(engagement_labels),
        "engagement_return_samples": np.stack(engagement_samples),
        "scenarios": np.asarray(scenarios),
        "source_seeds": np.asarray(source_seeds, dtype=np.int64),
        "target_indices": np.asarray(target_indices, dtype=np.int64),
        "target_group_ids": np.asarray(target_groups),
        "target_scenarios": np.asarray(target_scenarios),
    }


def engagement_sign_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    return_samples: np.ndarray,
    *,
    uncertainty_z: float = 1.96,
) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    samples = np.asarray(return_samples, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[1] != 2 or truth.shape != estimate.shape:
        raise ValueError("Engagement labels and predictions must have shape [groups, 2]")
    if samples.ndim != 3 or samples.shape[:2] != truth.shape:
        raise ValueError("Engagement return samples must have shape [groups, 2, rollouts]")
    correct = 0
    count = 0
    for index in range(len(truth)):
        paired = samples[index, 1] - samples[index, 0]
        paired = paired[np.isfinite(paired)]
        if len(paired) < 2:
            continue
        standard_error = float(np.std(paired, ddof=1) / np.sqrt(len(paired)))
        difference = float(truth[index, 1] - truth[index, 0])
        if abs(difference) <= uncertainty_z * standard_error:
            continue
        predicted_difference = float(estimate[index, 1] - estimate[index, 0])
        correct += int(
            predicted_difference != 0.0
            and np.sign(predicted_difference) == np.sign(difference)
        )
        count += 1
    return {"count": count, "accuracy": float(correct / count) if count else float("nan")}


def hierarchical_q_metrics(
    *,
    hierarchy: dict[str, np.ndarray],
    dataset: dict[str, np.ndarray],
    engagement_predictions: np.ndarray,
    target_predictions: np.ndarray,
) -> dict[str, Any]:
    engagement_labels = hierarchy["engagement_labels"]
    engagement_regression = regression_metrics(
        engagement_labels.reshape(-1), engagement_predictions.reshape(-1)
    )
    engagement_sign = engagement_sign_metrics(
        engagement_labels,
        engagement_predictions,
        hierarchy["engagement_return_samples"],
    )
    target_indices = hierarchy["target_indices"]
    target_labels = dataset["q_labels"][target_indices]
    target_samples = dataset["return_samples"][target_indices]
    target_regression = regression_metrics(target_labels, target_predictions)
    target_ranking = pairwise_ranking_accuracy(
        target_labels,
        target_predictions,
        hierarchy["target_group_ids"],
        return_samples=target_samples,
    )
    target_top = top_action_accuracy(
        target_labels,
        target_predictions,
        hierarchy["target_group_ids"],
        return_samples=target_samples,
    )
    values: dict[str, Any] = {
        "engagement_mae": engagement_regression["mae"],
        "engagement_rmse": engagement_regression["rmse"],
        "engagement_sign_accuracy": engagement_sign["accuracy"],
        "engagement_sign_count": engagement_sign["count"],
        "target_mae": target_regression["mae"],
        "target_rmse": target_regression["rmse"],
        "target_ranking_accuracy": target_ranking["accuracy"],
        "target_ranking_count": target_ranking["count"],
        "target_top_accuracy": target_top["accuracy"],
        "target_top_count": target_top["count"],
    }
    for scenario in np.unique(hierarchy["scenarios"]):
        engage_selected = hierarchy["scenarios"] == scenario
        scenario_engage = engagement_sign_metrics(
            engagement_labels[engage_selected],
            engagement_predictions[engage_selected],
            hierarchy["engagement_return_samples"][engage_selected],
        )
        target_selected = hierarchy["target_scenarios"] == scenario
        scenario_target = pairwise_ranking_accuracy(
            target_labels[target_selected],
            target_predictions[target_selected],
            hierarchy["target_group_ids"][target_selected],
            return_samples=target_samples[target_selected],
        )
        values[f"scenario_{scenario}_engagement_accuracy"] = scenario_engage["accuracy"]
        values[f"scenario_{scenario}_engagement_count"] = scenario_engage["count"]
        values[f"scenario_{scenario}_target_accuracy"] = scenario_target["accuracy"]
        values[f"scenario_{scenario}_target_count"] = scenario_target["count"]
    return values
