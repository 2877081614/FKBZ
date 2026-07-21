from __future__ import annotations

from typing import Sequence

import numpy as np

from .engagement_utility_diagnostics import oracle_classification_metrics


def leave_one_batch_out_folds(batch_ids: Sequence[object]) -> np.ndarray:
    batches = np.asarray(batch_ids)
    if batches.ndim != 1 or len(batches) == 0:
        raise ValueError("batch_ids must be a non-empty vector")
    unique = np.unique(batches)
    if len(unique) < 3:
        raise ValueError("Leave-one-batch-out requires at least three batches")
    mapping = {batch: index for index, batch in enumerate(unique.tolist())}
    return np.asarray([mapping[batch] for batch in batches], dtype=np.int64)


def batch_scenario_groups(
    batch_ids: Sequence[object], scenarios: Sequence[object]
) -> np.ndarray:
    batches = np.asarray(batch_ids)
    strata = np.asarray(scenarios)
    if batches.ndim != 1 or batches.shape != strata.shape:
        raise ValueError("batch_ids and scenarios must be aligned vectors")
    return np.asarray(
        [f"{batch}/{scenario}" for batch, scenario in zip(batches, strata)]
    )


def grouped_oracle_metrics(
    oracle_labels: np.ndarray,
    predicted_labels: np.ndarray,
    groups: Sequence[object],
) -> dict[str, dict[str, float | int]]:
    truth = np.asarray(oracle_labels, dtype=np.int64)
    predicted = np.asarray(predicted_labels, dtype=np.int64)
    group_values = np.asarray(groups)
    if truth.shape != predicted.shape or truth.shape != group_values.shape:
        raise ValueError("Labels, predictions, and groups must align")
    return {
        str(group): oracle_classification_metrics(
            truth[group_values == group], predicted[group_values == group]
        )
        for group in np.unique(group_values)
    }


def minimum_group_class_recall(
    metrics: dict[str, dict[str, float | int]],
) -> float:
    recalls = [
        float(row[name])
        for row in metrics.values()
        for name, count_name in (
            ("engage_recall", "engage_count"),
            ("noop_recall", "noop_count"),
        )
        if int(row[count_name]) > 0
    ]
    return min(recalls) if recalls else float("nan")
