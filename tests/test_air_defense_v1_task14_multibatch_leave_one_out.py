from __future__ import annotations

import numpy as np
import pytest

from rein_learning.common import (
    batch_scenario_groups,
    grouped_oracle_metrics,
    leave_one_batch_out_folds,
    minimum_group_class_recall,
)


def test_leave_one_batch_out_assigns_whole_batches_to_folds() -> None:
    batches = np.asarray(["a", "a", "b", "b", "c", "c"])
    folds = leave_one_batch_out_folds(batches)
    assert len(np.unique(folds)) == 3
    for batch in np.unique(batches):
        assert len(np.unique(folds[batches == batch])) == 1


def test_leave_one_batch_out_requires_three_independent_batches() -> None:
    with pytest.raises(ValueError, match="at least three batches"):
        leave_one_batch_out_folds(["a", "a", "b"])


def test_batch_scenario_group_keeps_both_axes() -> None:
    groups = batch_scenario_groups(
        ["a", "a", "b"], ["medium", "time", "medium"]
    )
    assert groups.tolist() == ["a/medium", "a/time", "b/medium"]


def test_grouped_metrics_expose_worst_batch_class_recall() -> None:
    truth = np.asarray([1, 0, 1, 0, 1, 0])
    predicted = np.asarray([1, 0, 1, 1, 0, 0])
    metrics = grouped_oracle_metrics(
        truth, predicted, np.asarray(["a", "a", "b", "b", "c", "c"])
    )
    assert metrics["a"]["engage_recall"] == 1.0
    assert metrics["b"]["noop_recall"] == 0.0
    assert minimum_group_class_recall(metrics) == 0.0
