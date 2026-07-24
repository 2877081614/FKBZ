import numpy as np

from rein_learning.common import (
    ParetoRecallConstraints,
    audit_pareto_thresholds,
    complete_threshold_candidates,
    pareto_frontier_mask,
    threshold_operating_point,
)


def test_complete_threshold_candidates_cover_all_binary_splits() -> None:
    scores = np.asarray([-2.0, -1.0, 1.0, 3.0])
    thresholds = complete_threshold_candidates(scores)

    predictions = {
        tuple((scores > threshold).astype(int)) for threshold in thresholds
    }

    assert len(thresholds) == 5
    assert predictions == {
        (1, 1, 1, 1),
        (0, 1, 1, 1),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 0),
    }


def test_threshold_operating_point_enforces_batch_and_scenario_recalls() -> None:
    scores = np.asarray([-2.0, 2.0, -1.0, 1.0, -3.0, 3.0])
    labels = np.asarray([0, 1, 0, 1, 0, 1])
    batches = np.asarray(["a", "a", "b", "b", "c", "c"])
    scenarios = np.asarray(["x", "x", "y", "y", "z", "z"])

    point = threshold_operating_point(
        scores,
        labels,
        batches,
        scenarios,
        0.0,
        safety_sign_accuracy=0.8,
    )

    assert point["feasible"] is True
    assert point["worst_batch_engage_recall"] == 1.0
    assert point["worst_scenario_noop_recall"] == 1.0

    failed_point = threshold_operating_point(
        np.asarray([-2.0, 2.0, -1.0, -0.5, -3.0, 3.0]),
        labels,
        batches,
        scenarios,
        0.0,
        safety_sign_accuracy=0.8,
    )

    assert failed_point["feasible"] is False
    assert failed_point["worst_batch_engage_recall"] == 0.0
    assert failed_point["checks"]["batch_engage_recall"] is False


def test_pareto_audit_reports_no_boundary_for_conflicting_batches() -> None:
    scores = np.asarray([0.8, 0.9, -0.9, -0.8, 0.7, 0.6])
    labels = np.asarray([0, 1, 0, 1, 0, 1])
    batches = np.asarray(["a", "a", "b", "b", "c", "c"])
    scenarios = np.asarray(["x", "x", "y", "y", "z", "z"])
    constraints = ParetoRecallConstraints(
        engage_recall=1.0,
        noop_recall=1.0,
        batch_engage_recall=1.0,
        batch_noop_recall=1.0,
        scenario_engage_recall=1.0,
        scenario_noop_recall=1.0,
    )

    _, summary = audit_pareto_thresholds(
        scores,
        labels,
        batches,
        scenarios,
        safety_sign_accuracy=0.8,
        constraints=constraints,
    )

    assert summary["has_feasible_threshold"] is False
    assert summary["selected"]["minimum_constraint_margin"] < 0.0


def test_pareto_frontier_excludes_dominated_operating_point() -> None:
    rows = [
        {"worst_scenario_engage_recall": 0.8, "worst_scenario_noop_recall": 0.7},
        {"worst_scenario_engage_recall": 0.7, "worst_scenario_noop_recall": 0.6},
        {"worst_scenario_engage_recall": 0.6, "worst_scenario_noop_recall": 0.9},
    ]

    mask = pareto_frontier_mask(rows)

    assert mask.tolist() == [True, False, True]
