import numpy as np
import pytest

from rein_learning.common import (
    confirmation_power,
    evaluate_frozen_threshold,
    frozen_thresholds_from_rows,
)


def test_frozen_thresholds_require_every_seed_once() -> None:
    rows = [
        {
            "objective": "frozen",
            "model_seed": seed,
            "has_feasible_threshold": "True",
            "selected_threshold": value,
        }
        for seed, value in ((20, 0.1), (21, 0.2), (22, 0.3))
    ]

    thresholds = frozen_thresholds_from_rows(
        rows, objective="frozen", model_seeds=(20, 21, 22)
    )

    assert thresholds == {20: 0.1, 21: 0.2, 22: 0.3}
    with pytest.raises(ValueError, match="Missing frozen thresholds"):
        frozen_thresholds_from_rows(
            rows[:-1], objective="frozen", model_seeds=(20, 21, 22)
        )


def test_confirmation_power_requires_both_classes_in_every_scenario() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 0])
    scenarios = np.asarray(["a", "a", "b", "b", "c", "c"])

    power = confirmation_power(
        labels,
        scenarios,
        minimum_valid=6,
        minimum_per_class=2,
        minimum_per_scenario=2,
    )

    assert power["passed"] is False
    assert power["checks"]["scenario_classes_complete"] is False


def test_frozen_threshold_evaluation_does_not_recalibrate() -> None:
    scores = np.asarray([-2.0, 2.0, -1.0, 1.0, -3.0, 0.2])
    labels = np.asarray([0, 1, 0, 1, 0, 1])
    scenarios = np.asarray(["a", "a", "b", "b", "c", "c"])

    passing = evaluate_frozen_threshold(
        scores,
        labels,
        scenarios,
        threshold=0.0,
        safety_sign_accuracy=0.8,
    )
    failing = evaluate_frozen_threshold(
        scores,
        labels,
        scenarios,
        threshold=0.5,
        safety_sign_accuracy=0.8,
    )

    assert passing["feasible"] is True
    assert failing["feasible"] is False
    assert failing["threshold"] == 0.5
    assert failing["worst_scenario_engage_recall"] == 0.0
