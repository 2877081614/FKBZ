import numpy as np

from rein_learning.common import (
    CrossBatchCalibrationConfig,
    assemble_calibration_features,
    calibrated_operating_point,
    equal_block_weights,
    fit_cross_batch_calibrator,
)


def test_equal_block_weights_give_each_observed_block_equal_mass() -> None:
    labels = np.asarray([0, 0, 1, 0, 1, 1])
    batches = np.asarray(["a", "a", "a", "b", "b", "b"])
    scenarios = np.asarray(["x", "x", "x", "x", "x", "x"])

    weights = equal_block_weights(labels, batches, scenarios)
    masses = []
    for batch in ("a", "b"):
        for label in (0, 1):
            selected = (batches == batch) & (labels == label)
            masses.append(float(np.sum(weights[selected])))

    assert np.allclose(masses, masses[0])
    assert np.isclose(np.sum(weights), len(labels))


def test_value_context_features_have_frozen_scenario_order() -> None:
    features, names = assemble_calibration_features(
        [0.1, 0.2],
        [1.0, 2.0],
        [-1.0, 0.5],
        [0.2, 0.3],
        ["medium", "pressure"],
        feature_set="value_context",
        scenario_levels=("medium", "pressure"),
    )

    assert names[-2:] == ("scenario=medium", "scenario=pressure")
    assert features[:, 2].tolist() == [0.0, 0.5]
    assert features[:, -2:].tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_cross_batch_calibrator_learns_separable_probability() -> None:
    score = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    batches = np.asarray(["a", "b", "c", "a", "b", "c"])
    scenarios = np.asarray(["x", "x", "x", "x", "x", "x"])
    features = score[:, None]
    config = CrossBatchCalibrationConfig("test", "score_only", 0.0)

    model = fit_cross_batch_calibrator(
        features, ("score",), labels, batches, scenarios, config
    )
    prediction = model.predict(features)
    point = calibrated_operating_point(
        prediction,
        labels,
        batches,
        scenarios,
        safety_sign_accuracy=0.8,
    )

    assert prediction["probability"][0] < prediction["probability"][-1]
    assert point["feasible"] is True


def test_confidence_bound_never_increases_engagements() -> None:
    score = np.asarray([-2.0, -1.0, 1.0, 2.0])[:, None]
    labels = np.asarray([0, 0, 1, 1])
    batches = np.asarray(["a", "b", "a", "b"])
    scenarios = np.asarray(["x", "x", "x", "x"])
    base = fit_cross_batch_calibrator(
        score,
        ("score",),
        labels,
        batches,
        scenarios,
        CrossBatchCalibrationConfig("base", "score_only", 0.0),
    )
    conservative = fit_cross_batch_calibrator(
        score,
        ("score",),
        labels,
        batches,
        scenarios,
        CrossBatchCalibrationConfig("lcb", "score_only", 1.0),
    )

    base_count = int(np.sum(base.predict(score)["predicted_label"]))
    conservative_count = int(
        np.sum(conservative.predict(score)["predicted_label"])
    )

    assert conservative_count <= base_count
