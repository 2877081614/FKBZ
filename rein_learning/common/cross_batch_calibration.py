from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .pareto_feasibility import (
    ParetoRecallConstraints,
    threshold_operating_point,
)


@dataclass(frozen=True)
class CrossBatchCalibrationConfig:
    name: str
    feature_set: str
    confidence_z: float
    l2: float = 1e-2

    def __post_init__(self) -> None:
        if self.feature_set not in {"score_only", "value_context"}:
            raise ValueError("Unsupported calibration feature set")
        if self.confidence_z < 0.0 or self.l2 <= 0.0:
            raise ValueError("Calibration regularization must be positive")

    def signature(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FittedCrossBatchCalibrator:
    config: CrossBatchCalibrationConfig
    feature_names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    covariance: np.ndarray
    hessian_condition: float

    def __post_init__(self) -> None:
        feature_count = len(self.feature_names)
        if self.center.shape != (feature_count,):
            raise ValueError("Calibration center has the wrong shape")
        if self.scale.shape != (feature_count,) or np.any(self.scale <= 0.0):
            raise ValueError("Calibration scale has the wrong shape")
        if self.coefficients.shape != (feature_count + 1,):
            raise ValueError("Calibration coefficients have the wrong shape")
        if self.covariance.shape != (feature_count + 1, feature_count + 1):
            raise ValueError("Calibration covariance has the wrong shape")

    def predict(self, features: np.ndarray) -> dict[str, np.ndarray]:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("Prediction features have the wrong shape")
        standardized = (values - self.center) / self.scale
        design = np.column_stack((np.ones(len(values)), standardized))
        logits = design @ self.coefficients
        variances = np.einsum(
            "ij,jk,ik->i", design, self.covariance, design
        )
        standard_errors = np.sqrt(np.maximum(variances, 0.0))
        conservative = logits - self.config.confidence_z * standard_errors
        return {
            "probability": _sigmoid(logits),
            "calibrated_logit": logits,
            "prediction_se": standard_errors,
            "conservative_logit": conservative,
            "predicted_label": (conservative > 0.0).astype(np.int64),
        }

    def signature(self) -> dict[str, object]:
        return {
            "config": self.config.signature(),
            "feature_names": list(self.feature_names),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "covariance": self.covariance.tolist(),
            "hessian_condition": float(self.hessian_condition),
        }


def calibration_candidates() -> tuple[CrossBatchCalibrationConfig, ...]:
    return (
        CrossBatchCalibrationConfig("score_platt", "score_only", 0.0),
        CrossBatchCalibrationConfig("value_platt", "value_context", 0.0),
        CrossBatchCalibrationConfig("value_lcb_050", "value_context", 0.5),
        CrossBatchCalibrationConfig("value_lcb_100", "value_context", 1.0),
    )


def assemble_calibration_features(
    scores: Sequence[float],
    safety_predictions: Sequence[float],
    cost_predictions: Sequence[float],
    budget_multipliers: Sequence[float],
    scenarios: Sequence[object],
    *,
    feature_set: str,
    scenario_levels: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    score = np.asarray(scores, dtype=np.float64)
    safety = np.asarray(safety_predictions, dtype=np.float64)
    cost = np.asarray(cost_predictions, dtype=np.float64)
    budget = np.asarray(budget_multipliers, dtype=np.float64)
    strata = np.asarray(scenarios).astype(str)
    if not (
        score.shape == safety.shape == cost.shape == budget.shape == strata.shape
    ) or score.ndim != 1:
        raise ValueError("Calibration feature arrays must be aligned vectors")
    if np.any(~np.isfinite(np.column_stack((score, safety, cost, budget)))):
        raise ValueError("Calibration features must be finite")
    if feature_set == "score_only":
        return score[:, None], ("score",)
    if feature_set != "value_context":
        raise ValueError("Unsupported calibration feature set")
    levels = tuple(str(level) for level in scenario_levels)
    unknown = set(np.unique(strata)) - set(levels)
    if unknown:
        raise ValueError(f"Unknown calibration scenarios: {sorted(unknown)}")
    one_hot = np.column_stack(
        [(strata == level).astype(np.float64) for level in levels]
    )
    values = np.column_stack(
        (score, safety, np.maximum(cost, 0.0), budget, one_hot)
    )
    names = (
        "score",
        "safety_prediction",
        "positive_cost_prediction",
        "budget_multiplier",
        *(f"scenario={level}" for level in levels),
    )
    return values, names


def equal_block_weights(
    labels: Sequence[int],
    batch_ids: Sequence[object],
    scenarios: Sequence[object],
) -> np.ndarray:
    truth = np.asarray(labels, dtype=np.int64)
    batches = np.asarray(batch_ids).astype(str)
    strata = np.asarray(scenarios).astype(str)
    if not (truth.shape == batches.shape == strata.shape) or truth.ndim != 1:
        raise ValueError("Weighting arrays must be aligned vectors")
    if np.any((truth < 0) | (truth > 1)):
        raise ValueError("Block weighting requires binary labels")
    blocks = np.asarray(
        [
            f"{batch}|{scenario}|{label}"
            for batch, scenario, label in zip(batches, strata, truth)
        ]
    )
    unique = np.unique(blocks)
    weights = np.zeros(len(truth), dtype=np.float64)
    for block in unique:
        selected = blocks == block
        weights[selected] = 1.0 / (len(unique) * int(np.sum(selected)))
    return weights * len(weights)


def fit_cross_batch_calibrator(
    features: np.ndarray,
    feature_names: Sequence[str],
    labels: Sequence[int],
    batch_ids: Sequence[object],
    scenarios: Sequence[object],
    config: CrossBatchCalibrationConfig,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> FittedCrossBatchCalibrator:
    values = np.asarray(features, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != len(truth):
        raise ValueError("Features and labels must align")
    if values.shape[1] != len(tuple(feature_names)):
        raise ValueError("Feature names must align with columns")
    if np.any((truth < 0) | (truth > 1)) or len(np.unique(truth)) != 2:
        raise ValueError("Calibration fitting requires both binary classes")
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = q75 - q25
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (values - center) / scale
    design = np.column_stack((np.ones(len(values)), standardized))
    weights = equal_block_weights(truth, batch_ids, scenarios)
    penalty = np.diag([0.0, *([config.l2] * values.shape[1])])
    coefficients = np.zeros(values.shape[1] + 1, dtype=np.float64)
    for _ in range(max_iterations):
        probability = _sigmoid(design @ coefficients)
        curvature = weights * probability * (1.0 - probability)
        hessian = design.T @ (design * curvature[:, None]) + penalty
        gradient = design.T @ (weights * (probability - truth))
        gradient += penalty @ coefficients
        step = np.linalg.pinv(hessian, rcond=1e-10) @ gradient
        coefficients -= step
        if float(np.max(np.abs(step))) <= tolerance:
            break
    probability = _sigmoid(design @ coefficients)
    curvature = weights * probability * (1.0 - probability)
    hessian = design.T @ (design * curvature[:, None]) + penalty
    covariance = np.linalg.pinv(hessian, rcond=1e-10)
    return FittedCrossBatchCalibrator(
        config=config,
        feature_names=tuple(feature_names),
        center=center,
        scale=scale,
        coefficients=coefficients,
        covariance=covariance,
        hessian_condition=float(np.linalg.cond(hessian)),
    )


def calibrated_operating_point(
    prediction: dict[str, np.ndarray],
    oracle_labels: Sequence[int],
    batch_ids: Sequence[object],
    scenarios: Sequence[object],
    *,
    safety_sign_accuracy: float,
    constraints: ParetoRecallConstraints | None = None,
) -> dict[str, object]:
    truth = np.asarray(oracle_labels, dtype=np.int64)
    probability = np.asarray(prediction["probability"], dtype=np.float64)
    conservative = np.asarray(
        prediction["conservative_logit"], dtype=np.float64
    )
    point = threshold_operating_point(
        conservative,
        truth,
        batch_ids,
        scenarios,
        0.0,
        safety_sign_accuracy=safety_sign_accuracy,
        constraints=constraints,
    )
    valid = truth >= 0
    clipped = np.clip(probability[valid], 1e-8, 1.0 - 1e-8)
    point["brier_score"] = float(np.mean((clipped - truth[valid]) ** 2))
    point["log_loss"] = float(
        -np.mean(
            truth[valid] * np.log(clipped)
            + (1 - truth[valid]) * np.log(1.0 - clipped)
        )
    )
    point["mean_prediction_se"] = float(
        np.mean(np.asarray(prediction["prediction_se"])[valid])
    )
    unconstrained = np.asarray(prediction["calibrated_logit"])[valid] > 0.0
    constrained = conservative[valid] > 0.0
    point["lcb_change_rate"] = float(np.mean(unconstrained != constrained))
    return point


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))
