from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .engagement_utility_diagnostics import oracle_classification_metrics


@dataclass(frozen=True)
class EngagementBoundaryConstraints:
    balanced_accuracy: float = 0.70
    engage_recall: float = 0.60
    noop_recall: float = 0.65
    scenario_engage_recall: float = 0.60
    scenario_noop_recall: float = 0.65

    def signature(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class EngagementBoundaryConfig:
    threshold: float
    dual_weight: float
    logit_scale: float
    feasible: bool

    @property
    def family(self) -> str:
        return "global_threshold" if self.dual_weight == 0.0 else "resource_dual"

    def signature(self) -> dict[str, float | bool | str]:
        return {
            "family": self.family,
            "threshold": float(self.threshold),
            "dual_weight": float(self.dual_weight),
            "logit_scale": float(self.logit_scale),
            "feasible": bool(self.feasible),
        }


def resource_pressure_from_observations(
    observations: np.ndarray,
    unit_indices: np.ndarray,
    *,
    num_zones: int,
    num_targets: int,
    num_units: int,
    zone_feature_dim: int = 7,
    target_feature_dim: int = 15,
    unit_feature_dim: int = 15,
    global_feature_dim: int = 8,
) -> np.ndarray:
    values = np.asarray(observations, dtype=np.float64)
    units = np.asarray(unit_indices, dtype=np.int64)
    if values.ndim != 2 or units.shape != (len(values),):
        raise ValueError("Observations and unit indices must be aligned batches")
    if np.any((units < 0) | (units >= num_units)):
        raise ValueError("unit_indices contain an invalid unit")
    unit_start = num_zones * zone_feature_dim + num_targets * target_feature_dim
    unit_end = unit_start + num_units * unit_feature_dim
    expected_dim = unit_end + global_feature_dim
    if values.shape[1] != expected_dim:
        raise ValueError("Observations are incompatible with the supplied layout")
    structured = values[:, unit_start:unit_end].reshape(
        len(values), num_units, unit_feature_dim
    )
    selected = structured[np.arange(len(values)), units]
    ammo_fraction = np.clip(selected[:, 3], 0.0, 1.0)
    unit_cost_norm = np.maximum(selected[:, 10], 0.0)
    return (unit_cost_norm * (2.0 - ammo_fraction)).astype(np.float32)


def apply_engagement_boundary(
    logits: np.ndarray,
    config: EngagementBoundaryConfig,
    resource_pressure: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("logits must be a vector")
    if resource_pressure is None:
        pressure = np.zeros_like(values)
    else:
        pressure = np.asarray(resource_pressure, dtype=np.float64)
        if pressure.shape != values.shape:
            raise ValueError("resource_pressure must match logits")
    boundary = (
        config.threshold
        + config.dual_weight * config.logit_scale * pressure
    )
    return (values > boundary).astype(np.int64)


def scenario_classification_metrics(
    oracle_labels: np.ndarray,
    predicted_labels: np.ndarray,
    scenarios: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    truth = np.asarray(oracle_labels, dtype=np.int64)
    predicted = np.asarray(predicted_labels, dtype=np.int64)
    strata = np.asarray(scenarios)
    if truth.shape != predicted.shape or truth.shape != strata.shape:
        raise ValueError("Labels, predictions, and scenarios must be aligned")
    return {
        str(scenario): oracle_classification_metrics(
            truth[strata == scenario], predicted[strata == scenario]
        )
        for scenario in np.unique(strata)
    }


def _constraint_checks(
    metrics: dict[str, float | int],
    scenario_metrics: dict[str, dict[str, float | int]],
    constraints: EngagementBoundaryConstraints,
) -> dict[str, bool]:
    return {
        "balanced_accuracy": float(metrics["balanced_accuracy"])
        >= constraints.balanced_accuracy,
        "engage_recall": float(metrics["engage_recall"])
        >= constraints.engage_recall,
        "noop_recall": float(metrics["noop_recall"]) >= constraints.noop_recall,
        "scenario_engage_recall": all(
            float(row["engage_recall"]) >= constraints.scenario_engage_recall
            for row in scenario_metrics.values()
            if int(row["engage_count"]) > 0
        ),
        "scenario_noop_recall": all(
            float(row["noop_recall"]) >= constraints.scenario_noop_recall
            for row in scenario_metrics.values()
            if int(row["noop_count"]) > 0
        ),
    }


def _threshold_candidates(values: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(values, dtype=np.float64))
    if len(unique) == 0:
        raise ValueError("At least one calibration logit is required")
    span = max(float(np.ptp(unique)), 1.0)
    edge = span * 1e-6
    if len(unique) == 1:
        return np.asarray([unique[0] - edge, unique[0] + edge])
    middle = (unique[:-1] + unique[1:]) / 2.0
    return np.concatenate(([unique[0] - edge], middle, [unique[-1] + edge]))


def calibrate_engagement_boundary(
    logits: np.ndarray,
    oracle_labels: np.ndarray,
    scenarios: np.ndarray,
    resource_pressure: np.ndarray,
    *,
    dual_weights: Iterable[float],
    constraints: EngagementBoundaryConstraints | None = None,
) -> tuple[EngagementBoundaryConfig, list[dict[str, object]]]:
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(oracle_labels, dtype=np.int64)
    strata = np.asarray(scenarios)
    pressure = np.asarray(resource_pressure, dtype=np.float64)
    if not (values.shape == truth.shape == strata.shape == pressure.shape):
        raise ValueError("Calibration arrays must be aligned vectors")
    valid = truth >= 0
    if np.sum(truth[valid] == 0) == 0 or np.sum(truth[valid] == 1) == 0:
        raise ValueError("Calibration requires both reliable oracle classes")
    values = values[valid]
    truth = truth[valid]
    strata = strata[valid]
    pressure = pressure[valid]
    frozen_constraints = constraints or EngagementBoundaryConstraints()
    logit_scale = max(float(np.std(values)), 1e-6)
    rows: list[dict[str, object]] = []
    ranked: list[tuple[tuple[float, ...], EngagementBoundaryConfig]] = []
    weights = tuple(float(weight) for weight in dual_weights)
    if not weights or any(weight < 0.0 for weight in weights):
        raise ValueError("dual_weights must contain non-negative values")

    for dual_weight in weights:
        adjusted = values - dual_weight * logit_scale * pressure
        for threshold in _threshold_candidates(adjusted):
            config = EngagementBoundaryConfig(
                threshold=float(threshold),
                dual_weight=dual_weight,
                logit_scale=logit_scale,
                feasible=False,
            )
            predicted = apply_engagement_boundary(values, config, pressure)
            metrics = oracle_classification_metrics(truth, predicted)
            by_scenario = scenario_classification_metrics(
                truth, predicted, strata
            )
            checks = _constraint_checks(metrics, by_scenario, frozen_constraints)
            feasible = all(checks.values())
            passed_checks = sum(checks.values())
            minimum_scenario_recall = min(
                [
                    float(metric[name])
                    for metric in by_scenario.values()
                    for name, count_name in (
                        ("engage_recall", "engage_count"),
                        ("noop_recall", "noop_count"),
                    )
                    if int(metric[count_name]) > 0
                ],
                default=-1.0,
            )
            selected_config = EngagementBoundaryConfig(
                threshold=float(threshold),
                dual_weight=dual_weight,
                logit_scale=logit_scale,
                feasible=feasible,
            )
            score = (
                float(feasible),
                float(passed_checks),
                float(metrics["balanced_accuracy"]),
                minimum_scenario_recall,
                -float(metrics["wasteful_engage_rate"]),
                -dual_weight,
                -abs(float(threshold)),
            )
            ranked.append((score, selected_config))
            rows.append(
                {
                    **selected_config.signature(),
                    **metrics,
                    **{f"check_{key}": value for key, value in checks.items()},
                    "passed_constraint_count": passed_checks,
                    "minimum_scenario_recall": minimum_scenario_recall,
                }
            )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1], rows
