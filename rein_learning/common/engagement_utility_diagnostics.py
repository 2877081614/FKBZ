from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class EngagementUtilityConfig:
    cost_weight: float = 1.0
    damage_weight: float = 30.0
    high_threat_leak_weight: float = 0.0
    cvar_weight: float = 0.0
    cvar_alpha: float = 0.25

    def __post_init__(self) -> None:
        weights = (
            self.cost_weight,
            self.damage_weight,
            self.high_threat_leak_weight,
            self.cvar_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Utility weights must be non-negative")
        if not 0.0 < self.cvar_alpha <= 1.0:
            raise ValueError("cvar_alpha must be in (0, 1]")

    def signature(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def lower_tail_cvar(samples: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("samples need at least two rollout values")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    count = max(1, int(np.ceil(alpha * values.shape[-1])))
    return np.mean(np.sort(values, axis=-1)[..., :count], axis=-1)


def engagement_utility_labels(
    components: Mapping[str, np.ndarray],
    config: EngagementUtilityConfig,
) -> tuple[np.ndarray, np.ndarray]:
    required = (
        "operational_return_samples",
        "resource_cost_samples",
        "damage_samples",
        "high_threat_leak_samples",
    )
    arrays = {
        key: np.asarray(components[key], dtype=np.float64) for key in required
    }
    shape = arrays[required[0]].shape
    if len(shape) != 3 or shape[1] != 2 or shape[2] < 2:
        raise ValueError("Component samples must have shape [groups, 2, rollouts>=2]")
    if any(array.shape != shape for array in arrays.values()):
        raise ValueError("All component sample arrays must have the same shape")
    utility_samples = (
        arrays["operational_return_samples"]
        - config.cost_weight * arrays["resource_cost_samples"]
        - config.damage_weight * arrays["damage_samples"]
        - config.high_threat_leak_weight * arrays["high_threat_leak_samples"]
    )
    means = np.mean(utility_samples, axis=-1)
    lower_cvar = lower_tail_cvar(utility_samples, config.cvar_alpha)
    labels = means - config.cvar_weight * (means - lower_cvar)
    return labels.astype(np.float32), utility_samples.astype(np.float32)


def _paired_standard_error(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Paired samples must contain at least two values")
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def safety_resource_oracle(
    components: Mapping[str, np.ndarray],
    *,
    damage_weight: float = 30.0,
    high_threat_leak_weight: float = 20.0,
    uncertainty_z: float = 1.645,
) -> dict[str, np.ndarray]:
    damage = np.asarray(components["damage_samples"], dtype=np.float64)
    leaks = np.asarray(components["high_threat_leak_samples"], dtype=np.float64)
    costs = np.asarray(components["resource_cost_samples"], dtype=np.float64)
    if damage.shape != leaks.shape or damage.shape != costs.shape:
        raise ValueError("Oracle component arrays must have the same shape")
    if damage.ndim != 3 or damage.shape[1] != 2 or damage.shape[2] < 2:
        raise ValueError("Oracle components must have shape [groups, 2, rollouts>=2]")
    if damage_weight < 0.0 or high_threat_leak_weight < 0.0:
        raise ValueError("Oracle harm weights must be non-negative")
    if uncertainty_z < 0.0:
        raise ValueError("uncertainty_z must be non-negative")

    harm = damage_weight * damage + high_threat_leak_weight * leaks
    harm_delta_samples = harm[:, 1] - harm[:, 0]
    cost_delta_samples = costs[:, 1] - costs[:, 0]
    harm_delta = np.mean(harm_delta_samples, axis=1)
    cost_delta = np.mean(cost_delta_samples, axis=1)
    harm_se = np.asarray(
        [_paired_standard_error(row) for row in harm_delta_samples]
    )
    cost_se = np.asarray(
        [_paired_standard_error(row) for row in cost_delta_samples]
    )
    labels = np.full(len(harm_delta), -1, dtype=np.int64)
    engage = harm_delta + uncertainty_z * harm_se < 0.0
    no_safety_improvement = ~engage
    costly = cost_delta - uncertainty_z * cost_se > 0.0
    labels[engage] = 1
    labels[no_safety_improvement & costly] = 0
    return {
        "labels": labels,
        "harm_delta": harm_delta.astype(np.float32),
        "harm_standard_error": harm_se.astype(np.float32),
        "cost_delta": cost_delta.astype(np.float32),
        "cost_standard_error": cost_se.astype(np.float32),
    }


def oracle_classification_metrics(
    oracle_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, float | int]:
    truth = np.asarray(oracle_labels, dtype=np.int64)
    predicted = np.asarray(predicted_labels, dtype=np.int64)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("Oracle and predicted labels must be aligned vectors")
    selected = truth >= 0
    truth = truth[selected]
    predicted = predicted[selected]
    if np.any((predicted < 0) | (predicted > 1)):
        raise ValueError("Predicted labels must be binary on oracle-valid groups")
    engage_count = int(np.sum(truth == 1))
    noop_count = int(np.sum(truth == 0))
    true_engage = int(np.sum((truth == 1) & (predicted == 1)))
    true_noop = int(np.sum((truth == 0) & (predicted == 0)))
    false_noop = engage_count - true_engage
    wasteful_engage = noop_count - true_noop
    engage_recall = true_engage / engage_count if engage_count else float("nan")
    noop_recall = true_noop / noop_count if noop_count else float("nan")
    balanced = (
        float((engage_recall + noop_recall) / 2.0)
        if engage_count and noop_count
        else float("nan")
    )
    total = len(truth)
    return {
        "count": int(total),
        "engage_count": engage_count,
        "noop_count": noop_count,
        "accuracy": float((true_engage + true_noop) / total) if total else float("nan"),
        "balanced_accuracy": balanced,
        "engage_recall": float(engage_recall),
        "noop_recall": float(noop_recall),
        "false_noop_rate": float(false_noop / engage_count)
        if engage_count
        else float("nan"),
        "wasteful_engage_rate": float(wasteful_engage / noop_count)
        if noop_count
        else float("nan"),
    }


def utility_oracle_metrics(
    utility_labels: np.ndarray,
    oracle_labels: np.ndarray,
) -> dict[str, float | int]:
    values = np.asarray(utility_labels, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Utility labels must have shape [groups, 2]")
    differences = values[:, 1] - values[:, 0]
    predicted = (differences > 0.0).astype(np.int64)
    return oracle_classification_metrics(oracle_labels, predicted)
