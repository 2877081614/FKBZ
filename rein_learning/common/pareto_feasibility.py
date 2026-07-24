from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .engagement_utility_diagnostics import oracle_classification_metrics
from .multibatch_diagnostics import grouped_oracle_metrics


@dataclass(frozen=True)
class ParetoRecallConstraints:
    balanced_accuracy: float = 0.70
    engage_recall: float = 0.60
    noop_recall: float = 0.65
    batch_engage_recall: float = 0.60
    batch_noop_recall: float = 0.65
    scenario_engage_recall: float = 0.60
    scenario_noop_recall: float = 0.65
    safety_sign_accuracy: float = 0.70


def complete_threshold_candidates(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("scores must be a non-empty vector")
    if np.any(~np.isfinite(values)):
        raise ValueError("scores must be finite")
    unique = np.unique(values)
    span = max(float(np.ptp(unique)), 1.0)
    edge = span * 1e-6
    if len(unique) == 1:
        return np.asarray([unique[0] - edge, unique[0] + edge])
    middle = unique[:-1] + (unique[1:] - unique[:-1]) / 2.0
    return np.concatenate(([unique[0] - edge], middle, [unique[-1] + edge]))


def _minimum_recall(
    metrics: dict[str, dict[str, float | int]],
    recall_name: str,
    count_name: str,
) -> float:
    values = [
        float(row[recall_name])
        for row in metrics.values()
        if int(row[count_name]) > 0
    ]
    return min(values) if values else float("nan")


def threshold_operating_point(
    scores: Sequence[float],
    oracle_labels: Sequence[int],
    batch_ids: Sequence[object],
    scenarios: Sequence[object],
    threshold: float,
    *,
    safety_sign_accuracy: float,
    constraints: ParetoRecallConstraints | None = None,
) -> dict[str, float | int | bool | dict[str, bool]]:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(oracle_labels, dtype=np.int64)
    batches = np.asarray(batch_ids)
    strata = np.asarray(scenarios)
    if not (values.shape == truth.shape == batches.shape == strata.shape):
        raise ValueError("scores, labels, batches, and scenarios must align")
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("calibration arrays must be non-empty vectors")
    valid = truth >= 0
    values = values[valid]
    truth = truth[valid]
    batches = batches[valid]
    strata = strata[valid]
    if not np.any(truth == 0) or not np.any(truth == 1):
        raise ValueError("calibration requires both reliable oracle classes")

    predicted = (values > float(threshold)).astype(np.int64)
    overall = oracle_classification_metrics(truth, predicted)
    by_batch = grouped_oracle_metrics(truth, predicted, batches)
    by_scenario = grouped_oracle_metrics(truth, predicted, strata)
    worst_batch_engage = _minimum_recall(
        by_batch, "engage_recall", "engage_count"
    )
    worst_batch_noop = _minimum_recall(
        by_batch, "noop_recall", "noop_count"
    )
    worst_scenario_engage = _minimum_recall(
        by_scenario, "engage_recall", "engage_count"
    )
    worst_scenario_noop = _minimum_recall(
        by_scenario, "noop_recall", "noop_count"
    )
    frozen = constraints or ParetoRecallConstraints()
    checks = {
        "balanced_accuracy": float(overall["balanced_accuracy"])
        >= frozen.balanced_accuracy,
        "engage_recall": float(overall["engage_recall"])
        >= frozen.engage_recall,
        "noop_recall": float(overall["noop_recall"]) >= frozen.noop_recall,
        "batch_engage_recall": worst_batch_engage
        >= frozen.batch_engage_recall,
        "batch_noop_recall": worst_batch_noop >= frozen.batch_noop_recall,
        "scenario_engage_recall": worst_scenario_engage
        >= frozen.scenario_engage_recall,
        "scenario_noop_recall": worst_scenario_noop
        >= frozen.scenario_noop_recall,
        "safety_sign_accuracy": float(safety_sign_accuracy)
        >= frozen.safety_sign_accuracy,
    }
    engage_margin = min(
        float(overall["engage_recall"]) - frozen.engage_recall,
        worst_batch_engage - frozen.batch_engage_recall,
        worst_scenario_engage - frozen.scenario_engage_recall,
    )
    noop_margin = min(
        float(overall["noop_recall"]) - frozen.noop_recall,
        worst_batch_noop - frozen.batch_noop_recall,
        worst_scenario_noop - frozen.scenario_noop_recall,
    )
    balanced_margin = (
        float(overall["balanced_accuracy"]) - frozen.balanced_accuracy
    )
    safety_margin = float(safety_sign_accuracy) - frozen.safety_sign_accuracy
    return {
        "threshold": float(threshold),
        **overall,
        "worst_batch_engage_recall": worst_batch_engage,
        "worst_batch_noop_recall": worst_batch_noop,
        "worst_scenario_engage_recall": worst_scenario_engage,
        "worst_scenario_noop_recall": worst_scenario_noop,
        "safety_sign_accuracy": float(safety_sign_accuracy),
        "engage_margin": engage_margin,
        "noop_margin": noop_margin,
        "balanced_accuracy_margin": balanced_margin,
        "safety_sign_margin": safety_margin,
        "minimum_constraint_margin": min(
            engage_margin, noop_margin, balanced_margin, safety_margin
        ),
        "predicted_engage_count": int(np.sum(predicted == 1)),
        "checks": checks,
        "passed_check_count": int(sum(checks.values())),
        "feasible": bool(all(checks.values())),
    }


def pareto_frontier_mask(
    operating_points: Sequence[dict[str, float | int | bool | dict[str, bool]]],
) -> np.ndarray:
    engage = np.asarray(
        [float(row["worst_scenario_engage_recall"]) for row in operating_points]
    )
    noop = np.asarray(
        [float(row["worst_scenario_noop_recall"]) for row in operating_points]
    )
    frontier = np.ones(len(operating_points), dtype=bool)
    for index in range(len(operating_points)):
        dominates = (
            (engage >= engage[index])
            & (noop >= noop[index])
            & ((engage > engage[index]) | (noop > noop[index]))
        )
        frontier[index] = not bool(np.any(dominates))
    return frontier


def audit_pareto_thresholds(
    scores: Sequence[float],
    oracle_labels: Sequence[int],
    batch_ids: Sequence[object],
    scenarios: Sequence[object],
    *,
    safety_sign_accuracy: float,
    constraints: ParetoRecallConstraints | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    candidates = complete_threshold_candidates(scores)
    rows: list[dict[str, object]] = [
        threshold_operating_point(
            scores,
            oracle_labels,
            batch_ids,
            scenarios,
            float(threshold),
            safety_sign_accuracy=safety_sign_accuracy,
            constraints=constraints,
        )
        for threshold in candidates
    ]
    frontier = pareto_frontier_mask(rows)
    for row, is_frontier in zip(rows, frontier):
        row["pareto_frontier"] = bool(is_frontier)

    ranked = sorted(
        rows,
        key=lambda row: (
            bool(row["feasible"]),
            float(row["minimum_constraint_margin"]),
            float(row["balanced_accuracy"]),
            -abs(float(row["threshold"])),
        ),
        reverse=True,
    )
    feasible = [row for row in rows if bool(row["feasible"])]
    zero = threshold_operating_point(
        scores,
        oracle_labels,
        batch_ids,
        scenarios,
        0.0,
        safety_sign_accuracy=safety_sign_accuracy,
        constraints=constraints,
    )
    summary: dict[str, object] = {
        "candidate_count": len(rows),
        "pareto_point_count": int(np.sum(frontier)),
        "feasible_threshold_count": len(feasible),
        "has_feasible_threshold": bool(feasible),
        "feasible_threshold_min": (
            min(float(row["threshold"]) for row in feasible)
            if feasible
            else None
        ),
        "feasible_threshold_max": (
            max(float(row["threshold"]) for row in feasible)
            if feasible
            else None
        ),
        "selected": ranked[0],
        "zero_threshold": zero,
    }
    return rows, summary
