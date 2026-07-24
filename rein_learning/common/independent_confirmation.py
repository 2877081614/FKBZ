from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .pareto_feasibility import (
    ParetoRecallConstraints,
    threshold_operating_point,
)


def frozen_thresholds_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    objective: str,
    model_seeds: Sequence[int],
) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    expected = set(int(seed) for seed in model_seeds)
    for row in rows:
        if str(row["objective"]) != objective:
            continue
        seed = int(row["model_seed"])
        if seed not in expected:
            continue
        if seed in thresholds:
            raise ValueError(f"Duplicate frozen threshold for seed {seed}")
        if str(row["has_feasible_threshold"]).lower() not in {"true", "1"}:
            raise ValueError(f"Seed {seed} has no feasible frozen threshold")
        threshold = float(row["selected_threshold"])
        if not np.isfinite(threshold):
            raise ValueError(f"Seed {seed} threshold must be finite")
        thresholds[seed] = threshold
    missing = expected - set(thresholds)
    if missing:
        raise ValueError(f"Missing frozen thresholds for seeds {sorted(missing)}")
    return thresholds


def confirmation_power(
    oracle_labels: Sequence[int],
    scenarios: Sequence[object],
    *,
    minimum_valid: int = 40,
    minimum_per_class: int = 10,
    minimum_per_scenario: int = 8,
) -> dict[str, object]:
    labels = np.asarray(oracle_labels, dtype=np.int64)
    strata = np.asarray(scenarios)
    if labels.shape != strata.shape or labels.ndim != 1:
        raise ValueError("labels and scenarios must be aligned vectors")
    valid = labels >= 0
    scenario_counts: dict[str, dict[str, int]] = {}
    for scenario in np.unique(strata):
        mask = valid & (strata == scenario)
        scenario_counts[str(scenario)] = {
            "valid": int(np.sum(mask)),
            "engage": int(np.sum(mask & (labels == 1))),
            "noop": int(np.sum(mask & (labels == 0))),
        }
    checks = {
        "valid_count": int(np.sum(valid)) >= minimum_valid,
        "engage_count": int(np.sum(labels == 1)) >= minimum_per_class,
        "noop_count": int(np.sum(labels == 0)) >= minimum_per_class,
        "scenario_counts": all(
            row["valid"] >= minimum_per_scenario
            for row in scenario_counts.values()
        ),
        "scenario_classes_complete": all(
            row["engage"] > 0 and row["noop"] > 0
            for row in scenario_counts.values()
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "valid_count": int(np.sum(valid)),
        "engage_count": int(np.sum(labels == 1)),
        "noop_count": int(np.sum(labels == 0)),
        "scenario_counts": scenario_counts,
    }


def evaluate_frozen_threshold(
    scores: Sequence[float],
    oracle_labels: Sequence[int],
    scenarios: Sequence[object],
    *,
    threshold: float,
    safety_sign_accuracy: float,
    constraints: ParetoRecallConstraints | None = None,
) -> dict[str, object]:
    labels = np.asarray(oracle_labels)
    return threshold_operating_point(
        scores,
        labels,
        np.full(labels.shape, "independent_confirmation"),
        scenarios,
        threshold,
        safety_sign_accuracy=safety_sign_accuracy,
        constraints=constraints,
    )
