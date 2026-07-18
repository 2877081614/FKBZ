from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.common import (
    aggregate_decision_rows,
    aggregate_leak_attributions,
)
from rein_learning.experiments.air_defense_v1_benchmark import (
    DECISION_SUMMARY_FIELDNAMES,
    LEAK_ATTRIBUTION_SUMMARY_FIELDNAMES,
)


BASELINE = "autoregressive_ppo_order_012"
CANDIDATES = (
    "autoregressive_ppo_order_120",
    "autoregressive_ppo_order_201",
)
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task10_order_screening_30k_3seeds"
)
DEFAULT_TASK9_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task9_autoregressive_screening_30k_3seeds"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen Task-10 order-screening gates."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--task9-dir", type=Path, default=DEFAULT_TASK9_DIR)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: Iterable[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not values:
        raise ValueError(f"No values available for {field}")
    return sum(values) / len(values)


def _select(
    rows: Sequence[dict[str, str]],
    *,
    method: str,
    scenario: str | None = None,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["method"] == method
        and (scenario is None or row["eval_scenario"] == scenario)
    ]


def _paired_improvement_count(
    runs: Sequence[dict[str, str]],
    *,
    candidate: str,
    scenario: str,
    field: str,
) -> int:
    baseline_by_run = {
        int(row["run_index"]): float(row[field])
        for row in _select(runs, method=BASELINE, scenario=scenario)
    }
    candidate_by_run = {
        int(row["run_index"]): float(row[field])
        for row in _select(runs, method=candidate, scenario=scenario)
    }
    if baseline_by_run.keys() != candidate_by_run.keys():
        raise ValueError("Candidate and baseline run indices do not match")
    return sum(
        candidate_by_run[index] < baseline_by_run[index]
        for index in baseline_by_run
    )


def _screen_candidate(
    runs: Sequence[dict[str, str]],
    task9_runs: Sequence[dict[str, str]],
    candidate: str,
) -> dict[str, Any]:
    baseline_all = _select(runs, method=BASELINE)
    candidate_all = _select(runs, method=candidate)
    baseline_medium = _select(runs, method=BASELINE, scenario="medium")
    candidate_medium = _select(runs, method=candidate, scenario="medium")
    baseline_time = _select(runs, method=BASELINE, scenario="time_pressure")
    candidate_time = _select(runs, method=candidate, scenario="time_pressure")
    baseline_hetero = _select(
        runs,
        method=BASELINE,
        scenario="heterogeneity_pressure",
    )
    candidate_hetero = _select(
        runs,
        method=candidate,
        scenario="heterogeneity_pressure",
    )

    medium_reward_change = _mean(candidate_medium, "avg_reward") - _mean(
        baseline_medium, "avg_reward"
    )
    medium_damage_change = _mean(
        candidate_medium, "avg_total_damage"
    ) - _mean(baseline_medium, "avg_total_damage")
    time_cost_change = _mean(candidate_time, "avg_resource_cost") - _mean(
        baseline_time, "avg_resource_cost"
    )
    hetero_high_leak_reduction = _mean(
        baseline_hetero, "high_threat_leak_rate"
    ) - _mean(candidate_hetero, "high_threat_leak_rate")
    improved_seeds = _paired_improvement_count(
        runs,
        candidate=candidate,
        scenario="heterogeneity_pressure",
        field="high_threat_leak_rate",
    )
    hetero_damage_change = _mean(
        candidate_hetero, "avg_total_damage"
    ) - _mean(baseline_hetero, "avg_total_damage")
    decision_time_change = (
        _mean(candidate_all, "avg_decision_time_ms")
        / _mean(baseline_all, "avg_decision_time_ms")
        - 1.0
    )
    structural_max = max(
        float(row[field])
        for row in candidate_all
        for field in (
            "avg_invalid_actions",
            "assignment_conflict_rate",
            "overkill_rate",
        )
    )

    internal_checks = {
        "structural_zero_pass": structural_max == 0.0,
        "medium_reward_pass": medium_reward_change >= -5.0,
        "medium_damage_pass": medium_damage_change <= 0.10,
        "time_cost_pass": time_cost_change <= 0.50,
        "hetero_high_leak_mean_pass": hetero_high_leak_reduction >= 0.02,
        "hetero_high_leak_seed_pass": improved_seeds >= 2,
        "hetero_damage_pass": hetero_damage_change <= 0.10,
        "decision_time_pass": decision_time_change <= 0.25,
    }

    original_medium = _select(
        task9_runs,
        method="maskable_ppo",
        scenario="medium",
    )
    original_time = _select(
        task9_runs,
        method="maskable_ppo",
        scenario="time_pressure",
    )
    original_hetero = _select(
        task9_runs,
        method="maskable_ppo",
        scenario="heterogeneity_pressure",
    )
    discrete_time = _select(
        task9_runs,
        method="conflict_free_maskable_ppo",
        scenario="time_pressure",
    )
    external_differences = {
        "vs_maskable_medium_reward_change": (
            _mean(candidate_medium, "avg_reward")
            - _mean(original_medium, "avg_reward")
        ),
        "vs_maskable_medium_damage_change": (
            _mean(candidate_medium, "avg_total_damage")
            - _mean(original_medium, "avg_total_damage")
        ),
        "vs_maskable_time_reward_change": (
            _mean(candidate_time, "avg_reward")
            - _mean(original_time, "avg_reward")
        ),
        "vs_maskable_time_cost_change": (
            _mean(candidate_time, "avg_resource_cost")
            - _mean(original_time, "avg_resource_cost")
        ),
        "vs_maskable_hetero_damage_change": (
            _mean(candidate_hetero, "avg_total_damage")
            - _mean(original_hetero, "avg_total_damage")
        ),
        "vs_discrete_time_cost_change": (
            _mean(candidate_time, "avg_resource_cost")
            - _mean(discrete_time, "avg_resource_cost")
        ),
    }
    external_checks = {
        "external_medium_reward_pass": (
            external_differences["vs_maskable_medium_reward_change"] >= -5.0
        ),
        "external_medium_damage_pass": (
            external_differences["vs_maskable_medium_damage_change"] <= 0.10
        ),
        "external_time_reward_pass": (
            external_differences["vs_maskable_time_reward_change"] >= -5.0
        ),
        "external_time_cost_pass": (
            external_differences["vs_maskable_time_cost_change"] <= 0.50
        ),
        "external_hetero_damage_pass": (
            external_differences["vs_maskable_hetero_damage_change"] <= 0.10
        ),
        "external_discrete_cost_pass": (
            external_differences["vs_discrete_time_cost_change"] < 0.0
        ),
    }

    return {
        "candidate": candidate,
        "reference": BASELINE,
        "structural_max": structural_max,
        "medium_reward_change": medium_reward_change,
        "medium_damage_change": medium_damage_change,
        "time_cost_change": time_cost_change,
        "hetero_high_leak_reduction": hetero_high_leak_reduction,
        "hetero_high_leak_improved_seeds": improved_seeds,
        "hetero_damage_change": hetero_damage_change,
        "decision_time_percent_change": 100.0 * decision_time_change,
        **internal_checks,
        **external_differences,
        **external_checks,
        "internal_gate_pass": all(internal_checks.values()),
        "external_gate_pass": all(external_checks.values()),
        "confirmation_100k_eligible": (
            all(internal_checks.values()) and all(external_checks.values())
        ),
    }


def _parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _parse_optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _parse_decision_row(row: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = dict(row)
    for field in (
        "selected_noop",
        "avoidable_noop",
        "selected_high_threat",
    ):
        parsed[field] = _parse_bool(row[field])
    for field in (
        "num_conditional_legal_targets",
        "num_conditional_high_threat_targets",
        "prefix_denied_target_count",
        "num_base_legal_targets",
    ):
        parsed[field] = int(row[field])
    for field in (
        "matching_efficiency",
        "expected_damage_reduction",
        "target_threat",
    ):
        parsed[field] = _parse_optional_float(row[field])
    return parsed


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs = _read_csv(result_dir / "runs.csv")
    task9_runs = _read_csv(args.task9_dir.resolve() / "runs.csv")
    gate_rows = [
        _screen_candidate(runs, task9_runs, candidate)
        for candidate in CANDIDATES
    ]
    gate_fieldnames = tuple(gate_rows[0])
    _write_csv(
        result_dir / "screening_gate_evaluation.csv",
        gate_rows,
        gate_fieldnames,
    )

    decisions = [
        _parse_decision_row(row)
        for row in _read_csv(result_dir / "decisions.csv")
    ]
    pooled_decisions = aggregate_decision_rows(
        decisions,
        group_keys=(
            "method",
            "method_type",
            "train_scenario",
            "eval_scenario",
            "unit_order",
            "unit_index",
            "resource_type",
            "unit_order_position",
        ),
    )
    pooled_decision_fields = tuple(
        field
        for field in DECISION_SUMMARY_FIELDNAMES
        if field not in {"run_index", "train_seed"}
    )
    _write_csv(
        result_dir / "decision_pooled_summary.csv",
        pooled_decisions,
        pooled_decision_fields,
    )

    leak_rows = _read_csv(result_dir / "leak_attributions.csv")
    pooled_leaks = aggregate_leak_attributions(
        leak_rows,
        group_keys=(
            "method",
            "method_type",
            "train_scenario",
            "eval_scenario",
        ),
    )
    pooled_leak_fields = tuple(
        field
        for field in LEAK_ATTRIBUTION_SUMMARY_FIELDNAMES
        if field not in {"run_index", "train_seed"}
    )
    _write_csv(
        result_dir / "leak_attribution_pooled_summary.csv",
        pooled_leaks,
        pooled_leak_fields,
    )

    analysis = {
        "schema_version": 1,
        "result_dir": str(result_dir),
        "reference_method": BASELINE,
        "gate_results": gate_rows,
        "decision_rows": len(decisions),
        "leak_attribution_rows": len(leak_rows),
        "eligible_candidates": [
            row["candidate"]
            for row in gate_rows
            if row["confirmation_100k_eligible"]
        ],
        "decision": (
            "run_independent_100k_confirmation"
            if any(row["confirmation_100k_eligible"] for row in gate_rows)
            else "do_not_run_100k_confirmation"
        ),
    }
    with (result_dir / "task10_analysis.json").open(
        "w", encoding="utf-8"
    ) as json_file:
        json.dump(analysis, json_file, ensure_ascii=False, indent=2)
        json_file.write("\n")

    for row in gate_rows:
        print(
            f"{row['candidate']}: internal={row['internal_gate_pass']}, "
            f"external={row['external_gate_pass']}, "
            f"eligible={row['confirmation_100k_eligible']}, "
            f"hetero_high_leak_reduction="
            f"{row['hetero_high_leak_reduction']:.6f}, "
            f"improved_seeds={row['hetero_high_leak_improved_seeds']}/3, "
            f"time_cost_change={row['time_cost_change']:+.6f}"
        )
    print(f"Analysis written to: {result_dir}")


if __name__ == "__main__":
    main()
