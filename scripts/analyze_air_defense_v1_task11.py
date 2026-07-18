from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
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


MAIN_METHOD = "role_conditioned_ar_ppo_order_012"
ROLE_METHODS = (
    MAIN_METHOD,
    "role_conditioned_ar_ppo_order_120",
    "role_conditioned_ar_ppo_order_201",
)
TASK10_BASELINE = "autoregressive_ppo_order_012"
TASK10_ACTOR_PARAMETERS = 37_138
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task11_role_conditioned_screening_30k_3seeds"
)
DEFAULT_TASK10_DIR = (
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
        description="Evaluate the frozen Task-11 role-conditioned gates."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--task10-dir", type=Path, default=DEFAULT_TASK10_DIR)
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
    candidate_rows: Sequence[dict[str, str]],
    reference_rows: Sequence[dict[str, str]],
    *,
    field: str,
) -> int:
    candidate = {
        int(row["run_index"]): float(row[field]) for row in candidate_rows
    }
    reference = {
        int(row["run_index"]): float(row[field]) for row in reference_rows
    }
    if candidate.keys() != reference.keys():
        raise ValueError("Candidate and reference run indices do not match")
    return sum(candidate[index] < reference[index] for index in candidate)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _collapsed_unit_rows(
    decision_summary: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int], int] = {}
    for row in decision_summary:
        if row["method"] not in ROLE_METHODS:
            continue
        key = (
            row["method"],
            row["eval_scenario"],
            int(row["run_index"]),
            int(row["train_seed"]),
        )
        grouped[key] = grouped.get(key, 0) + int(
            _parse_bool(row["collapsed_unit"])
        )
    return [
        {
            "method": key[0],
            "eval_scenario": key[1],
            "run_index": key[2],
            "train_seed": key[3],
            "collapsed_unit_count": count,
        }
        for key, count in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def _main_gate_row(
    runs: Sequence[dict[str, str]],
    task10_runs: Sequence[dict[str, str]],
    task9_runs: Sequence[dict[str, str]],
    parameter_record: dict[str, Any],
    collapsed_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    candidate_all = _select(runs, method=MAIN_METHOD)
    candidate_medium = _select(runs, method=MAIN_METHOD, scenario="medium")
    candidate_time = _select(
        runs,
        method=MAIN_METHOD,
        scenario="time_pressure",
    )
    candidate_hetero = _select(
        runs,
        method=MAIN_METHOD,
        scenario="heterogeneity_pressure",
    )
    reference_all = _select(task10_runs, method=TASK10_BASELINE)
    reference_medium = _select(
        task10_runs,
        method=TASK10_BASELINE,
        scenario="medium",
    )
    reference_time = _select(
        task10_runs,
        method=TASK10_BASELINE,
        scenario="time_pressure",
    )
    reference_hetero = _select(
        task10_runs,
        method=TASK10_BASELINE,
        scenario="heterogeneity_pressure",
    )

    medium_reward_change = _mean(candidate_medium, "avg_reward") - _mean(
        reference_medium, "avg_reward"
    )
    medium_damage_change = _mean(
        candidate_medium, "avg_total_damage"
    ) - _mean(reference_medium, "avg_total_damage")
    time_cost_change = _mean(candidate_time, "avg_resource_cost") - _mean(
        reference_time, "avg_resource_cost"
    )
    hetero_high_leak_reduction = _mean(
        reference_hetero, "high_threat_leak_rate"
    ) - _mean(candidate_hetero, "high_threat_leak_rate")
    improved_seeds = _paired_improvement_count(
        candidate_hetero,
        reference_hetero,
        field="high_threat_leak_rate",
    )
    hetero_damage_change = _mean(
        candidate_hetero, "avg_total_damage"
    ) - _mean(reference_hetero, "avg_total_damage")
    decision_time_change = (
        _mean(candidate_all, "avg_decision_time_ms")
        / _mean(reference_all, "avg_decision_time_ms")
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
    actor_parameters = int(parameter_record["actor_parameters"])
    actor_parameter_ratio = actor_parameters / TASK10_ACTOR_PARAMETERS
    hetero_collapsed = sum(
        int(row["collapsed_unit_count"])
        for row in collapsed_rows
        if row["method"] == MAIN_METHOD
        and row["eval_scenario"] == "heterogeneity_pressure"
    )
    internal_checks = {
        "structural_zero_pass": structural_max == 0.0,
        "actor_capacity_pass": 0.85 <= actor_parameter_ratio <= 1.15,
        "medium_reward_pass": medium_reward_change >= -5.0,
        "medium_damage_pass": medium_damage_change <= 0.10,
        "time_cost_pass": time_cost_change <= 0.50,
        "hetero_high_leak_mean_pass": hetero_high_leak_reduction >= 0.02,
        "hetero_high_leak_seed_pass": improved_seeds >= 2,
        "hetero_damage_pass": hetero_damage_change <= 0.10,
        "hetero_no_collapsed_units_pass": hetero_collapsed == 0,
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
        "method": MAIN_METHOD,
        "reference": TASK10_BASELINE,
        "structural_max": structural_max,
        "actor_parameters": actor_parameters,
        "reference_actor_parameters": TASK10_ACTOR_PARAMETERS,
        "actor_parameter_ratio": actor_parameter_ratio,
        "medium_reward_change": medium_reward_change,
        "medium_damage_change": medium_damage_change,
        "time_cost_change": time_cost_change,
        "hetero_high_leak_reduction": hetero_high_leak_reduction,
        "hetero_high_leak_improved_seeds": improved_seeds,
        "hetero_damage_change": hetero_damage_change,
        "hetero_collapsed_unit_total": hetero_collapsed,
        "decision_time_percent_change": 100.0 * decision_time_change,
        **internal_checks,
        **external_differences,
        **external_checks,
        "internal_gate_pass": all(internal_checks.values()),
        "external_gate_pass": all(external_checks.values()),
    }


def _order_robustness_row(
    runs: Sequence[dict[str, str]],
    collapsed_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    hetero_leaks = {
        method: _mean(
            _select(runs, method=method, scenario="heterogeneity_pressure"),
            "high_threat_leak_rate",
        )
        for method in ROLE_METHODS
    }
    time_costs = {
        method: _mean(
            _select(runs, method=method, scenario="time_pressure"),
            "avg_resource_cost",
        )
        for method in ROLE_METHODS
    }
    hetero_damage = {
        method: _mean(
            _select(runs, method=method, scenario="heterogeneity_pressure"),
            "avg_total_damage",
        )
        for method in ROLE_METHODS
    }
    leak_range = max(hetero_leaks.values()) - min(hetero_leaks.values())
    cost_range = max(time_costs.values()) - min(time_costs.values())
    damage_range = max(hetero_damage.values()) - min(hetero_damage.values())
    collapsed_total = sum(
        int(row["collapsed_unit_count"])
        for row in collapsed_rows
        if row["eval_scenario"] == "heterogeneity_pressure"
    )
    checks = {
        "high_leak_range_pass": leak_range <= 0.03,
        "resource_cost_range_pass": cost_range <= 1.00,
        "damage_range_pass": damage_range <= 0.15,
        "all_orders_no_collapsed_units_pass": collapsed_total == 0,
    }
    return {
        "hetero_high_leak_range": leak_range,
        "time_resource_cost_range": cost_range,
        "hetero_damage_range": damage_range,
        "hetero_collapsed_unit_total": collapsed_total,
        **checks,
        "order_robustness_pass": all(checks.values()),
    }


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


def _unit_assignment_cv_rows(
    pooled_decisions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in pooled_decisions:
        if row["resource_type"] != "missile":
            continue
        key = (str(row["method"]), str(row["eval_scenario"]))
        grouped.setdefault(key, []).append(float(row["assignment_rate"]))
    output = []
    for (method, scenario), values in sorted(grouped.items()):
        mean = statistics.fmean(values)
        output.append(
            {
                "method": method,
                "eval_scenario": scenario,
                "missile_unit_count": len(values),
                "mean_assignment_rate": mean,
                "unit_assignment_cv": (
                    statistics.pstdev(values) / mean if mean else 0.0
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs = _read_csv(result_dir / "runs.csv")
    task10_runs = _read_csv(args.task10_dir.resolve() / "runs.csv")
    task9_runs = _read_csv(args.task9_dir.resolve() / "runs.csv")
    decision_summary = _read_csv(result_dir / "decision_summary.csv")
    collapsed_rows = _collapsed_unit_rows(decision_summary)
    with (result_dir / "model_parameter_counts.json").open(
        encoding="utf-8"
    ) as json_file:
        parameter_payload = json.load(json_file)
    main_parameter_record = next(
        row
        for row in parameter_payload["models"]
        if row["method"] == MAIN_METHOD
    )

    main_gate = _main_gate_row(
        runs,
        task10_runs,
        task9_runs,
        main_parameter_record,
        collapsed_rows,
    )
    order_gate = _order_robustness_row(runs, collapsed_rows)
    eligible = (
        main_gate["internal_gate_pass"]
        and main_gate["external_gate_pass"]
        and order_gate["order_robustness_pass"]
    )
    main_gate["order_robustness_pass"] = order_gate["order_robustness_pass"]
    main_gate["confirmation_100k_eligible"] = eligible
    _write_csv(
        result_dir / "screening_gate_evaluation.csv",
        [main_gate],
        tuple(main_gate),
    )
    _write_csv(
        result_dir / "order_robustness_evaluation.csv",
        [order_gate],
        tuple(order_gate),
    )
    _write_csv(
        result_dir / "collapsed_units.csv",
        collapsed_rows,
        (
            "method",
            "eval_scenario",
            "run_index",
            "train_seed",
            "collapsed_unit_count",
        ),
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
    cv_rows = _unit_assignment_cv_rows(pooled_decisions)
    _write_csv(
        result_dir / "unit_assignment_cv.csv",
        cv_rows,
        (
            "method",
            "eval_scenario",
            "missile_unit_count",
            "mean_assignment_rate",
            "unit_assignment_cv",
        ),
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
        "main_method": MAIN_METHOD,
        "task10_reference": TASK10_BASELINE,
        "main_gate": main_gate,
        "order_robustness_gate": order_gate,
        "decision_rows": len(decisions),
        "leak_attribution_rows": len(leak_rows),
        "decision": (
            "run_independent_100k_confirmation"
            if eligible
            else "do_not_run_100k_confirmation"
        ),
    }
    with (result_dir / "task11_analysis.json").open(
        "w", encoding="utf-8"
    ) as json_file:
        json.dump(analysis, json_file, ensure_ascii=False, indent=2)
        json_file.write("\n")

    print(
        f"main_internal={main_gate['internal_gate_pass']}, "
        f"external={main_gate['external_gate_pass']}, "
        f"order_robustness={order_gate['order_robustness_pass']}, "
        f"eligible={eligible}"
    )
    print(
        f"hetero_high_leak_reduction="
        f"{main_gate['hetero_high_leak_reduction']:.6f}, "
        f"improved_seeds={main_gate['hetero_high_leak_improved_seeds']}/3, "
        f"time_cost_change={main_gate['time_cost_change']:+.6f}, "
        f"collapsed_units={main_gate['hetero_collapsed_unit_total']}"
    )
    print(f"Analysis written to: {result_dir}")


if __name__ == "__main__":
    main()
