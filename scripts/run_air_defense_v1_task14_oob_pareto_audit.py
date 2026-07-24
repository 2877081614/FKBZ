from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.common import (
    ParetoRecallConstraints,
    audit_pareto_thresholds,
    complete_threshold_candidates,
    threshold_operating_point,
)


DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_multibatch_leave_one_out"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_oob_pareto_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit OOB safety-stop Pareto threshold feasibility."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selected-objective")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    reliable = [row for row in rows if int(row["oracle_label"]) >= 0]
    return {
        "scores": np.asarray(
            [float(row["score"]) for row in reliable], dtype=np.float64
        ),
        "labels": np.asarray(
            [int(row["oracle_label"]) for row in reliable], dtype=np.int64
        ),
        "batches": np.asarray([row["batch_id"] for row in reliable]),
        "scenarios": np.asarray([row["scenario"] for row in reliable]),
    }


def _classes_complete(values: dict[str, np.ndarray], group_name: str) -> bool:
    labels = values["labels"]
    groups = values[group_name]
    return all(
        bool(np.any(labels[groups == group] == 0))
        and bool(np.any(labels[groups == group] == 1))
        for group in np.unique(groups)
    )


def _flatten_point(point: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in point.items()
        if key != "checks"
    } | {
        f"check_{key}": value
        for key, value in dict(point["checks"]).items()
    }


def _shared_threshold_audit(
    records_by_seed: dict[int, dict[str, np.ndarray]],
    safety_by_seed: dict[int, float],
    constraints: ParetoRecallConstraints,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = complete_threshold_candidates(
        np.concatenate([values["scores"] for values in records_by_seed.values()])
    )
    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        points = {
            seed: threshold_operating_point(
                values["scores"],
                values["labels"],
                values["batches"],
                values["scenarios"],
                float(threshold),
                safety_sign_accuracy=safety_by_seed[seed],
                constraints=constraints,
            )
            for seed, values in records_by_seed.items()
        }
        rows.append(
            {
                "threshold": float(threshold),
                "passed_seed_count": sum(
                    bool(point["feasible"]) for point in points.values()
                ),
                "minimum_seed_constraint_margin": min(
                    float(point["minimum_constraint_margin"])
                    for point in points.values()
                ),
                "mean_balanced_accuracy": float(
                    np.mean(
                        [
                            float(point["balanced_accuracy"])
                            for point in points.values()
                        ]
                    )
                ),
                "minimum_worst_batch_engage_recall": min(
                    float(point["worst_batch_engage_recall"])
                    for point in points.values()
                ),
                "minimum_worst_batch_noop_recall": min(
                    float(point["worst_batch_noop_recall"])
                    for point in points.values()
                ),
                "minimum_worst_scenario_engage_recall": min(
                    float(point["worst_scenario_engage_recall"])
                    for point in points.values()
                ),
                "minimum_worst_scenario_noop_recall": min(
                    float(point["worst_scenario_noop_recall"])
                    for point in points.values()
                ),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            int(row["passed_seed_count"]),
            float(row["minimum_seed_constraint_margin"]),
            float(row["mean_balanced_accuracy"]),
            -abs(float(row["threshold"])),
        ),
        reverse=True,
    )
    best = ranked[0]
    return rows, {
        "candidate_count": len(rows),
        "selected": best,
        "has_two_seed_shared_threshold": int(best["passed_seed_count"]) >= 2,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.input_dir / "oob_predictions.csv"
    config_path = args.input_dir / "experiment_config.json"
    summary_path = args.input_dir / "gate_summary.json"
    with config_path.open(encoding="utf-8") as handle:
        source_config = json.load(handle)
    with summary_path.open(encoding="utf-8") as handle:
        source_summary = json.load(handle)
    rows = _read_csv(prediction_path)
    objectives = tuple(str(value) for value in source_config["objectives"])
    model_seeds = tuple(int(value) for value in source_config["model_seeds"])
    selected_objective = (
        args.selected_objective
        or str(source_summary["leave_one_batch_out"]["selected_objective"])
    )
    if selected_objective not in objectives:
        raise ValueError(f"Unknown selected objective: {selected_objective}")

    constraints = ParetoRecallConstraints()
    point_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    objective_results: dict[str, dict[str, Any]] = {}
    primary_records: dict[int, dict[str, np.ndarray]] = {}
    primary_safety: dict[int, float] = {}
    combination_counts: dict[str, int] = {}
    class_checks: dict[str, bool] = {}

    for objective in objectives:
        objective_results[objective] = {}
        for model_seed in model_seeds:
            selected_rows = [
                row
                for row in rows
                if row["objective"] == objective
                and int(row["model_seed"]) == model_seed
            ]
            if not selected_rows:
                raise ValueError(f"Missing OOB rows for {objective}/seed{model_seed}")
            values = _records(selected_rows)
            safety_sign = float(
                source_summary["leave_one_batch_out"]["results"][
                    str(model_seed)
                ][objective]["value_metrics"]["safety_sign_accuracy"]
            )
            audit_rows, audit_summary = audit_pareto_thresholds(
                values["scores"],
                values["labels"],
                values["batches"],
                values["scenarios"],
                safety_sign_accuracy=safety_sign,
                constraints=constraints,
            )
            for point in audit_rows:
                point_rows.append(
                    {
                        "objective": objective,
                        "model_seed": model_seed,
                        **_flatten_point(point),
                    }
                )
            selected = dict(audit_summary["selected"])
            zero = dict(audit_summary["zero_threshold"])
            objective_results[objective][str(model_seed)] = audit_summary
            seed_rows.append(
                {
                    "objective": objective,
                    "model_seed": model_seed,
                    "reliable_rows": len(values["labels"]),
                    "has_feasible_threshold": audit_summary[
                        "has_feasible_threshold"
                    ],
                    "feasible_threshold_count": audit_summary[
                        "feasible_threshold_count"
                    ],
                    "feasible_threshold_min": audit_summary[
                        "feasible_threshold_min"
                    ],
                    "feasible_threshold_max": audit_summary[
                        "feasible_threshold_max"
                    ],
                    "selected_threshold": selected["threshold"],
                    "selected_minimum_constraint_margin": selected[
                        "minimum_constraint_margin"
                    ],
                    "selected_balanced_accuracy": selected[
                        "balanced_accuracy"
                    ],
                    "selected_engage_recall": selected["engage_recall"],
                    "selected_noop_recall": selected["noop_recall"],
                    "selected_worst_batch_engage": selected[
                        "worst_batch_engage_recall"
                    ],
                    "selected_worst_batch_noop": selected[
                        "worst_batch_noop_recall"
                    ],
                    "selected_worst_scenario_engage": selected[
                        "worst_scenario_engage_recall"
                    ],
                    "selected_worst_scenario_noop": selected[
                        "worst_scenario_noop_recall"
                    ],
                    "zero_threshold_feasible": zero["feasible"],
                    "zero_threshold_minimum_constraint_margin": zero[
                        "minimum_constraint_margin"
                    ],
                    "safety_sign_accuracy": safety_sign,
                }
            )
            key = f"{objective}/seed{model_seed}"
            combination_counts[key] = len(selected_rows)
            class_checks[key] = _classes_complete(
                values, "batches"
            ) and _classes_complete(values, "scenarios")
            if objective == selected_objective:
                primary_records[model_seed] = values
                primary_safety[model_seed] = safety_sign

    shared_rows, shared_summary = _shared_threshold_audit(
        primary_records, primary_safety, constraints
    )
    selected_results = objective_results[selected_objective]
    feasible_seed_count = sum(
        bool(selected_results[str(seed)]["has_feasible_threshold"])
        for seed in model_seeds
    )
    zero_threshold_passes = sum(
        bool(selected_results[str(seed)]["zero_threshold"]["feasible"])
        for seed in model_seeds
    )
    batch_count = len(
        np.unique(
            np.concatenate(
                [value["batches"] for value in primary_records.values()]
            )
        )
    )
    scenario_count = len(
        np.unique(
            np.concatenate(
                [value["scenarios"] for value in primary_records.values()]
            )
        )
    )
    data_checks = {
        "prediction_rows_present": len(rows) > 0,
        "objectives_complete": set(row["objective"] for row in rows)
        == set(objectives),
        "model_seeds_complete": {
            int(row["model_seed"]) for row in rows
        }
        == set(model_seeds),
        "three_batches": batch_count == 3,
        "three_scenarios": scenario_count == 3,
        "reliable_classes_by_batch_and_scenario": all(class_checks.values()),
        "selected_objective_frozen": selected_objective
        == source_summary["leave_one_batch_out"]["selected_objective"],
        "test_data_accessed": False,
        "new_rollouts": 0,
    }
    data_integrity = all(
        bool(value) if key not in {"test_data_accessed", "new_rollouts"}
        else value in {False, 0}
        for key, value in data_checks.items()
    )
    stage_checks = {
        "data_integrity": data_integrity,
        "selected_objective_two_of_three_feasible": feasible_seed_count >= 2,
        "selected_objective_safety_sign": all(
            primary_safety[seed] >= constraints.safety_sign_accuracy
            for seed in model_seeds
        ),
    }
    stage_passed = all(stage_checks.values())
    next_action = (
        "freeze_oob_threshold_rule_and_run_one_independent_confirmation_batch"
        if stage_passed
        else "revise_value_semantics_or_explicit_constraints"
    )
    summary = {
        "schema_version": 1,
        "source": {
            "input_dir": str(args.input_dir),
            "prediction_file": str(prediction_path),
            "prediction_sha256": _sha256(prediction_path),
            "prediction_row_count": len(rows),
            "combination_row_counts": combination_counts,
        },
        "protocol": {
            "selected_objective": selected_objective,
            "diagnostic_objectives": list(objectives),
            "model_seeds": list(model_seeds),
            "constraints": constraints.__dict__,
            "threshold_rule": "score > threshold",
            "test_data_accessed": False,
            "new_rollouts": 0,
        },
        "data_audit": {
            "checks": data_checks,
            "passed": data_integrity,
        },
        "objective_results": objective_results,
        "primary_gate": {
            "zero_threshold_passed_seed_count": zero_threshold_passes,
            "robust_threshold_feasible_seed_count": feasible_seed_count,
            "required_feasible_seed_count": 2,
            "checks": stage_checks,
            "passed": stage_passed,
        },
        "shared_raw_threshold": shared_summary,
        "task14_oob_pareto_feasibility_passed": stage_passed,
        "allow_independent_confirmation": stage_passed,
        "resume_mch_ppo": False,
        "enter_gnn": False,
        "next_action": next_action,
    }
    _write_csv(args.output_dir / "pareto_points.csv", point_rows)
    _write_csv(args.output_dir / "seed_summary.csv", seed_rows)
    _write_csv(args.output_dir / "shared_threshold_points.csv", shared_rows)
    with (args.output_dir / "gate_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (args.output_dir / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "input_dir": str(args.input_dir),
                "selected_objective": selected_objective,
                "model_seeds": list(model_seeds),
                "objectives": list(objectives),
                "constraints": constraints.__dict__,
                "new_rollouts": 0,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
