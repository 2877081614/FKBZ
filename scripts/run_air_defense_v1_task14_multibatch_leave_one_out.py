from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.common import (
    batch_scenario_groups,
    constrained_value_metrics,
    engagement_delta_targets,
    grouped_oracle_metrics,
    leave_one_batch_out_folds,
    minimum_group_class_recall,
    oracle_classification_metrics,
    paired_delta_reliability,
    safety_resource_oracle,
    scenario_classification_metrics,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import AirDefenseV1ObservationLayout
from scripts.run_air_defense_v1_task14_cross_scenario_robust_value import (
    DEFAULT_PREVIOUS_DIR,
    OBJECTIVES,
    _evaluate_value_model,
    _load_previous_model,
    _worst_scenario_recall,
)
from scripts.run_air_defense_v1_task14_engagement_calibration import (
    DEFAULT_CRITIC_DIR,
    OLD_DATASETS,
    _load_critic,
    _load_npz,
    _predict,
)
from scripts.run_air_defense_v1_task14_engagement_utility import (
    COMPONENT_KEYS,
    _components,
    _observation_overlap_count,
)
from scripts.run_air_defense_v1_task14_q_critic import (
    DEFAULT_MODEL_DIR,
    SCENARIOS,
    _write_csv,
)
from scripts.run_air_defense_v1_task14_state_conditioned_value import (
    _fit_final,
    _fit_fold,
    _generate_test_dataset,
    _model_gate,
    _predict_value,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_multibatch_leave_one_out"
)
ROW_KEYS = (
    "observations",
    "unit_indices",
    "prefix_occupancy",
    "legal_action_masks",
    "group_ids",
    "state_ids",
    "source_seeds",
    "scenarios",
    *COMPONENT_KEYS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-batch leave-one-batch-out engagement diagnostics."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC_DIR)
    parser.add_argument("--previous-dir", type=Path, default=DEFAULT_PREVIOUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-seeds", nargs="+", type=int, default=(8, 10))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument(
        "--training-batch-seeds",
        nargs="+",
        type=int,
        default=(701_000, 719_000, 737_000),
    )
    parser.add_argument("--training-states-per-stratum", type=int, default=8)
    parser.add_argument("--training-episodes-per-stratum", type=int, default=24)
    parser.add_argument("--eval-seed", type=int, default=809_000)
    parser.add_argument("--states-per-stratum", type=int, default=12)
    parser.add_argument("--episodes-per-stratum", type=int, default=30)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=(20, 21, 22))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-training-batches", action="store_true")
    parser.add_argument("--reuse-test-dataset", action="store_true")
    return parser.parse_args()


def _batch_generation_args(
    args: argparse.Namespace, batch_seed: int, output_dir: Path
) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "eval_seed": batch_seed,
            "output_dir": output_dir,
            "states_per_stratum": args.training_states_per_stratum,
            "episodes_per_stratum": args.training_episodes_per_stratum,
        }
    )
    return argparse.Namespace(**values)


def _batch_power(
    dataset: dict[str, np.ndarray], scenarios: tuple[str, ...] | list[str]
) -> dict[str, Any]:
    indices = np.arange(len(dataset["group_ids"]))
    labels = safety_resource_oracle(_components(dataset, indices))["labels"]
    valid = labels >= 0
    scenario_counts = {
        scenario: int(np.sum(valid & (dataset["scenarios"] == scenario)))
        for scenario in scenarios
    }
    checks = {
        "valid_count": int(np.sum(valid)) >= 30,
        "engage_count": int(np.sum(labels == 1)) >= 6,
        "noop_count": int(np.sum(labels == 0)) >= 6,
        "scenario_counts": all(count >= 5 for count in scenario_counts.values()),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "valid_count": int(np.sum(valid)),
        "engage_count": int(np.sum(labels == 1)),
        "noop_count": int(np.sum(labels == 0)),
        "scenario_counts": scenario_counts,
    }


def _combine_batches(
    batches: list[tuple[str, dict[str, np.ndarray]]]
) -> dict[str, np.ndarray]:
    combined = {
        key: np.concatenate([dataset[key] for _, dataset in batches], axis=0)
        for key in ROW_KEYS
    }
    batch_ids = np.concatenate(
        [np.full(len(dataset["group_ids"]), batch_id) for batch_id, dataset in batches]
    )
    combined["batch_ids"] = batch_ids
    combined["robust_groups"] = batch_scenario_groups(
        batch_ids, combined["scenarios"]
    )
    combined["splits"] = np.full(len(batch_ids), "train")
    combined["state_count"] = np.asarray(
        sum(int(dataset["state_count"]) for _, dataset in batches)
    )
    combined["generation_seconds"] = np.asarray(
        sum(float(dataset["generation_seconds"]) for _, dataset in batches)
    )
    return combined


def _generate_training_corpus(
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = args.output_dir / "training_batches"
    root.mkdir(parents=True, exist_ok=True)
    batches: list[tuple[str, dict[str, np.ndarray]]] = []
    audits: dict[str, Any] = {}
    previous_observations: list[np.ndarray] = []
    old_datasets = [
        _load_npz(path) for path in OLD_DATASETS if path.exists()
    ]
    for batch_seed in args.training_batch_seeds:
        batch_id = f"batch_{batch_seed}"
        batch_dir = root / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_path = batch_dir / "test_dataset.npz"
        dataset = (
            _load_npz(batch_path)
            if args.reuse_training_batches and batch_path.exists()
            else _generate_test_dataset(
                _batch_generation_args(args, batch_seed, batch_dir)
            )
        )
        overlaps = {
            f"old_{index}": _observation_overlap_count(
                dataset["observations"], previous["observations"]
            )
            for index, previous in enumerate(old_datasets)
        }
        overlaps.update(
            {
                f"training_{index}": _observation_overlap_count(
                    dataset["observations"], observations
                )
                for index, observations in enumerate(previous_observations)
            }
        )
        power = _batch_power(dataset, list(args.scenarios))
        expected_states = (
            len(args.source_seeds)
            * len(args.scenarios)
            * args.training_states_per_stratum
        )
        audits[batch_id] = {
            "states": int(dataset["state_count"]),
            "groups": int(len(dataset["group_ids"])),
            "rollouts": int(dataset["total_return_samples"].shape[2]),
            "power": power,
            "overlaps": overlaps,
            "checks": {
                "state_count": int(dataset["state_count"]) == expected_states,
                "rollouts": int(dataset["total_return_samples"].shape[2])
                == args.rollouts,
                "power": bool(power["passed"]),
                "overlap_zero": all(count == 0 for count in overlaps.values()),
            },
        }
        audits[batch_id]["passed"] = all(audits[batch_id]["checks"].values())
        batches.append((batch_id, dataset))
        previous_observations.append(dataset["observations"])
    combined = _combine_batches(batches)
    np.savez_compressed(args.output_dir / "training_dataset.npz", **combined)
    return combined, audits


def _evaluate_oob(
    *,
    dataset: dict[str, np.ndarray],
    oracle: np.ndarray,
    targets: dict[str, np.ndarray],
    score: np.ndarray,
    safety: np.ndarray,
    cost: np.ndarray,
    budget: np.ndarray,
) -> dict[str, Any]:
    predicted = (score > 0.0).astype(np.int64)
    scenario_metrics = scenario_classification_metrics(
        oracle, predicted, dataset["scenarios"]
    )
    batch_metrics = grouped_oracle_metrics(
        oracle, predicted, dataset["batch_ids"]
    )
    valid = oracle >= 0
    value_metrics = constrained_value_metrics(
        targets["safety_gain"][valid],
        targets["cost_delta"][valid],
        safety[valid],
        cost[valid],
    )
    metrics = oracle_classification_metrics(oracle, predicted)
    feasible = (
        float(metrics["balanced_accuracy"]) >= 0.70
        and float(metrics["engage_recall"]) >= 0.60
        and float(metrics["noop_recall"]) >= 0.65
        and _worst_scenario_recall(scenario_metrics) >= 0.60
        and all(
            float(row["noop_recall"]) >= 0.65
            for row in scenario_metrics.values()
            if int(row["noop_count"]) > 0
        )
        and minimum_group_class_recall(batch_metrics) >= 0.60
        and all(
            float(row["noop_recall"]) >= 0.65
            for row in batch_metrics.values()
            if int(row["noop_count"]) > 0
        )
        and float(value_metrics["safety_sign_accuracy"]) >= 0.70
    )
    return {
        "metrics": metrics,
        "scenario_metrics": scenario_metrics,
        "batch_metrics": batch_metrics,
        "worst_scenario_recall": _worst_scenario_recall(scenario_metrics),
        "worst_batch_recall": minimum_group_class_recall(batch_metrics),
        "value_metrics": value_metrics,
        "budget_mean": float(np.mean(budget)),
        "budget_std": float(np.std(budget)),
        "feasible": feasible,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)
    training_path = args.output_dir / "training_dataset.npz"
    audit_path = args.output_dir / "training_batch_audit.json"
    if args.reuse_training_batches and training_path.exists() and audit_path.exists():
        training_dataset = _load_npz(training_path)
        with audit_path.open(encoding="utf-8") as handle:
            batch_audits = json.load(handle)
    else:
        training_dataset, batch_audits = _generate_training_corpus(args)
        with audit_path.open("w", encoding="utf-8") as handle:
            json.dump(batch_audits, handle, indent=2, ensure_ascii=False)

    test_path = args.output_dir / "test_dataset.npz"
    test_dataset = (
        _load_npz(test_path)
        if args.reuse_test_dataset and test_path.exists()
        else _generate_test_dataset(args)
    )
    train_indices = np.arange(len(training_dataset["group_ids"]))
    fold_assignments = leave_one_batch_out_folds(training_dataset["batch_ids"])
    fold_count = len(np.unique(fold_assignments))

    environment = AirDefenseResourceAssignmentEnvV1()
    layout = AirDefenseV1ObservationLayout.infer(
        environment.observation_space, environment.action_space
    )
    environment.close()

    training_targets = engagement_delta_targets(
        _components(training_dataset, train_indices)
    )
    training_oracle = safety_resource_oracle(
        _components(training_dataset, train_indices)
    )["labels"]
    cost_reliability = paired_delta_reliability(
        training_targets["cost_delta_samples"]
    )
    test_indices = np.arange(len(test_dataset["group_ids"]))
    test_targets = engagement_delta_targets(_components(test_dataset, test_indices))
    test_oracle = safety_resource_oracle(
        _components(test_dataset, test_indices)
    )["labels"]

    margin_logits: dict[int, np.ndarray] = {}
    test_margin_logits: dict[int, np.ndarray] = {}
    test_regression_logits: dict[int, np.ndarray] = {}
    for model_seed in args.model_seeds:
        margin_model, _ = _load_critic(
            args.critic_dir / f"balanced_bce_margin_seed{model_seed}.pt",
            layout,
            args.device,
        )
        regression_model, regression_checkpoint = _load_critic(
            args.critic_dir / f"risk_regression_seed{model_seed}.pt",
            layout,
            args.device,
        )
        values = _predict(
            margin_model, training_dataset, train_indices, args.device
        )
        margin_logits[model_seed] = values[:, 1] - values[:, 0]
        test_values = _predict(
            margin_model, test_dataset, test_indices, args.device
        )
        test_margin_logits[model_seed] = test_values[:, 1] - test_values[:, 0]
        regression_values = _predict(
            regression_model,
            test_dataset,
            test_indices,
            args.device,
            regression_checkpoint["normalization"],
        )
        test_regression_logits[model_seed] = (
            regression_values[:, 1] - regression_values[:, 0]
        )

    oob_results: dict[str, dict[str, Any]] = {}
    best_epochs: dict[tuple[int, str], list[int]] = {}
    curve_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for model_seed in args.model_seeds:
        oob_results[str(model_seed)] = {}
        for objective in OBJECTIVES:
            oob_score = np.full(len(train_indices), np.nan, dtype=np.float32)
            oob_safety = np.full(len(train_indices), np.nan, dtype=np.float32)
            oob_cost = np.full(len(train_indices), np.nan, dtype=np.float32)
            oob_budget = np.full(len(train_indices), np.nan, dtype=np.float32)
            epochs: list[int] = []
            for fold in range(fold_count):
                fit = train_indices[fold_assignments != fold]
                validation = train_indices[fold_assignments == fold]
                _, prediction, _, _, best_epoch, curves = _fit_fold(
                    dataset=training_dataset,
                    fit=fit,
                    validation=validation,
                    safety_targets=training_targets["safety_gain"],
                    cost_targets=training_targets["cost_delta"],
                    oracle_labels=training_oracle,
                    margin_logits=margin_logits[model_seed],
                    layout=layout,
                    mode="state_budget",
                    seed=model_seed,
                    fold_index=fold,
                    args=args,
                    objective=objective,
                    cost_reliability=cost_reliability,
                )
                oob_score[validation] = prediction["score"]
                oob_safety[validation] = prediction["safety_gain"]
                oob_cost[validation] = prediction["cost_delta"]
                oob_budget[validation] = prediction["budget_multiplier"]
                epochs.append(best_epoch)
                curve_rows.extend(curves)
            if np.any(~np.isfinite(oob_score)):
                raise RuntimeError("Leave-one-batch-out left missing predictions")
            result = _evaluate_oob(
                dataset=training_dataset,
                oracle=training_oracle,
                targets=training_targets,
                score=oob_score,
                safety=oob_safety,
                cost=oob_cost,
                budget=oob_budget,
            )
            result["best_epochs"] = epochs
            oob_results[str(model_seed)][objective] = result
            best_epochs[(model_seed, objective)] = epochs
            for index in train_indices:
                prediction_rows.append(
                    {
                        "model_seed": model_seed,
                        "objective": objective,
                        "batch_id": str(training_dataset["batch_ids"][index]),
                        "scenario": str(training_dataset["scenarios"][index]),
                        "group_id": str(training_dataset["group_ids"][index]),
                        "oracle_label": int(training_oracle[index]),
                        "score": float(oob_score[index]),
                        "safety_prediction": float(oob_safety[index]),
                        "cost_prediction": float(oob_cost[index]),
                        "budget_multiplier": float(oob_budget[index]),
                    }
                )
    _write_csv(args.output_dir / "oob_curves.csv", curve_rows)
    _write_csv(args.output_dir / "oob_predictions.csv", prediction_rows)

    complexity = {
        "standard": 2.0,
        "scenario_robust": 1.0,
        "scenario_robust_reliable_cost": 0.0,
    }
    ranking: list[tuple[tuple[float, float, float, float, float], str]] = []
    for objective in OBJECTIVES:
        rows = [oob_results[str(seed)][objective] for seed in args.model_seeds]
        worst = [
            min(row["worst_batch_recall"], row["worst_scenario_recall"])
            for row in rows
        ]
        ranking.append(
            (
                (
                    float(sum(bool(row["feasible"]) for row in rows)),
                    float(np.mean(worst)),
                    float(
                        np.mean(
                            [row["metrics"]["balanced_accuracy"] for row in rows]
                        )
                    ),
                    float(
                        np.nanmean(
                            [
                                row["value_metrics"]["cost_correlation"]
                                for row in rows
                            ]
                        )
                    ),
                    complexity[objective],
                ),
                objective,
            )
        )
    ranking.sort(reverse=True)
    selected_objective = ranking[0][1]
    selected_oob_passes = sum(
        bool(oob_results[str(seed)][selected_objective]["feasible"])
        for seed in args.model_seeds
    )

    model_results: dict[str, Any] = {}
    per_seed_gates: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    passed_seed_count = 0
    for model_seed in args.model_seeds:
        final_epochs = max(
            1,
            int(np.median(best_epochs[(model_seed, selected_objective)])) + 1,
        )
        model, scales, margin_scale = _fit_final(
            dataset=training_dataset,
            fit=train_indices,
            safety_targets=training_targets["safety_gain"],
            cost_targets=training_targets["cost_delta"],
            oracle_labels=training_oracle,
            margin_logits=margin_logits[model_seed],
            layout=layout,
            mode="state_budget",
            seed=model_seed,
            epochs=final_epochs,
            args=args,
            objective=selected_objective,
            cost_reliability=cost_reliability,
        )
        started = perf_counter()
        candidate_prediction = _predict_value(
            model,
            test_dataset,
            test_indices,
            args.device,
            test_margin_logits[model_seed],
            scales,
            margin_scale,
        )
        inference_seconds = perf_counter() - started
        candidate = _evaluate_value_model(
            prediction=candidate_prediction,
            oracle=test_oracle,
            scenarios=test_dataset["scenarios"],
            targets=test_targets,
        )
        candidate["inference_seconds"] = inference_seconds
        candidate["final_epochs"] = final_epochs

        previous_model, previous_checkpoint = _load_previous_model(
            args.previous_dir / "models" / f"state_budget_seed{model_seed}.pt",
            layout,
            args.device,
        )
        previous_prediction = _predict_value(
            previous_model,
            test_dataset,
            test_indices,
            args.device,
            test_margin_logits[model_seed],
            previous_checkpoint["scales"],
            float(previous_checkpoint["margin_scale"]),
        )
        previous = _evaluate_value_model(
            prediction=previous_prediction,
            oracle=test_oracle,
            scenarios=test_dataset["scenarios"],
            targets=test_targets,
        )
        regression_predictions = (
            test_regression_logits[model_seed] > 0.0
        ).astype(np.int64)
        regression = {
            "metrics": oracle_classification_metrics(
                test_oracle, regression_predictions
            ),
            "scenario_metrics": scenario_classification_metrics(
                test_oracle,
                regression_predictions,
                test_dataset["scenarios"],
            ),
        }
        checks = _model_gate(candidate, regression)
        checks["inference_faster_than_rollouts"] = (
            inference_seconds < float(test_dataset["generation_seconds"])
        )
        passed = all(checks.values())
        passed_seed_count += int(passed)
        per_seed_gates[str(model_seed)] = {**checks, "passed": passed}
        model_results[str(model_seed)] = {
            "frozen_state_budget": previous,
            "risk_regression": regression,
            "candidate": candidate,
        }
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_signature": model.signature(),
                "selected_objective": selected_objective,
                "scales": scales,
                "margin_scale": margin_scale,
                "final_epochs": final_epochs,
            },
            args.output_dir
            / "models"
            / f"{selected_objective}_seed{model_seed}.pt",
        )
        for method, result in (
            ("frozen_state_budget", previous),
            ("risk_regression", regression),
            (selected_objective, candidate),
        ):
            metric_rows.append(
                {"model_seed": model_seed, "method": method, **result["metrics"]}
            )
        for index in test_indices:
            diagnostics.append(
                {
                    "model_seed": model_seed,
                    "group_id": str(test_dataset["group_ids"][index]),
                    "scenario": str(test_dataset["scenarios"][index]),
                    "oracle_label": int(test_oracle[index]),
                    "safety_target": float(test_targets["safety_gain"][index]),
                    "safety_prediction": float(
                        candidate_prediction["safety_gain"][index]
                    ),
                    "cost_target": float(test_targets["cost_delta"][index]),
                    "cost_prediction": float(
                        candidate_prediction["cost_delta"][index]
                    ),
                    "budget_multiplier": float(
                        candidate_prediction["budget_multiplier"][index]
                    ),
                    "score": float(candidate_prediction["score"][index]),
                }
            )
    _write_csv(args.output_dir / "model_metrics.csv", metric_rows)
    _write_csv(args.output_dir / "test_group_diagnostics.csv", diagnostics)

    candidate_worst = float(
        np.mean(
            [
                model_results[str(seed)]["candidate"]["worst_scenario_recall"]
                for seed in args.model_seeds
            ]
        )
    )
    previous_worst = float(
        np.mean(
            [
                model_results[str(seed)]["frozen_state_budget"][
                    "worst_scenario_recall"
                ]
                for seed in args.model_seeds
            ]
        )
    )
    candidate_ba = float(
        np.mean(
            [
                model_results[str(seed)]["candidate"]["metrics"][
                    "balanced_accuracy"
                ]
                for seed in args.model_seeds
            ]
        )
    )
    previous_ba = float(
        np.mean(
            [
                model_results[str(seed)]["frozen_state_budget"]["metrics"][
                    "balanced_accuracy"
                ]
                for seed in args.model_seeds
            ]
        )
    )

    final_overlaps: dict[str, int] = {}
    overlap_paths = [path for path in OLD_DATASETS if path.exists()]
    overlap_paths.append(args.previous_dir / "test_dataset.npz")
    for path in overlap_paths:
        if not path.exists():
            continue
        previous_dataset = _load_npz(path)
        final_overlaps[str(path)] = _observation_overlap_count(
            test_dataset["observations"], previous_dataset["observations"]
        )
    final_overlaps["training_dataset"] = _observation_overlap_count(
        test_dataset["observations"], training_dataset["observations"]
    )
    reconstructed = (
        test_dataset["operational_return_samples"]
        - test_dataset["resource_cost_samples"]
        - 30.0 * test_dataset["damage_samples"]
    )
    reconstruction_error = float(
        np.max(np.abs(test_dataset["total_return_samples"] - reconstructed))
    )
    expected_test_states = (
        len(args.source_seeds) * len(args.scenarios) * args.states_per_stratum
    )
    valid = test_oracle >= 0
    scenario_counts = {
        str(scenario): int(np.sum(valid & (test_dataset["scenarios"] == scenario)))
        for scenario in np.unique(test_dataset["scenarios"])
    }
    engage_scenarios = sum(
        bool(
            np.any(
                (test_oracle == 1) & (test_dataset["scenarios"] == scenario)
            )
        )
        for scenario in np.unique(test_dataset["scenarios"])
    )
    data_checks = {
        "training_batches": all(
            bool(audit["passed"]) for audit in batch_audits.values()
        ),
        "test_state_count": int(test_dataset["state_count"])
        == expected_test_states,
        "test_rollouts": int(test_dataset["total_return_samples"].shape[2])
        == args.rollouts,
        "test_overlap_zero": all(count == 0 for count in final_overlaps.values()),
        "return_reconstruction": reconstruction_error <= 1e-4,
    }
    power_checks = {
        "valid_count": int(np.sum(valid)) >= 40,
        "engage_count": int(np.sum(test_oracle == 1)) >= 10,
        "noop_count": int(np.sum(test_oracle == 0)) >= 10,
        "scenario_counts": all(count >= 8 for count in scenario_counts.values()),
        "engage_scenarios": engage_scenarios >= 2,
    }
    stage_checks = {
        "oob_passed_seed_count": selected_oob_passes >= 2,
        "test_passed_seed_count": passed_seed_count >= 2,
        "worst_scenario_noninferior": candidate_worst >= previous_worst,
        "balanced_accuracy_noninferior": candidate_ba >= previous_ba - 0.02,
    }
    data_integrity = all(data_checks.values())
    power_sufficient = all(power_checks.values())
    stage_passed = data_integrity and power_sufficient and all(stage_checks.values())
    summary = {
        "schema_version": 1,
        "training_batches": batch_audits,
        "data_audit": {
            "checks": data_checks,
            "passed": data_integrity,
            "final_observation_overlaps": final_overlaps,
            "return_reconstruction_max_error": reconstruction_error,
        },
        "dataset": {
            "training_states": int(training_dataset["state_count"]),
            "training_groups": int(len(training_dataset["group_ids"])),
            "training_batch_count": len(np.unique(training_dataset["batch_ids"])),
            "test_states": int(test_dataset["state_count"]),
            "test_groups": int(len(test_dataset["group_ids"])),
            "rollouts": int(test_dataset["total_return_samples"].shape[2]),
            "generation_seconds": float(
                training_dataset["generation_seconds"]
                + test_dataset["generation_seconds"]
            ),
        },
        "power": {
            "checks": power_checks,
            "passed": power_sufficient,
            "valid_count": int(np.sum(valid)),
            "engage_count": int(np.sum(test_oracle == 1)),
            "noop_count": int(np.sum(test_oracle == 0)),
            "scenario_counts": scenario_counts,
            "engage_scenario_count": int(engage_scenarios),
        },
        "leave_one_batch_out": {
            "fold_count": fold_count,
            "results": oob_results,
            "selected_objective": selected_objective,
            "selected_passed_seed_count": selected_oob_passes,
        },
        "model_results": model_results,
        "model_gate": {
            "per_seed": per_seed_gates,
            "passed_seed_count": passed_seed_count,
            "required_passed_seed_count": 2,
        },
        "stage_comparison": {
            "candidate_mean_worst_scenario_recall": candidate_worst,
            "previous_mean_worst_scenario_recall": previous_worst,
            "candidate_mean_balanced_accuracy": candidate_ba,
            "previous_mean_balanced_accuracy": previous_ba,
            "checks": stage_checks,
        },
        "task14_multibatch_leave_one_out_passed": stage_passed,
        "resume_mch_ppo": stage_passed,
        "enter_gnn": False,
    }
    with (args.output_dir / "gate_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    config = {
        "schema_version": 1,
        "training_batch_seeds": list(args.training_batch_seeds),
        "training_states_per_stratum": args.training_states_per_stratum,
        "training_episodes_per_stratum": args.training_episodes_per_stratum,
        "eval_seed": args.eval_seed,
        "states_per_stratum": args.states_per_stratum,
        "episodes_per_stratum": args.episodes_per_stratum,
        "source_seeds": list(args.source_seeds),
        "scenarios": list(args.scenarios),
        "rollouts": args.rollouts,
        "model_seeds": list(args.model_seeds),
        "objectives": list(OBJECTIVES),
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "selected_objective": selected_objective,
    }
    with (args.output_dir / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
