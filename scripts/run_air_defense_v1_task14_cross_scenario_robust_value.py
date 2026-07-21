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
    constrained_value_metrics,
    engagement_delta_targets,
    oracle_classification_metrics,
    paired_delta_reliability,
    safety_resource_oracle,
    scenario_classification_metrics,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    StateConditionedEngagementValue,
    StateConditionedEngagementValueConfig,
)
from scripts.run_air_defense_v1_task14_engagement_calibration import (
    DEFAULT_CRITIC_DIR,
    OLD_DATASETS,
    _load_critic,
    _load_npz,
    _predict,
)
from scripts.run_air_defense_v1_task14_engagement_utility import (
    _components,
    _observation_overlap_count,
)
from scripts.run_air_defense_v1_task14_q_critic import (
    DEFAULT_MODEL_DIR,
    SCENARIOS,
    _write_csv,
)
from scripts.run_air_defense_v1_task14_state_conditioned_value import (
    DEFAULT_CALIBRATION_DATASET,
    FOLDS,
    _fit_final,
    _fit_fold,
    _generate_test_dataset,
    _grouped_folds,
    _model_gate,
    _predict_value,
)


DEFAULT_PREVIOUS_DIR = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task14_state_conditioned_value"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_cross_scenario_robust_value"
)
OBJECTIVES = (
    "standard",
    "scenario_robust",
    "scenario_robust_reliable_cost",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-scenario robust engagement value diagnostics."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC_DIR)
    parser.add_argument(
        "--training-dataset", type=Path, default=DEFAULT_CALIBRATION_DATASET
    )
    parser.add_argument("--previous-dir", type=Path, default=DEFAULT_PREVIOUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-seeds", nargs="+", type=int, default=(8, 10))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--episodes-per-stratum", type=int, default=30)
    parser.add_argument("--states-per-stratum", type=int, default=12)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--eval-seed", type=int, default=641_000)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=(20, 21, 22))
    parser.add_argument("--fold-seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-test-dataset", action="store_true")
    return parser.parse_args()


def _worst_scenario_recall(
    scenario_metrics: dict[str, dict[str, float | int]],
) -> float:
    recalls = [
        float(row[name])
        for row in scenario_metrics.values()
        for name, count_name in (
            ("engage_recall", "engage_count"),
            ("noop_recall", "noop_count"),
        )
        if int(row[count_name]) > 0
    ]
    return min(recalls) if recalls else float("nan")


def _evaluate_value_model(
    *,
    prediction: dict[str, np.ndarray],
    oracle: np.ndarray,
    scenarios: np.ndarray,
    targets: dict[str, np.ndarray],
) -> dict[str, Any]:
    predicted = (prediction["score"] > 0.0).astype(np.int64)
    valid = oracle >= 0
    scenario_metrics = scenario_classification_metrics(
        oracle, predicted, scenarios
    )
    return {
        "metrics": oracle_classification_metrics(oracle, predicted),
        "scenario_metrics": scenario_metrics,
        "worst_scenario_recall": _worst_scenario_recall(scenario_metrics),
        "value_metrics": constrained_value_metrics(
            targets["safety_gain"][valid],
            targets["cost_delta"][valid],
            prediction["safety_gain"][valid],
            prediction["cost_delta"][valid],
        ),
        "budget": {
            "mean": float(np.mean(prediction["budget_multiplier"])),
            "std": float(np.std(prediction["budget_multiplier"])),
            "min": float(np.min(prediction["budget_multiplier"])),
            "max": float(np.max(prediction["budget_multiplier"])),
        },
    }


def _load_previous_model(
    path: Path,
    layout: AirDefenseV1ObservationLayout,
    device: str,
) -> tuple[StateConditionedEngagementValue, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = StateConditionedEngagementValue(
        layout,
        StateConditionedEngagementValueConfig(
            budget_mode=str(checkpoint["selected_mode"])
        ),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)
    test_path = args.output_dir / "test_dataset.npz"
    test_dataset = (
        _load_npz(test_path)
        if args.reuse_test_dataset
        else _generate_test_dataset(args)
    )
    training_dataset = _load_npz(args.training_dataset)
    train_indices = np.flatnonzero(training_dataset["splits"] != "test")
    fold_assignments = _grouped_folds(
        training_dataset,
        train_indices,
        folds=FOLDS,
        seed=args.fold_seed,
    )

    environment = AirDefenseResourceAssignmentEnvV1()
    layout = AirDefenseV1ObservationLayout.infer(
        environment.observation_space, environment.action_space
    )
    environment.close()

    all_training = np.arange(len(training_dataset["group_ids"]))
    training_targets = engagement_delta_targets(
        _components(training_dataset, all_training)
    )
    training_oracle = safety_resource_oracle(
        _components(training_dataset, all_training)
    )["labels"]
    cost_reliability = paired_delta_reliability(
        training_targets["cost_delta_samples"]
    )
    test_indices = np.arange(len(test_dataset["group_ids"]))
    test_targets = engagement_delta_targets(_components(test_dataset, test_indices))
    test_oracle_details = safety_resource_oracle(
        _components(test_dataset, test_indices)
    )
    test_oracle = test_oracle_details["labels"]

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
            margin_model, training_dataset, all_training, args.device
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

    crossfit_results: dict[str, dict[str, Any]] = {}
    crossfit_epochs: dict[tuple[int, str], list[int]] = {}
    curve_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    for model_seed in args.model_seeds:
        crossfit_results[str(model_seed)] = {}
        for objective in OBJECTIVES:
            oof_score = np.full(len(train_indices), np.nan, dtype=np.float32)
            oof_safety = np.full(len(train_indices), np.nan, dtype=np.float32)
            oof_cost = np.full(len(train_indices), np.nan, dtype=np.float32)
            oof_budget = np.full(len(train_indices), np.nan, dtype=np.float32)
            best_epochs: list[int] = []
            for fold in range(FOLDS):
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
                local = np.flatnonzero(fold_assignments == fold)
                oof_score[local] = prediction["score"]
                oof_safety[local] = prediction["safety_gain"]
                oof_cost[local] = prediction["cost_delta"]
                oof_budget[local] = prediction["budget_multiplier"]
                best_epochs.append(best_epoch)
                curve_rows.extend(curves)
            oracle = training_oracle[train_indices]
            scenarios = training_dataset["scenarios"][train_indices]
            predicted = (oof_score > 0.0).astype(np.int64)
            metrics = oracle_classification_metrics(oracle, predicted)
            by_scenario = scenario_classification_metrics(
                oracle, predicted, scenarios
            )
            valid = oracle >= 0
            values = constrained_value_metrics(
                training_targets["safety_gain"][train_indices][valid],
                training_targets["cost_delta"][train_indices][valid],
                oof_safety[valid],
                oof_cost[valid],
            )
            feasible = (
                float(metrics["balanced_accuracy"]) >= 0.70
                and float(metrics["engage_recall"]) >= 0.60
                and float(metrics["noop_recall"]) >= 0.65
                and _worst_scenario_recall(by_scenario) >= 0.60
                and all(
                    float(row["noop_recall"]) >= 0.65
                    for row in by_scenario.values()
                    if int(row["noop_count"]) > 0
                )
            )
            crossfit_results[str(model_seed)][objective] = {
                "metrics": metrics,
                "scenario_metrics": by_scenario,
                "worst_scenario_recall": _worst_scenario_recall(by_scenario),
                "value_metrics": values,
                "feasible": feasible,
                "best_epochs": best_epochs,
                "budget_mean": float(np.mean(oof_budget)),
                "budget_std": float(np.std(oof_budget)),
            }
            crossfit_epochs[(model_seed, objective)] = best_epochs
            for local, dataset_index in enumerate(train_indices):
                oof_rows.append(
                    {
                        "model_seed": model_seed,
                        "objective": objective,
                        "fold": int(fold_assignments[local]),
                        "group_id": str(training_dataset["group_ids"][dataset_index]),
                        "scenario": str(training_dataset["scenarios"][dataset_index]),
                        "oracle_label": int(oracle[local]),
                        "score": float(oof_score[local]),
                        "safety_prediction": float(oof_safety[local]),
                        "cost_prediction": float(oof_cost[local]),
                        "budget_multiplier": float(oof_budget[local]),
                    }
                )
    _write_csv(args.output_dir / "crossfit_curves.csv", curve_rows)
    _write_csv(args.output_dir / "crossfit_predictions.csv", oof_rows)

    complexity = {
        "standard": 2.0,
        "scenario_robust": 1.0,
        "scenario_robust_reliable_cost": 0.0,
    }
    ranked: list[tuple[tuple[float, float, float, float, float], str]] = []
    for objective in OBJECTIVES:
        rows = [
            crossfit_results[str(seed)][objective] for seed in args.model_seeds
        ]
        ranked.append(
            (
                (
                    float(sum(bool(row["feasible"]) for row in rows)),
                    float(np.mean([row["worst_scenario_recall"] for row in rows])),
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
    ranked.sort(reverse=True)
    selected_objective = ranked[0][1]

    model_results: dict[str, Any] = {}
    per_seed_gates: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    passed_seed_count = 0
    for model_seed in args.model_seeds:
        epochs = max(
            1,
            int(
                np.median(crossfit_epochs[(model_seed, selected_objective)])
            )
            + 1,
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
            epochs=epochs,
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
        candidate["final_epochs"] = epochs

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
                "final_epochs": epochs,
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
                    "cost_reliability": float(
                        paired_delta_reliability(
                            test_targets["cost_delta_samples"]
                        )[index]
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
    candidate_cost_corr = float(
        np.nanmean(
            [
                model_results[str(seed)]["candidate"]["value_metrics"][
                    "cost_correlation"
                ]
                for seed in args.model_seeds
            ]
        )
    )
    previous_cost_corr = float(
        np.nanmean(
            [
                model_results[str(seed)]["frozen_state_budget"]["value_metrics"][
                    "cost_correlation"
                ]
                for seed in args.model_seeds
            ]
        )
    )
    stage_checks = {
        "passed_seed_count": passed_seed_count >= 2,
        "worst_scenario_noninferior": candidate_worst >= previous_worst,
        "balanced_accuracy_noninferior": candidate_ba >= previous_ba - 0.02,
        "cost_correlation_noninferior": candidate_cost_corr >= previous_cost_corr,
    }

    overlap_paths = (*OLD_DATASETS, args.previous_dir / "test_dataset.npz")
    overlaps: dict[str, int] = {}
    for path in overlap_paths:
        if not path.exists():
            continue
        previous_dataset = _load_npz(path)
        overlaps[str(path)] = _observation_overlap_count(
            test_dataset["observations"], previous_dataset["observations"]
        )
    reconstructed = (
        test_dataset["operational_return_samples"]
        - test_dataset["resource_cost_samples"]
        - 30.0 * test_dataset["damage_samples"]
    )
    reconstruction_error = float(
        np.max(np.abs(test_dataset["total_return_samples"] - reconstructed))
    )
    expected_states = (
        len(args.source_seeds) * len(args.scenarios) * args.states_per_stratum
    )
    data_checks = {
        "state_count": int(test_dataset["state_count"]) == expected_states,
        "rollout_count": int(test_dataset["total_return_samples"].shape[2])
        == args.rollouts,
        "previous_observation_overlap_zero": all(
            count == 0 for count in overlaps.values()
        ),
        "return_reconstruction": reconstruction_error <= 1e-4,
    }
    valid = test_oracle >= 0
    engage_count = int(np.sum(test_oracle == 1))
    noop_count = int(np.sum(test_oracle == 0))
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
    power_checks = {
        "valid_count": int(np.sum(valid)) >= 40,
        "engage_count": engage_count >= 10,
        "noop_count": noop_count >= 10,
        "scenario_counts": all(count >= 8 for count in scenario_counts.values()),
        "engage_scenarios": engage_scenarios >= 2,
    }
    data_integrity = all(data_checks.values())
    power_sufficient = all(power_checks.values())
    stage_passed = data_integrity and power_sufficient and all(stage_checks.values())
    summary = {
        "schema_version": 1,
        "data_audit": {
            "checks": data_checks,
            "passed": data_integrity,
            "previous_observation_overlaps": overlaps,
            "return_reconstruction_max_error": reconstruction_error,
        },
        "dataset": {
            "training_non_test_groups": int(len(train_indices)),
            "test_states": int(test_dataset["state_count"]),
            "test_groups": int(len(test_indices)),
            "rollouts": int(test_dataset["total_return_samples"].shape[2]),
            "generation_seconds": float(test_dataset["generation_seconds"]),
        },
        "power": {
            "checks": power_checks,
            "passed": power_sufficient,
            "valid_count": int(np.sum(valid)),
            "engage_count": engage_count,
            "noop_count": noop_count,
            "scenario_counts": scenario_counts,
            "engage_scenario_count": int(engage_scenarios),
        },
        "crossfit": {
            "fold_count": FOLDS,
            "fold_seed": args.fold_seed,
            "results": crossfit_results,
            "selected_objective": selected_objective,
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
            "candidate_mean_cost_correlation": candidate_cost_corr,
            "previous_mean_cost_correlation": previous_cost_corr,
            "checks": stage_checks,
        },
        "task14_cross_scenario_robust_value_passed": stage_passed,
        "resume_mch_ppo": stage_passed,
        "enter_gnn": False,
    }
    with (args.output_dir / "gate_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    config = {
        "schema_version": 1,
        "training_dataset": str(args.training_dataset.resolve()),
        "previous_dir": str(args.previous_dir.resolve()),
        "source_seeds": list(args.source_seeds),
        "scenarios": list(args.scenarios),
        "episodes_per_stratum": args.episodes_per_stratum,
        "states_per_stratum": args.states_per_stratum,
        "rollouts": args.rollouts,
        "gamma": args.gamma,
        "eval_seed": args.eval_seed,
        "model_seeds": list(args.model_seeds),
        "objectives": list(OBJECTIVES),
        "fold_count": FOLDS,
        "fold_seed": args.fold_seed,
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
