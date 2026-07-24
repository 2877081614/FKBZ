from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.common import (
    ParetoRecallConstraints,
    assemble_calibration_features,
    calibrated_operating_point,
    calibration_candidates,
    confirmation_power,
    constrained_value_metrics,
    engagement_delta_targets,
    fit_cross_batch_calibrator,
    safety_resource_oracle,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import AirDefenseV1ObservationLayout
from scripts.run_air_defense_v1_task14_engagement_calibration import (
    DEFAULT_CRITIC_DIR,
    _load_critic,
    _load_npz,
    _predict,
)
from scripts.run_air_defense_v1_task14_engagement_utility import (
    _components,
    _observation_overlap_count,
)
from scripts.run_air_defense_v1_task14_independent_confirmation import (
    DEFAULT_SOURCE_DIR,
    _generation_args,
    _historical_dataset_paths,
    _load_value_model,
    _sha256,
)
from scripts.run_air_defense_v1_task14_q_critic import (
    DEFAULT_MODEL_DIR,
    SCENARIOS,
    _write_csv,
)
from scripts.run_air_defense_v1_task14_state_conditioned_value import (
    _generate_test_dataset,
    _predict_value,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_cross_batch_calibration"
)
FROZEN_OBJECTIVE = "scenario_robust_reliable_cost"
SCENARIO_LEVELS = tuple(str(value) for value in SCENARIOS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-batch probability and uncertainty calibration."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--source-model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-seeds", nargs="+", type=int, default=(8, 10))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--episodes-per-stratum", type=int, default=30)
    parser.add_argument("--states-per-stratum", type=int, default=12)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--eval-seed", type=int, default=941_000)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=(20, 21, 22))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-test-dataset", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _oob_records(
    rows: list[dict[str, str]], model_seed: int
) -> dict[str, np.ndarray]:
    selected = sorted(
        (
            row
            for row in rows
            if row["objective"] == FROZEN_OBJECTIVE
            and int(row["model_seed"]) == model_seed
        ),
        key=lambda row: row["group_id"],
    )
    if not selected:
        raise ValueError(f"Missing OOB records for seed {model_seed}")
    return {
        "group_ids": np.asarray([row["group_id"] for row in selected]),
        "batch_ids": np.asarray([row["batch_id"] for row in selected]),
        "scenarios": np.asarray([row["scenario"] for row in selected]),
        "labels": np.asarray(
            [int(row["oracle_label"]) for row in selected], dtype=np.int64
        ),
        "score": np.asarray(
            [float(row["score"]) for row in selected], dtype=np.float64
        ),
        "safety": np.asarray(
            [float(row["safety_prediction"]) for row in selected],
            dtype=np.float64,
        ),
        "cost": np.asarray(
            [float(row["cost_prediction"]) for row in selected],
            dtype=np.float64,
        ),
        "budget": np.asarray(
            [float(row["budget_multiplier"]) for row in selected],
            dtype=np.float64,
        ),
    }


def _features(
    records: dict[str, np.ndarray], feature_set: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    return assemble_calibration_features(
        records["score"],
        records["safety"],
        records["cost"],
        records["budget"],
        records["scenarios"],
        feature_set=feature_set,
        scenario_levels=SCENARIO_LEVELS,
    )


def _cross_fit_candidate(
    records: dict[str, np.ndarray],
    config: Any,
    safety_sign_accuracy: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    features, names = _features(records, config.feature_set)
    batches = records["batch_ids"]
    labels = records["labels"]
    output = {
        "probability": np.full(len(labels), np.nan),
        "calibrated_logit": np.full(len(labels), np.nan),
        "prediction_se": np.full(len(labels), np.nan),
        "conservative_logit": np.full(len(labels), np.nan),
        "predicted_label": np.full(len(labels), -1, dtype=np.int64),
    }
    fold_rows: list[dict[str, object]] = []
    for held_batch in np.unique(batches):
        fit = (batches != held_batch) & (labels >= 0)
        validation = batches == held_batch
        model = fit_cross_batch_calibrator(
            features[fit],
            names,
            labels[fit],
            batches[fit],
            records["scenarios"][fit],
            config,
        )
        prediction = model.predict(features[validation])
        for key in output:
            output[key][validation] = prediction[key]
        fold_rows.append(
            {
                "held_batch": str(held_batch),
                "fit_count": int(np.sum(fit)),
                "validation_count": int(np.sum(validation)),
                "hessian_condition": model.hessian_condition,
            }
        )
    continuous_keys = tuple(key for key in output if key != "predicted_label")
    if any(np.any(~np.isfinite(output[key])) for key in continuous_keys):
        raise RuntimeError("Cross-batch calibration left missing predictions")
    point = calibrated_operating_point(
        output,
        labels,
        batches,
        records["scenarios"],
        safety_sign_accuracy=safety_sign_accuracy,
    )
    return point, fold_rows


def _flatten(point: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in point.items() if key != "checks"
    } | {
        f"check_{key}": value for key, value in point["checks"].items()
    }


def _candidate_ranking(
    results: dict[str, dict[str, dict[str, object]]],
    model_seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    simplicity = {
        "score_platt": 4,
        "value_platt": 3,
        "value_lcb_050": 2,
        "value_lcb_100": 1,
    }
    rows: list[dict[str, object]] = []
    for candidate in results:
        seed_results = [results[candidate][str(seed)] for seed in model_seeds]
        rows.append(
            {
                "candidate": candidate,
                "feasible_seed_count": sum(
                    bool(row["feasible"]) for row in seed_results
                ),
                "minimum_constraint_margin": min(
                    float(row["minimum_constraint_margin"])
                    for row in seed_results
                ),
                "mean_balanced_accuracy": float(
                    np.mean(
                        [float(row["balanced_accuracy"]) for row in seed_results]
                    )
                ),
                "mean_brier_score": float(
                    np.mean([float(row["brier_score"]) for row in seed_results])
                ),
                "simplicity": simplicity[candidate],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            int(row["feasible_seed_count"]),
            float(row["minimum_constraint_margin"]),
            float(row["mean_balanced_accuracy"]),
            -float(row["mean_brier_score"]),
            int(row["simplicity"]),
        ),
        reverse=True,
    )


def _value_records(
    prediction: dict[str, np.ndarray], scenarios: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "score": prediction["score"],
        "safety": prediction["safety_gain"],
        "cost": prediction["cost_delta"],
        "budget": prediction["budget_multiplier"],
        "scenarios": scenarios,
    }


def _write_early_exit(
    args: argparse.Namespace,
    summary: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    with (args.output_dir / "gate_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (args.output_dir / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oob_path = args.source_dir / "oob_predictions.csv"
    source_gate_path = args.source_dir / "gate_summary.json"
    source_config_path = args.source_dir / "experiment_config.json"
    with source_gate_path.open(encoding="utf-8") as handle:
        source_gate = json.load(handle)
    with source_config_path.open(encoding="utf-8") as handle:
        source_config = json.load(handle)
    model_seeds = tuple(int(seed) for seed in args.model_seeds)
    candidates = calibration_candidates()
    protocol = {
        "schema_version": 1,
        "objective": FROZEN_OBJECTIVE,
        "candidate_configs": [candidate.signature() for candidate in candidates],
        "scenario_levels": list(SCENARIO_LEVELS),
        "outer_validation": "leave_one_batch_out",
        "selection_order": [
            "feasible_seed_count",
            "minimum_constraint_margin",
            "mean_balanced_accuracy",
            "mean_brier_score",
            "simplicity",
        ],
        "minimum_oob_seed_count": 2,
        "eval_seed": args.eval_seed,
        "source_seeds": list(args.source_seeds),
        "scenarios": list(args.scenarios),
        "states_per_stratum": args.states_per_stratum,
        "episodes_per_stratum": args.episodes_per_stratum,
        "rollouts": args.rollouts,
        "gamma": args.gamma,
        "model_seeds": list(model_seeds),
        "confirmation_batches_used_for_fitting": 0,
        "test_threshold_refit": False,
    }

    rows = _read_csv(oob_path)
    oob_results: dict[str, dict[str, dict[str, object]]] = {
        candidate.name: {} for candidate in candidates
    }
    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    oob_records_by_seed: dict[int, dict[str, np.ndarray]] = {}
    for model_seed in model_seeds:
        records = _oob_records(rows, model_seed)
        oob_records_by_seed[model_seed] = records
        safety_sign = float(
            source_gate["leave_one_batch_out"]["results"][str(model_seed)][
                FROZEN_OBJECTIVE
            ]["value_metrics"]["safety_sign_accuracy"]
        )
        for config in candidates:
            point, folds = _cross_fit_candidate(records, config, safety_sign)
            oob_results[config.name][str(model_seed)] = point
            metric_rows.append(
                {
                    "candidate": config.name,
                    "model_seed": model_seed,
                    **_flatten(point),
                }
            )
            fold_rows.extend(
                {
                    "candidate": config.name,
                    "model_seed": model_seed,
                    **fold,
                }
                for fold in folds
            )
    ranking = _candidate_ranking(oob_results, model_seeds)
    selected_name = str(ranking[0]["candidate"])
    selected_config = next(
        candidate for candidate in candidates if candidate.name == selected_name
    )
    selected_oob_passes = int(ranking[0]["feasible_seed_count"])
    _write_csv(args.output_dir / "oob_metrics.csv", metric_rows)
    _write_csv(args.output_dir / "oob_fold_audit.csv", fold_rows)
    _write_csv(args.output_dir / "candidate_ranking.csv", ranking)

    source_checks = {
        "objective_frozen": source_config["selected_objective"]
        == FROZEN_OBJECTIVE,
        "model_seeds_frozen": set(source_config["model_seeds"])
        == set(model_seeds),
        "three_training_batches": len(source_config["training_batch_seeds"]) == 3,
        "confirmation_labels_not_loaded_for_fit": True,
    }
    if selected_oob_passes < 2 or not all(source_checks.values()):
        readiness = {
            "schema_version": 1,
            "oob_gate_passed": False,
            "independent_gate_passed": False,
            "selected_candidate": selected_name,
            "critic_interfaces": [],
            "mch_ppo_prerequisites_complete": False,
            "next_action": "upgrade_explicit_constraint_representation",
        }
        with (args.output_dir / "mch_ppo_readiness.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(readiness, handle, indent=2, ensure_ascii=False)
        summary = {
            "schema_version": 1,
            "protocol": protocol,
            "source": {
                "oob_sha256": _sha256(oob_path),
                "source_gate_sha256": _sha256(source_gate_path),
                "source_config_sha256": _sha256(source_config_path),
                "checks": source_checks,
            },
            "oob": {
                "results": oob_results,
                "ranking": ranking,
                "selected_candidate": selected_name,
                "selected_passed_seed_count": selected_oob_passes,
                "passed": False,
            },
            "independent_test_generated": False,
            "task14_cross_batch_calibration_passed": False,
            "mch_ppo_prerequisites_complete": False,
            "resume_mch_ppo": False,
            "enter_gnn": False,
            "next_action": "upgrade_explicit_constraint_representation",
        }
        _write_early_exit(args, summary, protocol)
        return

    training_dataset = _load_npz(args.source_dir / "training_dataset.npz")
    train_indices = np.arange(len(training_dataset["group_ids"]))
    training_targets = engagement_delta_targets(
        _components(training_dataset, train_indices)
    )
    training_oracle = safety_resource_oracle(
        _components(training_dataset, train_indices)
    )["labels"]
    environment = AirDefenseResourceAssignmentEnvV1()
    layout = AirDefenseV1ObservationLayout.infer(
        environment.observation_space, environment.action_space
    )
    environment.close()

    final_models: dict[int, Any] = {}
    final_checkpoints: dict[int, dict[str, Any]] = {}
    final_calibrators: dict[int, Any] = {}
    margin_models: dict[int, Any] = {}
    checkpoint_hashes: dict[str, str] = {}
    for model_seed in model_seeds:
        checkpoint_path = (
            args.source_dir
            / "models"
            / f"{FROZEN_OBJECTIVE}_seed{model_seed}.pt"
        )
        model, checkpoint = _load_value_model(
            checkpoint_path, layout, args.device
        )
        margin_model, _ = _load_critic(
            args.critic_dir / f"balanced_bce_margin_seed{model_seed}.pt",
            layout,
            args.device,
        )
        margin_values = _predict(
            margin_model, training_dataset, train_indices, args.device
        )
        margin_logits = margin_values[:, 1] - margin_values[:, 0]
        prediction = _predict_value(
            model,
            training_dataset,
            train_indices,
            args.device,
            margin_logits,
            checkpoint["scales"],
            float(checkpoint["margin_scale"]),
        )
        records = _value_records(prediction, training_dataset["scenarios"])
        features, names = _features(records, selected_config.feature_set)
        valid = training_oracle >= 0
        calibrator = fit_cross_batch_calibrator(
            features[valid],
            names,
            training_oracle[valid],
            training_dataset["batch_ids"][valid],
            training_dataset["scenarios"][valid],
            selected_config,
        )
        final_models[model_seed] = model
        final_checkpoints[model_seed] = checkpoint
        final_calibrators[model_seed] = calibrator
        margin_models[model_seed] = margin_model
        checkpoint_hashes[str(model_seed)] = _sha256(checkpoint_path)
        with (
            args.output_dir / f"calibrator_seed{model_seed}.json"
        ).open("w", encoding="utf-8") as handle:
            json.dump(calibrator.signature(), handle, indent=2)

    history_paths = _historical_dataset_paths(args.output_dir, args.source_dir)
    test_path = args.output_dir / "test_dataset.npz"
    generation_args = _generation_args(args)
    generation_args.model_dir = args.source_model_dir
    test_dataset = (
        _load_npz(test_path)
        if args.reuse_test_dataset and test_path.exists()
        else _generate_test_dataset(generation_args)
    )
    test_indices = np.arange(len(test_dataset["group_ids"]))
    test_targets = engagement_delta_targets(
        _components(test_dataset, test_indices)
    )
    test_oracle = safety_resource_oracle(
        _components(test_dataset, test_indices)
    )["labels"]
    test_valid = test_oracle >= 0
    power = confirmation_power(test_oracle, test_dataset["scenarios"])
    test_results: dict[str, Any] = {}
    test_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    passed_seed_count = 0
    for model_seed in model_seeds:
        margin_values = _predict(
            margin_models[model_seed], test_dataset, test_indices, args.device
        )
        margin_logits = margin_values[:, 1] - margin_values[:, 0]
        checkpoint = final_checkpoints[model_seed]
        started = perf_counter()
        value_prediction = _predict_value(
            final_models[model_seed],
            test_dataset,
            test_indices,
            args.device,
            margin_logits,
            checkpoint["scales"],
            float(checkpoint["margin_scale"]),
        )
        records = _value_records(value_prediction, test_dataset["scenarios"])
        features, _ = _features(records, selected_config.feature_set)
        calibrated = final_calibrators[model_seed].predict(features)
        inference_seconds = perf_counter() - started
        value_metrics = constrained_value_metrics(
            test_targets["safety_gain"][test_valid],
            test_targets["cost_delta"][test_valid],
            value_prediction["safety_gain"][test_valid],
            value_prediction["cost_delta"][test_valid],
        )
        point = calibrated_operating_point(
            calibrated,
            test_oracle,
            np.full(len(test_oracle), "independent_batch_941000"),
            test_dataset["scenarios"],
            safety_sign_accuracy=float(value_metrics["safety_sign_accuracy"]),
        )
        inference_passed = inference_seconds < float(
            test_dataset["generation_seconds"]
        )
        passed = bool(point["feasible"]) and inference_passed
        passed_seed_count += int(passed)
        test_results[str(model_seed)] = {
            "metrics": point,
            "value_metrics": value_metrics,
            "hessian_condition": final_calibrators[
                model_seed
            ].hessian_condition,
            "inference_seconds": inference_seconds,
            "inference_faster_than_rollouts": inference_passed,
            "passed": passed,
        }
        test_rows.append(
            {
                "model_seed": model_seed,
                "candidate": selected_name,
                **_flatten(point),
                "inference_seconds": inference_seconds,
            }
        )
        for index in test_indices:
            diagnostic_rows.append(
                {
                    "model_seed": model_seed,
                    "group_id": str(test_dataset["group_ids"][index]),
                    "scenario": str(test_dataset["scenarios"][index]),
                    "oracle_label": int(test_oracle[index]),
                    "probability": float(calibrated["probability"][index]),
                    "calibrated_logit": float(
                        calibrated["calibrated_logit"][index]
                    ),
                    "prediction_se": float(
                        calibrated["prediction_se"][index]
                    ),
                    "conservative_logit": float(
                        calibrated["conservative_logit"][index]
                    ),
                    "predicted_engage": int(
                        calibrated["predicted_label"][index]
                    ),
                    "raw_score": float(value_prediction["score"][index]),
                }
            )
    _write_csv(args.output_dir / "test_metrics.csv", test_rows)
    _write_csv(args.output_dir / "test_diagnostics.csv", diagnostic_rows)

    overlaps = {
        str(path): _observation_overlap_count(
            test_dataset["observations"], _load_npz(path)["observations"]
        )
        for path in history_paths
    }
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
        "historical_overlap_zero": all(count == 0 for count in overlaps.values()),
        "return_reconstruction": reconstruction_error <= 1e-4,
        "single_independent_batch": True,
        "test_threshold_refit": False,
        "confirmation_labels_used_for_fitting": False,
    }
    stage_checks = {
        "source_integrity": all(source_checks.values()),
        "oob_two_of_three": selected_oob_passes >= 2,
        "data_integrity": all(data_checks.values()),
        "power_sufficient": bool(power["passed"]),
        "test_two_of_three": passed_seed_count >= 2,
    }
    stage_passed = all(stage_checks.values())
    readiness = {
        "schema_version": 1,
        "oob_gate_passed": selected_oob_passes >= 2,
        "independent_gate_passed": passed_seed_count >= 2,
        "selected_candidate": selected_name,
        "critic_interfaces": [
            "calibrated_probability",
            "conservative_logit",
            "prediction_se",
        ],
        "mch_ppo_prerequisites_complete": stage_passed,
        "next_action": (
            "freeze_mch_ppo_method_and_run_30k_three_seed_screening"
            if stage_passed
            else "upgrade_explicit_constraint_representation"
        ),
    }
    with (args.output_dir / "mch_ppo_readiness.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(readiness, handle, indent=2, ensure_ascii=False)
    summary = {
        "schema_version": 1,
        "protocol": protocol,
        "source": {
            "oob_sha256": _sha256(oob_path),
            "source_gate_sha256": _sha256(source_gate_path),
            "source_config_sha256": _sha256(source_config_path),
            "checkpoint_sha256": checkpoint_hashes,
            "test_dataset_sha256": _sha256(test_path),
            "checks": source_checks,
        },
        "oob": {
            "results": oob_results,
            "ranking": ranking,
            "selected_candidate": selected_name,
            "selected_passed_seed_count": selected_oob_passes,
            "passed": selected_oob_passes >= 2,
        },
        "independent_test_generated": True,
        "data_audit": {
            "checks": data_checks,
            "passed": all(data_checks.values()),
            "historical_dataset_count": len(history_paths),
            "historical_observation_overlaps": overlaps,
            "return_reconstruction_max_error": reconstruction_error,
        },
        "dataset": {
            "states": int(test_dataset["state_count"]),
            "groups": int(len(test_dataset["group_ids"])),
            "rollouts": int(test_dataset["total_return_samples"].shape[2]),
            "generation_seconds": float(test_dataset["generation_seconds"]),
        },
        "power": power,
        "test": {
            "results": test_results,
            "passed_seed_count": passed_seed_count,
            "required_passed_seed_count": 2,
        },
        "stage_gate": {"checks": stage_checks, "passed": stage_passed},
        "task14_cross_batch_calibration_passed": stage_passed,
        "mch_ppo_prerequisites_complete": stage_passed,
        "resume_mch_ppo": stage_passed,
        "enter_gnn": False,
        "next_action": readiness["next_action"],
    }
    _write_early_exit(args, summary, protocol)


if __name__ == "__main__":
    main()
