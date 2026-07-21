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
    EngagementBoundaryConfig,
    EngagementBoundaryConstraints,
    apply_engagement_boundary,
    calibrate_engagement_boundary,
    oracle_classification_metrics,
    resource_pressure_from_observations,
    safety_resource_oracle,
    scenario_classification_metrics,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    RiskAwareEngagementCritic,
)
from scripts.run_air_defense_v1_task14_balanced_engagement import (
    _collect_targeted_snapshot_pool,
)
from scripts.run_air_defense_v1_task14_engagement_utility import (
    COMPONENT_KEYS,
    PREVIOUS_TESTS,
    _components,
    _model_inputs,
    _observation_overlap_count,
    _snapshot_groups,
)
from scripts.run_air_defense_v1_task14_q_critic import (
    DEFAULT_MODEL_DIR,
    SCENARIOS,
    _load_model,
    _seed_everything,
    _write_csv,
)


DEFAULT_BALANCED_DIR = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task14_balanced_engagement"
)
DEFAULT_CALIBRATION_DATASET = DEFAULT_BALANCED_DIR / "analysis_dataset.npz"
DEFAULT_CRITIC_DIR = DEFAULT_BALANCED_DIR / "models"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task14_engagement_calibration"
)
OLD_DATASETS = (
    *PREVIOUS_TESTS,
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_engagement_utility"
    / "dataset.npz",
    DEFAULT_BALANCED_DIR / "targeted_dataset.npz",
    DEFAULT_CALIBRATION_DATASET,
)
FROZEN_CONSTRAINTS = EngagementBoundaryConstraints()
GLOBAL_WEIGHTS = (0.0,)
RESOURCE_DUAL_WEIGHTS = (0.25, 0.5, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate and independently test resource-aware engagement boundaries."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC_DIR)
    parser.add_argument(
        "--calibration-dataset", type=Path, default=DEFAULT_CALIBRATION_DATASET
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-seeds", nargs="+", type=int, default=(8, 10))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--episodes-per-stratum", type=int, default=30)
    parser.add_argument("--states-per-stratum", type=int, default=12)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--eval-seed", type=int, default=487_000)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=(20, 21, 22))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-test-dataset", action="store_true")
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _generate_test_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
    metadata: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    started = perf_counter()
    state_counter = 0
    for source_seed in args.source_seeds:
        source_model = _load_model(args.model_dir, source_seed, args.device)
        for scenario_index, scenario in enumerate(args.scenarios):
            collection_seed = (
                args.eval_seed + source_seed * 100_000 + scenario_index * 10_000
            )
            _seed_everything(collection_seed)
            snapshots = _collect_targeted_snapshot_pool(
                model=source_model,
                source_seed=source_seed,
                scenario=scenario,
                episodes=args.episodes_per_stratum,
                state_count=args.states_per_stratum,
                seed=collection_seed,
            )
            for snapshot in snapshots:
                rows, values = _snapshot_groups(
                    model=source_model,
                    snapshot=snapshot,
                    gamma=args.gamma,
                    rollouts=args.rollouts,
                    base_seed=args.eval_seed + state_counter * 1_000_000,
                )
                for row in rows:
                    row["group_id"] = row["group_id"].replace(
                        "task14u/", "task14c/", 1
                    )
                    row["state_id"] = row["state_id"].replace(
                        "task14u/", "task14c/", 1
                    )
                    row["split"] = "test"
                    for key in (
                        "criticality_score",
                        "max_damage_potential",
                        "min_time_to_impact",
                        "max_threat",
                        "legal_relation_count",
                        "step_fraction",
                    ):
                        row[key] = snapshot[key]
                metadata.extend(rows)
                arrays.extend(values)
                state_counter += 1
            print(
                f"calibration test source_seed={source_seed} scenario={scenario} "
                f"states={len(snapshots)} groups={len(metadata)}",
                flush=True,
            )

    if not metadata:
        raise RuntimeError("Independent test generation produced no engagement groups")
    _write_csv(args.output_dir / "test_groups.csv", metadata)
    dataset: dict[str, np.ndarray] = {
        "observations": np.stack([row["observation"] for row in arrays]),
        "unit_indices": np.asarray([row["unit_index"] for row in arrays]),
        "prefix_occupancy": np.stack([row["prefix_occupancy"] for row in arrays]),
        "legal_action_masks": np.stack([row["legal_action_mask"] for row in arrays]),
        "group_ids": np.asarray([row["group_id"] for row in metadata]),
        "state_ids": np.asarray([row["state_id"] for row in metadata]),
        "source_seeds": np.asarray(
            [row["source_seed"] for row in metadata], dtype=np.int64
        ),
        "scenarios": np.asarray([row["scenario"] for row in metadata]),
        "splits": np.full(len(metadata), "test"),
        "criticality_scores": np.asarray(
            [row["criticality_score"] for row in metadata], dtype=np.float32
        ),
        "min_times_to_impact": np.asarray(
            [row["min_time_to_impact"] for row in metadata], dtype=np.float32
        ),
        "generation_seconds": np.asarray(perf_counter() - started),
        "state_count": np.asarray(
            len(np.unique([row["state_id"] for row in metadata]))
        ),
    }
    for key in COMPONENT_KEYS:
        dataset[key] = np.stack([row[key] for row in arrays])
    np.savez_compressed(args.output_dir / "test_dataset.npz", **dataset)
    return dataset


def _load_critic(
    path: Path,
    layout: AirDefenseV1ObservationLayout,
    device: str,
) -> tuple[RiskAwareEngagementCritic, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = RiskAwareEngagementCritic(layout).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _predict(
    model: RiskAwareEngagementCritic,
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    device: str,
    normalization: dict[str, float] | None = None,
) -> np.ndarray:
    with torch.no_grad():
        values = model(*_model_inputs(dataset, indices, device)).cpu().numpy()
    if normalization is not None:
        values = values * float(normalization["scale"]) + float(
            normalization["mean"]
        )
    return values


def _pressure(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    layout: AirDefenseV1ObservationLayout,
) -> np.ndarray:
    return resource_pressure_from_observations(
        dataset["observations"][indices],
        dataset["unit_indices"][indices],
        **layout.signature(),
    )


def _evaluate_predictions(
    oracle: np.ndarray,
    predicted: np.ndarray,
    scenarios: np.ndarray,
) -> dict[str, Any]:
    return {
        "metrics": oracle_classification_metrics(oracle, predicted),
        "scenario_metrics": scenario_classification_metrics(
            oracle, predicted, scenarios
        ),
    }


def _calibrate_family(
    *,
    family: str,
    logits: np.ndarray,
    oracle: np.ndarray,
    scenarios: np.ndarray,
    pressure: np.ndarray,
) -> tuple[EngagementBoundaryConfig, list[dict[str, object]], dict[str, Any]]:
    weights = GLOBAL_WEIGHTS if family == "global_threshold" else RESOURCE_DUAL_WEIGHTS
    config, rows = calibrate_engagement_boundary(
        logits,
        oracle,
        scenarios,
        pressure,
        dual_weights=weights,
        constraints=FROZEN_CONSTRAINTS,
    )
    predictions = apply_engagement_boundary(logits, config, pressure)
    result = _evaluate_predictions(oracle, predictions, scenarios)
    result["config"] = config.signature()
    return config, rows, result


def _select_family(calibration: dict[str, dict[str, Any]]) -> str:
    ranked: list[tuple[tuple[float, float, float], str]] = []
    for family in ("global_threshold", "resource_dual"):
        results = [seed[family] for seed in calibration.values()]
        feasible_count = sum(bool(row["config"]["feasible"]) for row in results)
        mean_balanced = float(
            np.mean([row["metrics"]["balanced_accuracy"] for row in results])
        )
        simplicity = 1.0 if family == "global_threshold" else 0.0
        ranked.append(((float(feasible_count), mean_balanced, simplicity), family))
    ranked.sort(reverse=True)
    return ranked[0][1]


def _test_gate(
    candidate: dict[str, Any],
    zero_margin: dict[str, Any],
    regression: dict[str, Any],
) -> dict[str, bool]:
    metrics = candidate["metrics"]
    zero = zero_margin["metrics"]
    baseline = regression["metrics"]
    scenarios = candidate["scenario_metrics"]
    return {
        "balanced_accuracy": float(metrics["balanced_accuracy"]) >= 0.70,
        "false_noop_vs_regression": float(metrics["false_noop_rate"])
        <= float(baseline["false_noop_rate"]),
        "wasteful_engage_vs_regression": float(metrics["wasteful_engage_rate"])
        <= float(baseline["wasteful_engage_rate"]),
        "wasteful_engage_improvement": float(zero["wasteful_engage_rate"])
        - float(metrics["wasteful_engage_rate"])
        >= 0.10,
        "false_noop_tolerance": float(metrics["false_noop_rate"])
        - float(zero["false_noop_rate"])
        <= 0.10,
        "scenario_noop_recall": all(
            float(row["noop_recall"]) >= 0.65
            for row in scenarios.values()
            if int(row["noop_count"]) > 0
        ),
        "scenario_engage_recall": all(
            float(row["engage_recall"]) >= 0.60
            for row in scenarios.values()
            if int(row["engage_count"]) > 0
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_path = args.output_dir / "test_dataset.npz"
    test_dataset = (
        _load_npz(test_path)
        if args.reuse_test_dataset
        else _generate_test_dataset(args)
    )
    calibration_dataset = _load_npz(args.calibration_dataset)

    environment = AirDefenseResourceAssignmentEnvV1()
    layout = AirDefenseV1ObservationLayout.infer(
        environment.observation_space, environment.action_space
    )
    environment.close()

    validation = np.flatnonzero(calibration_dataset["splits"] == "validation")
    calibration_oracle = safety_resource_oracle(
        _components(calibration_dataset, validation)
    )["labels"]
    validation_scenarios = calibration_dataset["scenarios"][validation]
    validation_pressure = _pressure(calibration_dataset, validation, layout)
    test_indices = np.arange(len(test_dataset["group_ids"]))
    test_oracle_details = safety_resource_oracle(
        _components(test_dataset, test_indices)
    )
    test_oracle = test_oracle_details["labels"]
    test_scenarios = test_dataset["scenarios"]
    test_pressure = _pressure(test_dataset, test_indices, layout)

    calibration_results: dict[str, dict[str, Any]] = {}
    configs: dict[int, dict[str, EngagementBoundaryConfig]] = {}
    calibration_rows: list[dict[str, object]] = []
    loaded_models: dict[
        int, dict[str, tuple[RiskAwareEngagementCritic, dict[str, Any]]]
    ] = {}
    for model_seed in args.model_seeds:
        loaded_models[model_seed] = {}
        margin_model, margin_checkpoint = _load_critic(
            args.critic_dir / f"balanced_bce_margin_seed{model_seed}.pt",
            layout,
            args.device,
        )
        regression_model, regression_checkpoint = _load_critic(
            args.critic_dir / f"risk_regression_seed{model_seed}.pt",
            layout,
            args.device,
        )
        loaded_models[model_seed]["margin"] = (margin_model, margin_checkpoint)
        loaded_models[model_seed]["regression"] = (
            regression_model,
            regression_checkpoint,
        )
        validation_values = _predict(
            margin_model, calibration_dataset, validation, args.device
        )
        validation_logits = validation_values[:, 1] - validation_values[:, 0]
        calibration_results[str(model_seed)] = {}
        configs[model_seed] = {}
        for family in ("global_threshold", "resource_dual"):
            config, rows, result = _calibrate_family(
                family=family,
                logits=validation_logits,
                oracle=calibration_oracle,
                scenarios=validation_scenarios,
                pressure=validation_pressure,
            )
            configs[model_seed][family] = config
            calibration_results[str(model_seed)][family] = result
            calibration_rows.extend(
                {"model_seed": model_seed, "candidate_family": family, **row}
                for row in rows
            )
    selected_family = _select_family(calibration_results)
    _write_csv(args.output_dir / "calibration_grid.csv", calibration_rows)

    model_results: dict[str, Any] = {}
    model_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    per_seed_gates: dict[str, Any] = {}
    passed_seed_count = 0
    for model_seed in args.model_seeds:
        margin_model, _ = loaded_models[model_seed]["margin"]
        regression_model, regression_checkpoint = loaded_models[model_seed][
            "regression"
        ]
        margin_values = _predict(
            margin_model, test_dataset, test_indices, args.device
        )
        margin_logits = margin_values[:, 1] - margin_values[:, 0]
        regression_values = _predict(
            regression_model,
            test_dataset,
            test_indices,
            args.device,
            regression_checkpoint["normalization"],
        )
        regression_logits = regression_values[:, 1] - regression_values[:, 0]
        zero_predictions = (margin_logits > 0.0).astype(np.int64)
        regression_predictions = (regression_logits > 0.0).astype(np.int64)
        selected_config = configs[model_seed][selected_family]
        calibrated_predictions = apply_engagement_boundary(
            margin_logits, selected_config, test_pressure
        )
        zero_result = _evaluate_predictions(
            test_oracle, zero_predictions, test_scenarios
        )
        regression_result = _evaluate_predictions(
            test_oracle, regression_predictions, test_scenarios
        )
        calibrated_result = _evaluate_predictions(
            test_oracle, calibrated_predictions, test_scenarios
        )
        calibrated_result["config"] = selected_config.signature()
        checks = _test_gate(calibrated_result, zero_result, regression_result)
        passed = all(checks.values())
        passed_seed_count += int(passed)
        per_seed_gates[str(model_seed)] = {**checks, "passed": passed}
        model_results[str(model_seed)] = {
            "zero_margin": zero_result,
            "risk_regression": regression_result,
            "calibrated": calibrated_result,
        }
        for method, result in (
            ("zero_margin", zero_result),
            ("risk_regression", regression_result),
            ("calibrated", calibrated_result),
        ):
            model_rows.append(
                {
                    "model_seed": model_seed,
                    "method": method,
                    "selected_family": selected_family,
                    **result["metrics"],
                }
            )
        for index in test_indices:
            diagnostics.append(
                {
                    "model_seed": model_seed,
                    "group_id": str(test_dataset["group_ids"][index]),
                    "scenario": str(test_scenarios[index]),
                    "source_seed": int(test_dataset["source_seeds"][index]),
                    "unit_index": int(test_dataset["unit_indices"][index]),
                    "oracle_label": int(test_oracle[index]),
                    "resource_pressure": float(test_pressure[index]),
                    "margin_logit": float(margin_logits[index]),
                    "regression_delta": float(regression_logits[index]),
                    "zero_prediction": int(zero_predictions[index]),
                    "calibrated_prediction": int(calibrated_predictions[index]),
                    "harm_delta": float(test_oracle_details["harm_delta"][index]),
                    "cost_delta": float(test_oracle_details["cost_delta"][index]),
                }
            )
    _write_csv(args.output_dir / "model_metrics.csv", model_rows)
    _write_csv(args.output_dir / "test_group_diagnostics.csv", diagnostics)

    overlaps: dict[str, int] = {}
    for path in OLD_DATASETS:
        if not path.exists():
            continue
        old = _load_npz(path)
        overlaps[str(path)] = _observation_overlap_count(
            test_dataset["observations"], old["observations"]
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
        str(scenario): int(np.sum(valid & (test_scenarios == scenario)))
        for scenario in np.unique(test_scenarios)
    }
    engage_scenario_count = sum(
        bool(np.any((test_oracle == 1) & (test_scenarios == scenario)))
        for scenario in np.unique(test_scenarios)
    )
    power_checks = {
        "valid_count": int(np.sum(valid)) >= 40,
        "engage_count": engage_count >= 10,
        "noop_count": noop_count >= 10,
        "scenario_counts": all(count >= 8 for count in scenario_counts.values()),
        "engage_scenarios": engage_scenario_count >= 2,
    }
    data_integrity = all(data_checks.values())
    power_sufficient = all(power_checks.values())
    estimator_passed = passed_seed_count >= 2
    stage_passed = data_integrity and power_sufficient and estimator_passed
    summary = {
        "schema_version": 1,
        "data_audit": {
            "checks": data_checks,
            "passed": data_integrity,
            "previous_observation_overlaps": overlaps,
            "return_reconstruction_max_error": reconstruction_error,
        },
        "dataset": {
            "states": int(test_dataset["state_count"]),
            "groups": int(len(test_dataset["group_ids"])),
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
            "engage_scenario_count": int(engage_scenario_count),
        },
        "calibration": {
            "dataset": str(args.calibration_dataset.resolve()),
            "split": "validation",
            "validation_groups": int(len(validation)),
            "constraints": FROZEN_CONSTRAINTS.signature(),
            "families": calibration_results,
            "selected_family": selected_family,
        },
        "model_results": model_results,
        "model_gate": {
            "per_seed": per_seed_gates,
            "passed_seed_count": passed_seed_count,
            "required_passed_seed_count": 2,
            "passed": estimator_passed,
        },
        "task14_engagement_calibration_passed": stage_passed,
        "resume_mch_ppo": stage_passed,
        "enter_gnn": False,
    }
    with (args.output_dir / "gate_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    config = {
        "schema_version": 1,
        "model_dir": str(args.model_dir.resolve()),
        "critic_dir": str(args.critic_dir.resolve()),
        "calibration_dataset": str(args.calibration_dataset.resolve()),
        "source_seeds": list(args.source_seeds),
        "scenarios": list(args.scenarios),
        "episodes_per_stratum": args.episodes_per_stratum,
        "states_per_stratum": args.states_per_stratum,
        "rollouts": args.rollouts,
        "gamma": args.gamma,
        "eval_seed": args.eval_seed,
        "model_seeds": list(args.model_seeds),
        "global_weights": list(GLOBAL_WEIGHTS),
        "resource_dual_weights": list(RESOURCE_DUAL_WEIGHTS),
        "constraints": FROZEN_CONSTRAINTS.signature(),
        "selected_family": selected_family,
    }
    with (args.output_dir / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
