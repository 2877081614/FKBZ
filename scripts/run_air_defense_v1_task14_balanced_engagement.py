from __future__ import annotations

import argparse
from copy import deepcopy
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
    EngagementUtilityConfig,
    balanced_engagement_loss,
    engagement_criticality_features,
    engagement_utility_labels,
    grouped_state_split,
    oracle_classification_metrics,
    safety_resource_oracle,
    select_diverse_critical_snapshots,
)
from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    get_air_defense_v1_scenario,
)
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    RiskAwareEngagementCritic,
)
from scripts.run_air_defense_v1_task14_engagement_utility import (
    COMPONENT_KEYS,
    PREVIOUS_TESTS,
    _components,
    _model_inputs,
    _observation_overlap_count,
    _snapshot_groups,
    _train_model,
)
from scripts.run_air_defense_v1_task14_q_critic import (
    DEFAULT_MODEL_DIR,
    SCENARIOS,
    _distribution_and_value,
    _load_model,
    _seed_everything,
    _write_csv,
)


DEFAULT_HISTORICAL_DATASET = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_engagement_utility"
    / "dataset.npz"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task14_balanced_engagement"
)
FROZEN_UTILITY = EngagementUtilityConfig(
    cost_weight=2.0,
    damage_weight=30.0,
    high_threat_leak_weight=0.0,
    cvar_weight=0.5,
    cvar_alpha=0.25,
)
CLASSIFICATION_METHODS = {
    "balanced_bce": 0.0,
    "balanced_bce_margin": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run targeted critical-state balanced engagement estimation."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--historical-dataset", type=Path, default=DEFAULT_HISTORICAL_DATASET
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-seeds", nargs="+", type=int, default=(8, 10))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--episodes-per-stratum", type=int, default=48)
    parser.add_argument("--states-per-stratum", type=int, default=24)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--eval-seed", type=int, default=419_000)
    parser.add_argument("--split-seed", type=int, default=37)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=(20, 21, 22))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-targeted-dataset", action="store_true")
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _collect_targeted_snapshot_pool(
    *,
    model: Any,
    source_seed: int,
    scenario: str,
    episodes: int,
    state_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        env = AirDefenseResourceAssignmentEnvV1(
            get_air_defense_v1_scenario(scenario)
        )
        observation, _ = env.reset(seed=seed + episode_index)
        terminated = truncated = False
        step_index = 0
        while not (terminated or truncated):
            distribution, value = _distribution_and_value(
                model, observation, env.action_masks()
            )
            action_evaluation = distribution.sample(deterministic=False)
            actions = action_evaluation.actions
            probabilities, masks = distribution.conditional_probabilities(actions)
            action = actions[0].detach().cpu().numpy()
            conditional_masks = masks[0].detach().cpu().numpy()
            if bool(np.any(conditional_masks[:, : env.num_targets])):
                criticality = engagement_criticality_features(env, conditional_masks)
                candidates.append(
                    {
                        "state_id": (
                            f"seed{source_seed}/{scenario}/episode{episode_index}/"
                            f"step{step_index}"
                        ),
                        "source_seed": source_seed,
                        "scenario": scenario,
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "step_fraction": step_index / max(1, env.config.max_steps),
                        "observation": observation.copy(),
                        "env": deepcopy(env),
                        "base_action": action.copy(),
                        "probabilities": probabilities[0].detach().cpu().numpy(),
                        "conditional_masks": conditional_masks,
                        "critic_v": value,
                        **criticality,
                    }
                )
            observation, _, terminated, truncated, _ = env.step(action)
            step_index += 1
        env.close()
    return [
        dict(record)
        for record in select_diverse_critical_snapshots(
            candidates,
            state_count,
            seed=seed + 73,
            high_fraction=0.8,
            min_episode_step_gap=3,
        )
    ]


def _generate_targeted_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
    metadata: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    started = perf_counter()
    state_counter = 0
    for source_seed in args.source_seeds:
        model = _load_model(args.model_dir, source_seed, args.device)
        for scenario_index, scenario in enumerate(args.scenarios):
            collection_seed = (
                args.eval_seed + source_seed * 100_000 + scenario_index * 10_000
            )
            _seed_everything(collection_seed)
            snapshots = _collect_targeted_snapshot_pool(
                model=model,
                source_seed=source_seed,
                scenario=scenario,
                episodes=args.episodes_per_stratum,
                state_count=args.states_per_stratum,
                seed=collection_seed,
            )
            for snapshot in snapshots:
                rows, values = _snapshot_groups(
                    model=model,
                    snapshot=snapshot,
                    gamma=args.gamma,
                    rollouts=args.rollouts,
                    base_seed=args.eval_seed + state_counter * 1_000_000,
                )
                for row in rows:
                    row["group_id"] = row["group_id"].replace(
                        "task14u/", "task14b/", 1
                    )
                    row["state_id"] = row["state_id"].replace(
                        "task14u/", "task14b/", 1
                    )
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
                f"targeted source_seed={source_seed} scenario={scenario} "
                f"states={len(snapshots)} groups={len(metadata)}",
                flush=True,
            )

    state_ids = np.asarray([row["state_id"] for row in metadata])
    strata = np.asarray(
        [f"seed{row['source_seed']}/{row['scenario']}" for row in metadata]
    )
    splits = grouped_state_split(
        state_ids,
        strata=strata,
        validation_fraction=0.2,
        test_fraction=0.4,
        seed=args.split_seed,
    )
    for row, split in zip(metadata, splits.tolist()):
        row["split"] = split
    _write_csv(args.output_dir / "targeted_groups.csv", metadata)
    dataset: dict[str, np.ndarray] = {
        "observations": np.stack([row["observation"] for row in arrays]),
        "unit_indices": np.asarray([row["unit_index"] for row in arrays]),
        "prefix_occupancy": np.stack([row["prefix_occupancy"] for row in arrays]),
        "legal_action_masks": np.stack([row["legal_action_mask"] for row in arrays]),
        "group_ids": np.asarray([row["group_id"] for row in metadata]),
        "state_ids": state_ids,
        "source_seeds": np.asarray(
            [row["source_seed"] for row in metadata], dtype=np.int64
        ),
        "scenarios": np.asarray([row["scenario"] for row in metadata]),
        "splits": splits,
        "criticality_scores": np.asarray(
            [row["criticality_score"] for row in metadata], dtype=np.float32
        ),
        "min_times_to_impact": np.asarray(
            [row["min_time_to_impact"] for row in metadata], dtype=np.float32
        ),
        "generation_seconds": np.asarray(perf_counter() - started),
        "state_count": np.asarray(len(np.unique(state_ids))),
    }
    for key in COMPONENT_KEYS:
        dataset[key] = np.stack([row[key] for row in arrays])
    np.savez_compressed(args.output_dir / "targeted_dataset.npz", **dataset)
    return dataset


def _combine_training_data(
    historical: dict[str, np.ndarray], targeted: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    historical_train = np.flatnonzero(historical["splits"] != "test")
    keys = (
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
    combined: dict[str, np.ndarray] = {}
    common_rollouts = min(
        historical["total_return_samples"].shape[2],
        targeted["total_return_samples"].shape[2],
    )
    for key in keys:
        historical_values = historical[key][historical_train]
        targeted_values = targeted[key]
        if key in COMPONENT_KEYS:
            historical_values = historical_values[:, :, :common_rollouts]
            targeted_values = targeted_values[:, :, :common_rollouts]
        combined[key] = np.concatenate(
            (historical_values, targeted_values), axis=0
        )
    combined["splits"] = np.concatenate(
        (
            np.full(len(historical_train), "train"),
            targeted["splits"],
        )
    )
    combined["state_count"] = np.asarray(len(np.unique(combined["state_ids"])))
    return combined


def _predict(
    model: RiskAwareEngagementCritic,
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    device: str,
    *,
    mean: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return (
            model(*_model_inputs(dataset, indices, device)).cpu().numpy() * scale
            + mean
        )


def _train_classifier(
    *,
    dataset: dict[str, np.ndarray],
    oracle_labels: np.ndarray,
    layout: AirDefenseV1ObservationLayout,
    method: str,
    margin_weight: float,
    train_seed: int,
    args: argparse.Namespace,
) -> tuple[
    RiskAwareEngagementCritic,
    np.ndarray,
    list[dict[str, Any]],
    dict[str, Any],
]:
    _seed_everything(train_seed)
    train = np.flatnonzero(dataset["splits"] == "train")
    validation = np.flatnonzero(dataset["splits"] == "validation")
    test = np.flatnonzero(dataset["splits"] == "test")
    train_labels = torch.as_tensor(oracle_labels[train], device=args.device)
    validation_oracle = oracle_labels[validation]
    model = RiskAwareEngagementCritic(layout).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_score = -float("inf")
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    curves: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        prediction = model(*_model_inputs(dataset, train, args.device))
        loss, parts = balanced_engagement_loss(
            prediction,
            train_labels,
            margin_weight=margin_weight,
            margin=1.0,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        validation_prediction = _predict(
            model, dataset, validation, args.device
        )
        validation_metrics = oracle_classification_metrics(
            validation_oracle,
            (validation_prediction[:, 1] > validation_prediction[:, 0]).astype(
                np.int64
            ),
        )
        score = float(validation_metrics["balanced_accuracy"])
        if not np.isfinite(score):
            score = -1.0
        loss_value = float(loss.detach().cpu())
        curves.append(
            {
                "method": method,
                "train_seed": train_seed,
                "epoch": epoch,
                "loss": loss_value,
                "bce": float(parts["bce"].detach().cpu()),
                "margin": float(parts["margin"].detach().cpu()),
                "validation_balanced_accuracy": score,
                "validation_false_noop_rate": validation_metrics["false_noop_rate"],
                "validation_wasteful_engage_rate": validation_metrics[
                    "wasteful_engage_rate"
                ],
            }
        )
        improved = score > best_score + 1e-9 or (
            abs(score - best_score) <= 1e-9 and loss_value < best_loss - 1e-6
        )
        if improved:
            best_score = score
            best_loss = loss_value
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Balanced engagement training produced no checkpoint")
    model.load_state_dict(best_state)
    started = perf_counter()
    test_prediction = _predict(model, dataset, test, args.device)
    inference_seconds = perf_counter() - started
    record = {
        "method": method,
        "train_seed": train_seed,
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_score,
        "best_validation_loss": best_loss,
        "train_engage_count": int(np.sum(oracle_labels[train] == 1)),
        "train_noop_count": int(np.sum(oracle_labels[train] == 0)),
        "test_inference_seconds": inference_seconds,
        "parameter_count": model.parameter_count(),
    }
    return model, test_prediction, curves, record


def _scenario_metrics(
    dataset: dict[str, np.ndarray],
    test: np.ndarray,
    oracle: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for scenario in np.unique(dataset["scenarios"][test]):
        selected = dataset["scenarios"][test] == scenario
        result[str(scenario)] = oracle_classification_metrics(
            oracle[selected],
            (predictions[selected, 1] > predictions[selected, 0]).astype(np.int64),
        )
    return result


def _method_gate(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    scenario_metrics: dict[str, dict[str, float | int]],
) -> dict[str, bool]:
    scenario_noop = all(
        float(metrics["noop_recall"]) >= 0.65
        for metrics in scenario_metrics.values()
        if int(metrics["noop_count"]) > 0
    )
    scenario_engage = all(
        float(metrics["engage_recall"]) >= 0.60
        for metrics in scenario_metrics.values()
        if int(metrics["engage_count"]) > 0
    )
    return {
        "balanced_accuracy": float(candidate["balanced_accuracy"]) >= 0.70,
        "improvement": (
            float(candidate["balanced_accuracy"])
            - float(baseline["balanced_accuracy"])
            >= 0.10
        ),
        "false_noop_noninferior": (
            float(candidate["false_noop_rate"])
            <= float(baseline["false_noop_rate"])
        ),
        "wasteful_engage_noninferior": (
            float(candidate["wasteful_engage_rate"])
            <= float(baseline["wasteful_engage_rate"])
        ),
        "scenario_noop_recall": scenario_noop,
        "scenario_engage_recall": scenario_engage,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)
    targeted_path = args.output_dir / "targeted_dataset.npz"
    targeted = (
        _load_npz(targeted_path)
        if args.reuse_targeted_dataset
        else _generate_targeted_dataset(args)
    )
    historical = _load_npz(args.historical_dataset)
    combined = _combine_training_data(historical, targeted)
    np.savez_compressed(args.output_dir / "analysis_dataset.npz", **combined)

    environment = AirDefenseResourceAssignmentEnvV1()
    layout = AirDefenseV1ObservationLayout.infer(
        environment.observation_space, environment.action_space
    )
    environment.close()

    previous_paths = (*PREVIOUS_TESTS, args.historical_dataset)
    overlaps: dict[str, int] = {}
    targeted_test = np.flatnonzero(targeted["splits"] == "test")
    for path in previous_paths:
        if not path.exists():
            continue
        previous = _load_npz(path)
        previous_selected = (
            previous["splits"] == "test"
            if "splits" in previous
            else np.ones(len(previous["observations"]), dtype=bool)
        )
        overlaps[str(path)] = _observation_overlap_count(
            targeted["observations"][targeted_test],
            previous["observations"][previous_selected],
        )

    all_indices = np.arange(len(combined["group_ids"]))
    risk_labels, _ = engagement_utility_labels(
        _components(combined, all_indices), FROZEN_UTILITY
    )
    oracle_details = safety_resource_oracle(_components(combined, all_indices))
    oracle_labels = oracle_details["labels"]
    test = np.flatnonzero(combined["splits"] == "test")
    validation = np.flatnonzero(combined["splits"] == "validation")
    test_oracle = oracle_labels[test]

    all_curves: list[dict[str, Any]] = []
    result_models: dict[int, dict[str, RiskAwareEngagementCritic]] = {}
    predictions: dict[int, dict[str, np.ndarray]] = {}
    records: dict[int, dict[str, dict[str, Any]]] = {}
    validation_scores: dict[str, list[float]] = {
        method: [] for method in CLASSIFICATION_METHODS
    }
    for train_seed in args.train_seeds:
        result_models[train_seed] = {}
        predictions[train_seed] = {}
        records[train_seed] = {}
        regression, regression_prediction, curves, record = _train_model(
            dataset=combined,
            labels=risk_labels,
            layout=layout,
            method="risk_regression",
            train_seed=train_seed,
            args=args,
        )
        result_models[train_seed]["risk_regression"] = regression
        predictions[train_seed]["risk_regression"] = regression_prediction
        records[train_seed]["risk_regression"] = record
        all_curves.extend(curves)
        for method, margin_weight in CLASSIFICATION_METHODS.items():
            model, test_prediction, curves, record = _train_classifier(
                dataset=combined,
                oracle_labels=oracle_labels,
                layout=layout,
                method=method,
                margin_weight=margin_weight,
                train_seed=train_seed,
                args=args,
            )
            result_models[train_seed][method] = model
            predictions[train_seed][method] = test_prediction
            records[train_seed][method] = record
            validation_scores[method].append(
                float(record["best_validation_balanced_accuracy"])
            )
            all_curves.extend(curves)

    selected_method = max(
        CLASSIFICATION_METHODS,
        key=lambda method: (
            float(np.mean(validation_scores[method])),
            -CLASSIFICATION_METHODS[method],
        ),
    )
    model_rows: list[dict[str, Any]] = []
    formal_results: dict[str, Any] = {}
    for train_seed in args.train_seeds:
        seed_results: dict[str, Any] = {}
        for method in ("risk_regression", *CLASSIFICATION_METHODS):
            method_predictions = predictions[train_seed][method]
            metrics = oracle_classification_metrics(
                test_oracle,
                (method_predictions[:, 1] > method_predictions[:, 0]).astype(np.int64),
            )
            scenarios = _scenario_metrics(
                combined, test, test_oracle, method_predictions
            )
            seed_results[method] = {
                "metrics": metrics,
                "scenario_metrics": scenarios,
                "record": records[train_seed][method],
            }
            model_rows.append(
                {
                    **records[train_seed][method],
                    **metrics,
                    "selected_candidate": method == selected_method,
                }
            )
            model = result_models[train_seed][method]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_signature": model.signature(),
                    "method": method,
                    "utility_config": FROZEN_UTILITY.signature(),
                    "normalization": (
                        {
                            "mean": records[train_seed][method]["label_mean"],
                            "scale": records[train_seed][method]["label_scale"],
                        }
                        if method == "risk_regression"
                        else None
                    ),
                },
                args.output_dir / "models" / f"{method}_seed{train_seed}.pt",
            )
        formal_results[str(train_seed)] = seed_results
    _write_csv(args.output_dir / "training_curves.csv", all_curves)
    _write_csv(args.output_dir / "model_metrics.csv", model_rows)

    targeted_components = _components(targeted, np.arange(len(targeted["group_ids"])))
    targeted_oracle_details = safety_resource_oracle(targeted_components)
    targeted_oracle = targeted_oracle_details["labels"]
    targeted_test_oracle = targeted_oracle[targeted_test]
    valid = targeted_test_oracle >= 0
    engage_count = int(np.sum(targeted_test_oracle == 1))
    noop_count = int(np.sum(targeted_test_oracle == 0))
    scenario_counts = {
        str(scenario): int(
            np.sum(valid & (targeted["scenarios"][targeted_test] == scenario))
        )
        for scenario in np.unique(targeted["scenarios"][targeted_test])
    }
    engage_scenarios = int(
        sum(
            bool(
                np.any(
                    (targeted_test_oracle == 1)
                    & (targeted["scenarios"][targeted_test] == scenario)
                )
            )
            for scenario in np.unique(targeted["scenarios"][targeted_test])
        )
    )
    engage_rate = engage_count / int(np.sum(valid)) if np.any(valid) else 0.0
    previous_engage_rate = 3.0 / 57.0
    power_checks = {
        "valid_count": int(np.sum(valid)) >= 40,
        "engage_count": engage_count >= 8,
        "noop_count": noop_count >= 8,
        "scenario_counts": all(count >= 8 for count in scenario_counts.values()),
        "engage_scenarios": engage_scenarios >= 2,
        "engage_rate_enrichment": engage_rate - previous_engage_rate >= 0.03,
    }
    power_sufficient = all(power_checks.values())

    reconstructed = (
        targeted["operational_return_samples"]
        - targeted["resource_cost_samples"]
        - 30.0 * targeted["damage_samples"]
    )
    reconstruction_error = float(
        np.max(np.abs(targeted["total_return_samples"] - reconstructed))
    )
    split_leakage = any(
        len(set(targeted["splits"][targeted["state_ids"] == state_id])) != 1
        for state_id in np.unique(targeted["state_ids"])
    )
    data_checks = {
        "states": int(targeted["state_count"]) == (
            len(args.source_seeds) * len(args.scenarios) * args.states_per_stratum
        ),
        "rollouts": targeted["total_return_samples"].shape[2] == args.rollouts,
        "overlap_zero": all(count == 0 for count in overlaps.values()),
        "split_leakage_zero": not split_leakage,
        "return_reconstruction": reconstruction_error <= 1e-4,
    }
    data_integrity = all(data_checks.values())

    per_seed_gates: dict[str, Any] = {}
    passed_seed_count = 0
    for train_seed in args.train_seeds:
        baseline = formal_results[str(train_seed)]["risk_regression"]["metrics"]
        candidate_result = formal_results[str(train_seed)][selected_method]
        checks = _method_gate(
            candidate_result["metrics"], baseline, candidate_result["scenario_metrics"]
        )
        passed = all(checks.values())
        passed_seed_count += int(passed)
        per_seed_gates[str(train_seed)] = {**checks, "passed": passed}

    test_rows: list[dict[str, Any]] = []
    for local_index, dataset_index in enumerate(test):
        row: dict[str, Any] = {
            "group_id": str(combined["group_ids"][dataset_index]),
            "scenario": str(combined["scenarios"][dataset_index]),
            "source_seed": int(combined["source_seeds"][dataset_index]),
            "oracle_label": int(test_oracle[local_index]),
            "harm_delta": float(oracle_details["harm_delta"][dataset_index]),
            "cost_delta": float(oracle_details["cost_delta"][dataset_index]),
        }
        for train_seed in args.train_seeds:
            for method in ("risk_regression", selected_method):
                values = predictions[train_seed][method][local_index]
                row[f"{method}_seed{train_seed}_delta"] = float(values[1] - values[0])
        test_rows.append(row)
    _write_csv(args.output_dir / "test_group_diagnostics.csv", test_rows)

    estimator_passed = passed_seed_count >= 2
    stage_passed = data_integrity and power_sufficient and estimator_passed
    summary = {
        "schema_version": 1,
        "data_audit": {
            "checks": data_checks,
            "passed": data_integrity,
            "previous_test_overlaps": overlaps,
            "state_split_leakage": split_leakage,
            "return_reconstruction_max_error": reconstruction_error,
        },
        "dataset": {
            "targeted_states": int(targeted["state_count"]),
            "targeted_groups": int(len(targeted["group_ids"])),
            "combined_groups": int(len(combined["group_ids"])),
            "historical_train_groups": int(np.sum(historical["splits"] != "test")),
            "train_groups": int(np.sum(combined["splits"] == "train")),
            "validation_groups": int(np.sum(combined["splits"] == "validation")),
            "test_groups": int(np.sum(combined["splits"] == "test")),
            "rollouts": int(targeted["total_return_samples"].shape[2]),
            "generation_seconds": float(targeted["generation_seconds"]),
        },
        "frozen_utility": FROZEN_UTILITY.signature(),
        "power": {
            "checks": power_checks,
            "passed": power_sufficient,
            "valid_count": int(np.sum(valid)),
            "engage_count": engage_count,
            "noop_count": noop_count,
            "engage_rate": engage_rate,
            "previous_engage_rate": previous_engage_rate,
            "scenario_counts": scenario_counts,
            "engage_scenario_count": int(engage_scenarios),
        },
        "candidate_selection": {
            "validation_scores": {
                method: values for method, values in validation_scores.items()
            },
            "mean_validation_scores": {
                method: float(np.mean(values))
                for method, values in validation_scores.items()
            },
            "selected_method": selected_method,
        },
        "model_results": formal_results,
        "model_gate": {
            "per_seed": per_seed_gates,
            "passed_seed_count": passed_seed_count,
            "required_passed_seed_count": 2,
            "passed": estimator_passed,
        },
        "task14_balanced_engagement_passed": stage_passed,
        "resume_mch_ppo": stage_passed,
        "enter_gnn": False,
    }
    with (args.output_dir / "gate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    config = {
        "schema_version": 1,
        "historical_dataset": str(args.historical_dataset.resolve()),
        "source_seeds": list(args.source_seeds),
        "scenarios": list(args.scenarios),
        "episodes_per_stratum": args.episodes_per_stratum,
        "states_per_stratum": args.states_per_stratum,
        "rollouts": args.rollouts,
        "gamma": args.gamma,
        "eval_seed": args.eval_seed,
        "split_seed": args.split_seed,
        "train_seeds": list(args.train_seeds),
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "frozen_utility": FROZEN_UTILITY.signature(),
    }
    with (args.output_dir / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
