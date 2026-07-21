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
    robust_state_conditioned_value_loss,
    safety_resource_oracle,
    scenario_classification_metrics,
    state_conditioned_value_loss,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    StateConditionedEngagementValue,
    StateConditionedEngagementValueConfig,
)
from scripts.run_air_defense_v1_task14_balanced_engagement import (
    _collect_targeted_snapshot_pool,
)
from scripts.run_air_defense_v1_task14_engagement_calibration import (
    DEFAULT_BALANCED_DIR,
    DEFAULT_CALIBRATION_DATASET,
    DEFAULT_CRITIC_DIR,
    OLD_DATASETS,
    _load_critic,
    _load_npz,
    _predict,
)
from scripts.run_air_defense_v1_task14_engagement_utility import (
    COMPONENT_KEYS,
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


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task14_state_conditioned_value"
)
METHODS = ("safety_only", "global_budget", "state_budget")
FOLDS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run state-conditioned constrained engagement value diagnostics."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC_DIR)
    parser.add_argument(
        "--training-dataset", type=Path, default=DEFAULT_CALIBRATION_DATASET
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-seeds", nargs="+", type=int, default=(8, 10))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--episodes-per-stratum", type=int, default=30)
    parser.add_argument("--states-per-stratum", type=int, default=12)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--eval-seed", type=int, default=563_000)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=(20, 21, 22))
    parser.add_argument("--fold-seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-test-dataset", action="store_true")
    return parser.parse_args()


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
                    original_group = row["group_id"].removeprefix("task14u/")
                    original_state = row["state_id"].removeprefix("task14u/")
                    row["group_id"] = f"task14v/seed{args.eval_seed}/{original_group}"
                    row["state_id"] = f"task14v/seed{args.eval_seed}/{original_state}"
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
                f"state-value test source_seed={source_seed} scenario={scenario} "
                f"states={len(snapshots)} groups={len(metadata)}",
                flush=True,
            )
    if not metadata:
        raise RuntimeError("Independent test generation produced no engagement groups")
    _write_csv(args.output_dir / "test_groups.csv", metadata)
    state_ids = np.asarray([row["state_id"] for row in metadata])
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
        "splits": np.full(len(metadata), "test"),
        "generation_seconds": np.asarray(perf_counter() - started),
        "state_count": np.asarray(len(np.unique(state_ids))),
    }
    for key in COMPONENT_KEYS:
        dataset[key] = np.stack([row[key] for row in arrays])
    np.savez_compressed(args.output_dir / "test_dataset.npz", **dataset)
    return dataset


def _grouped_folds(
    dataset: dict[str, np.ndarray], indices: np.ndarray, *, folds: int, seed: int
) -> np.ndarray:
    state_ids = dataset["state_ids"][indices]
    strata = np.asarray(
        [
            f"{source}/{scenario}"
            for source, scenario in zip(
                dataset["source_seeds"][indices], dataset["scenarios"][indices]
            )
        ]
    )
    result = np.full(len(indices), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for stratum in np.unique(strata):
        stratum_states = np.unique(state_ids[strata == stratum])
        rng.shuffle(stratum_states)
        for position, state_id in enumerate(stratum_states):
            result[state_ids == state_id] = position % folds
    if np.any(result < 0):
        raise RuntimeError("Grouped fold assignment left unassigned rows")
    return result


def _target_scales(
    safety: np.ndarray, cost: np.ndarray, indices: np.ndarray
) -> dict[str, float]:
    return {
        "safety": max(float(np.sqrt(np.mean(np.square(safety[indices])))), 1e-6),
        "cost": max(float(np.sqrt(np.mean(np.square(cost[indices])))), 1e-6),
    }


def _value_inputs(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    device: str,
    margin_logits: np.ndarray,
    margin_scale: float,
) -> tuple[torch.Tensor, ...]:
    return (
        *_model_inputs(dataset, indices, device),
        torch.as_tensor(
            margin_logits[indices] / margin_scale,
            device=device,
            dtype=torch.float32,
        ),
    )


def _predict_value(
    model: StateConditionedEngagementValue,
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    device: str,
    margin_logits: np.ndarray,
    scales: dict[str, float],
    margin_scale: float,
) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        output = model(
            *_value_inputs(dataset, indices, device, margin_logits, margin_scale)
        )
    return {
        "safety_gain": output.safety_gain.cpu().numpy() * scales["safety"],
        "cost_delta": output.cost_delta.cpu().numpy() * scales["cost"],
        "budget_multiplier": output.budget_multiplier.cpu().numpy(),
        "score": output.score.cpu().numpy(),
    }


def _fit_fold(
    *,
    dataset: dict[str, np.ndarray],
    fit: np.ndarray,
    validation: np.ndarray,
    safety_targets: np.ndarray,
    cost_targets: np.ndarray,
    oracle_labels: np.ndarray,
    margin_logits: np.ndarray,
    layout: AirDefenseV1ObservationLayout,
    mode: str,
    seed: int,
    fold_index: int,
    args: argparse.Namespace,
    objective: str = "standard",
    cost_reliability: np.ndarray | None = None,
) -> tuple[
    StateConditionedEngagementValue,
    dict[str, np.ndarray],
    dict[str, float],
    float,
    int,
    list[dict[str, Any]],
]:
    _seed_everything(seed * 100 + fold_index)
    scales = _target_scales(safety_targets, cost_targets, fit)
    margin_scale = max(float(np.std(margin_logits[fit])), 1e-6)
    model = StateConditionedEngagementValue(
        layout, StateConditionedEngagementValueConfig(budget_mode=mode)
    ).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    fit_safety = torch.as_tensor(
        safety_targets[fit] / scales["safety"],
        device=args.device,
        dtype=torch.float32,
    )
    fit_cost = torch.as_tensor(
        cost_targets[fit] / scales["cost"],
        device=args.device,
        dtype=torch.float32,
    )
    fit_oracle = torch.as_tensor(oracle_labels[fit], device=args.device)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    curves: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        output = model(
            *_value_inputs(dataset, fit, args.device, margin_logits, margin_scale)
        )
        if objective == "standard":
            loss, parts = state_conditioned_value_loss(
                output, fit_safety, fit_cost, fit_oracle
            )
        elif objective in {
            "scenario_robust",
            "scenario_robust_reliable_cost",
        }:
            reliability = (
                torch.as_tensor(cost_reliability[fit], device=args.device)
                if objective == "scenario_robust_reliable_cost"
                and cost_reliability is not None
                else None
            )
            loss, parts = robust_state_conditioned_value_loss(
                output,
                fit_safety,
                fit_cost,
                fit_oracle,
                dataset.get("robust_groups", dataset["scenarios"])[fit],
                cost_reliability=reliability,
            )
        else:
            raise ValueError(f"Unsupported training objective: {objective}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        prediction = _predict_value(
            model,
            dataset,
            validation,
            args.device,
            margin_logits,
            scales,
            margin_scale,
        )
        metrics = oracle_classification_metrics(
            oracle_labels[validation], (prediction["score"] > 0.0).astype(np.int64)
        )
        score = float(metrics["balanced_accuracy"])
        if not np.isfinite(score):
            score = -1.0
        loss_value = float(loss.detach().cpu())
        curves.append(
            {
                "mode": mode,
                "objective": objective,
                "model_seed": seed,
                "fold": fold_index,
                "epoch": epoch,
                "loss": loss_value,
                "safety_loss": float(parts["safety"].detach().cpu()),
                "cost_loss": float(parts["cost"].detach().cpu()),
                "classification_loss": float(parts["classification"].detach().cpu()),
                "worst_block_loss": float(
                    parts.get("worst_block", torch.zeros(())).detach().cpu()
                ),
                "budget_mean": float(parts["budget_penalty"].detach().cpu()),
                "validation_balanced_accuracy": score,
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
        raise RuntimeError("Cross-fit training produced no checkpoint")
    model.load_state_dict(best_state)
    prediction = _predict_value(
        model,
        dataset,
        validation,
        args.device,
        margin_logits,
        scales,
        margin_scale,
    )
    return model, prediction, scales, margin_scale, best_epoch, curves


def _fit_final(
    *,
    dataset: dict[str, np.ndarray],
    fit: np.ndarray,
    safety_targets: np.ndarray,
    cost_targets: np.ndarray,
    oracle_labels: np.ndarray,
    margin_logits: np.ndarray,
    layout: AirDefenseV1ObservationLayout,
    mode: str,
    seed: int,
    epochs: int,
    args: argparse.Namespace,
    objective: str = "standard",
    cost_reliability: np.ndarray | None = None,
) -> tuple[StateConditionedEngagementValue, dict[str, float], float]:
    _seed_everything(seed * 1000 + 17)
    scales = _target_scales(safety_targets, cost_targets, fit)
    margin_scale = max(float(np.std(margin_logits[fit])), 1e-6)
    model = StateConditionedEngagementValue(
        layout, StateConditionedEngagementValueConfig(budget_mode=mode)
    ).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    safety = torch.as_tensor(
        safety_targets[fit] / scales["safety"],
        device=args.device,
        dtype=torch.float32,
    )
    cost = torch.as_tensor(
        cost_targets[fit] / scales["cost"],
        device=args.device,
        dtype=torch.float32,
    )
    oracle = torch.as_tensor(oracle_labels[fit], device=args.device)
    for _ in range(epochs):
        model.train()
        output = model(
            *_value_inputs(dataset, fit, args.device, margin_logits, margin_scale)
        )
        if objective == "standard":
            loss, _ = state_conditioned_value_loss(output, safety, cost, oracle)
        elif objective in {
            "scenario_robust",
            "scenario_robust_reliable_cost",
        }:
            reliability = (
                torch.as_tensor(cost_reliability[fit], device=args.device)
                if objective == "scenario_robust_reliable_cost"
                and cost_reliability is not None
                else None
            )
            loss, _ = robust_state_conditioned_value_loss(
                output,
                safety,
                cost,
                oracle,
                dataset.get("robust_groups", dataset["scenarios"])[fit],
                cost_reliability=reliability,
            )
        else:
            raise ValueError(f"Unsupported training objective: {objective}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model, scales, margin_scale


def _model_gate(
    result: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, bool]:
    metrics = result["metrics"]
    scenarios = result["scenario_metrics"]
    return {
        "balanced_accuracy": float(metrics["balanced_accuracy"]) >= 0.70,
        "engage_recall": float(metrics["engage_recall"]) >= 0.60,
        "noop_recall": float(metrics["noop_recall"]) >= 0.65,
        "false_noop_noninferior": float(metrics["false_noop_rate"])
        <= float(baseline["metrics"]["false_noop_rate"]),
        "wasteful_engage_noninferior": float(metrics["wasteful_engage_rate"])
        <= float(baseline["metrics"]["wasteful_engage_rate"]),
        "scenario_engage_recall": all(
            float(row["engage_recall"]) >= 0.60
            for row in scenarios.values()
            if int(row["engage_count"]) > 0
        ),
        "scenario_noop_recall": all(
            float(row["noop_recall"]) >= 0.65
            for row in scenarios.values()
            if int(row["noop_count"]) > 0
        ),
        "safety_sign_accuracy": float(
            result["value_metrics"]["safety_sign_accuracy"]
        )
        >= 0.70,
    }


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

    training_targets = engagement_delta_targets(
        _components(training_dataset, np.arange(len(training_dataset["group_ids"])))
    )
    training_oracle = safety_resource_oracle(
        _components(training_dataset, np.arange(len(training_dataset["group_ids"])))
    )["labels"]
    test_indices = np.arange(len(test_dataset["group_ids"]))
    test_targets = engagement_delta_targets(_components(test_dataset, test_indices))
    test_oracle_details = safety_resource_oracle(
        _components(test_dataset, test_indices)
    )
    test_oracle = test_oracle_details["labels"]

    margin_models: dict[int, Any] = {}
    regression_models: dict[int, Any] = {}
    training_margin_logits: dict[int, np.ndarray] = {}
    test_margin_logits: dict[int, np.ndarray] = {}
    test_regression_logits: dict[int, np.ndarray] = {}
    all_training_indices = np.arange(len(training_dataset["group_ids"]))
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
        margin_models[model_seed] = margin_model
        regression_models[model_seed] = regression_model
        values = _predict(
            margin_model,
            training_dataset,
            all_training_indices,
            args.device,
        )
        training_margin_logits[model_seed] = values[:, 1] - values[:, 0]
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
        for mode in METHODS:
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
                    margin_logits=training_margin_logits[model_seed],
                    layout=layout,
                    mode=mode,
                    seed=model_seed,
                    fold_index=fold,
                    args=args,
                )
                local = np.flatnonzero(fold_assignments == fold)
                oof_score[local] = prediction["score"]
                oof_safety[local] = prediction["safety_gain"]
                oof_cost[local] = prediction["cost_delta"]
                oof_budget[local] = prediction["budget_multiplier"]
                best_epochs.append(best_epoch)
                curve_rows.extend(curves)
            if np.any(~np.isfinite(oof_score)):
                raise RuntimeError("Cross-fitting left missing predictions")
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
                and all(
                    float(row["engage_recall"]) >= 0.60
                    for row in by_scenario.values()
                    if int(row["engage_count"]) > 0
                )
                and all(
                    float(row["noop_recall"]) >= 0.65
                    for row in by_scenario.values()
                    if int(row["noop_count"]) > 0
                )
            )
            crossfit_results[str(model_seed)][mode] = {
                "metrics": metrics,
                "scenario_metrics": by_scenario,
                "value_metrics": values,
                "feasible": feasible,
                "best_epochs": best_epochs,
                "budget_mean": float(np.mean(oof_budget)),
                "budget_std": float(np.std(oof_budget)),
            }
            crossfit_epochs[(model_seed, mode)] = best_epochs
            for local, dataset_index in enumerate(train_indices):
                oof_rows.append(
                    {
                        "model_seed": model_seed,
                        "mode": mode,
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

    ranked_methods: list[tuple[tuple[float, float, float, float], str]] = []
    complexity = {"safety_only": 2.0, "global_budget": 1.0, "state_budget": 0.0}
    for mode in METHODS:
        rows = [crossfit_results[str(seed)][mode] for seed in args.model_seeds]
        feasible_count = sum(bool(row["feasible"]) for row in rows)
        mean_balanced = float(
            np.mean([row["metrics"]["balanced_accuracy"] for row in rows])
        )
        mean_sign = float(
            np.mean([row["value_metrics"]["safety_sign_accuracy"] for row in rows])
        )
        ranked_methods.append(
            ((float(feasible_count), mean_balanced, mean_sign, complexity[mode]), mode)
        )
    ranked_methods.sort(reverse=True)
    selected_mode = ranked_methods[0][1]

    model_results: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    per_seed_gates: dict[str, Any] = {}
    passed_seed_count = 0
    inference_seconds: dict[str, float] = {}
    for model_seed in args.model_seeds:
        final_epochs = max(
            1,
            int(
                np.median(crossfit_epochs[(model_seed, selected_mode)])
            )
            + 1,
        )
        model, scales, margin_scale = _fit_final(
            dataset=training_dataset,
            fit=train_indices,
            safety_targets=training_targets["safety_gain"],
            cost_targets=training_targets["cost_delta"],
            oracle_labels=training_oracle,
            margin_logits=training_margin_logits[model_seed],
            layout=layout,
            mode=selected_mode,
            seed=model_seed,
            epochs=final_epochs,
            args=args,
        )
        started = perf_counter()
        prediction = _predict_value(
            model,
            test_dataset,
            test_indices,
            args.device,
            test_margin_logits[model_seed],
            scales,
            margin_scale,
        )
        inference_seconds[str(model_seed)] = perf_counter() - started
        predicted = (prediction["score"] > 0.0).astype(np.int64)
        valid = test_oracle >= 0
        candidate = {
            "metrics": oracle_classification_metrics(test_oracle, predicted),
            "scenario_metrics": scenario_classification_metrics(
                test_oracle, predicted, test_dataset["scenarios"]
            ),
            "value_metrics": constrained_value_metrics(
                test_targets["safety_gain"][valid],
                test_targets["cost_delta"][valid],
                prediction["safety_gain"][valid],
                prediction["cost_delta"][valid],
            ),
            "budget": {
                "mean": float(np.mean(prediction["budget_multiplier"])),
                "std": float(np.std(prediction["budget_multiplier"])),
                "min": float(np.min(prediction["budget_multiplier"])),
                "max": float(np.max(prediction["budget_multiplier"])),
            },
            "final_epochs": final_epochs,
            "inference_seconds": inference_seconds[str(model_seed)],
        }
        zero_predictions = (test_margin_logits[model_seed] > 0.0).astype(np.int64)
        regression_predictions = (
            test_regression_logits[model_seed] > 0.0
        ).astype(np.int64)
        zero = {
            "metrics": oracle_classification_metrics(test_oracle, zero_predictions),
            "scenario_metrics": scenario_classification_metrics(
                test_oracle, zero_predictions, test_dataset["scenarios"]
            ),
        }
        regression = {
            "metrics": oracle_classification_metrics(
                test_oracle, regression_predictions
            ),
            "scenario_metrics": scenario_classification_metrics(
                test_oracle, regression_predictions, test_dataset["scenarios"]
            ),
        }
        checks = _model_gate(candidate, regression)
        checks["inference_faster_than_rollouts"] = (
            inference_seconds[str(model_seed)]
            < float(test_dataset["generation_seconds"])
        )
        passed = all(checks.values())
        passed_seed_count += int(passed)
        per_seed_gates[str(model_seed)] = {**checks, "passed": passed}
        model_results[str(model_seed)] = {
            "zero_margin": zero,
            "risk_regression": regression,
            "candidate": candidate,
        }
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_signature": model.signature(),
                "selected_mode": selected_mode,
                "scales": scales,
                "margin_scale": margin_scale,
                "final_epochs": final_epochs,
            },
            args.output_dir / "models" / f"{selected_mode}_seed{model_seed}.pt",
        )
        for method, result in (
            ("zero_margin", zero),
            ("risk_regression", regression),
            (selected_mode, candidate),
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
                    "safety_prediction": float(prediction["safety_gain"][index]),
                    "cost_target": float(test_targets["cost_delta"][index]),
                    "cost_prediction": float(prediction["cost_delta"][index]),
                    "budget_multiplier": float(
                        prediction["budget_multiplier"][index]
                    ),
                    "score": float(prediction["score"][index]),
                    "prediction": int(predicted[index]),
                }
            )
    _write_csv(args.output_dir / "model_metrics.csv", metric_rows)
    _write_csv(args.output_dir / "test_group_diagnostics.csv", diagnostics)

    overlap_paths = (*OLD_DATASETS, DEFAULT_BALANCED_DIR / "analysis_dataset.npz")
    overlaps: dict[str, int] = {}
    for path in overlap_paths:
        if not path.exists():
            continue
        previous = _load_npz(path)
        overlaps[str(path)] = _observation_overlap_count(
            test_dataset["observations"], previous["observations"]
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
            "selected_mode": selected_mode,
        },
        "model_results": model_results,
        "model_gate": {
            "per_seed": per_seed_gates,
            "passed_seed_count": passed_seed_count,
            "required_passed_seed_count": 2,
            "passed": estimator_passed,
        },
        "task14_state_conditioned_value_passed": stage_passed,
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
        "model_dir": str(args.model_dir.resolve()),
        "critic_dir": str(args.critic_dir.resolve()),
        "source_seeds": list(args.source_seeds),
        "scenarios": list(args.scenarios),
        "episodes_per_stratum": args.episodes_per_stratum,
        "states_per_stratum": args.states_per_stratum,
        "rollouts": args.rollouts,
        "gamma": args.gamma,
        "eval_seed": args.eval_seed,
        "model_seeds": list(args.model_seeds),
        "methods": list(METHODS),
        "fold_count": FOLDS,
        "fold_seed": args.fold_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "selected_mode": selected_mode,
    }
    with (args.output_dir / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
