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
    DEFAULT_HIGH_THREAT_THRESHOLD,
    EngagementUtilityConfig,
    engagement_utility_labels,
    grouped_state_split,
    oracle_classification_metrics,
    q_critic_training_loss,
    safety_resource_oracle,
    utility_oracle_metrics,
    validation_difference_score,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    RiskAwareEngagementCritic,
)
from scripts.run_air_defense_v1_task14_hierarchical_q import (
    _observation_overlap_count,
)
from scripts.run_air_defense_v1_task14_q_critic import (
    DEFAULT_MODEL_DIR,
    SCENARIOS,
    _collect_snapshot_pool,
    _distribution_and_value,
    _load_model,
    _sample_policy_action,
    _seed_everything,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task14_engagement_utility"
)
PREVIOUS_TESTS = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_q_critic"
    / "dataset.npz",
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_q_critic_ranking_refinement"
    / "test_dataset.npz",
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_hierarchical_q"
    / "test_dataset.npz",
)
COMPONENT_KEYS = (
    "total_return_samples",
    "operational_return_samples",
    "resource_cost_samples",
    "damage_samples",
    "high_threat_leak_samples",
    "shot_samples",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the risk-aware engagement utility diagnostic."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-seeds", nargs="+", type=int, default=(8, 10))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--episodes-per-stratum", type=int, default=36)
    parser.add_argument("--states-per-stratum", type=int, default=18)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--eval-seed", type=int, default=337_000)
    parser.add_argument("--split-seed", type=int, default=31)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=(17, 18, 19))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-dataset", action="store_true")
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _high_threat_leak_count(env: AirDefenseResourceAssignmentEnvV1) -> int:
    return sum(
        target.status == "leaked" and target.threat >= DEFAULT_HIGH_THREAT_THRESHOLD
        for target in env.targets
    )


def _conditioned_branch_outcome(
    *,
    model: Any,
    snapshot: dict[str, Any],
    fixed_actions: np.ndarray,
    gamma: float,
    random_seed: int,
) -> dict[str, float]:
    env = deepcopy(snapshot["env"])
    initial_high_leaks = _high_threat_leak_count(env)
    env.np_random = np.random.default_rng(random_seed)
    torch.manual_seed(random_seed)
    distribution, _ = _distribution_and_value(
        model, snapshot["observation"], env.action_masks()
    )
    fixed_tensor = torch.as_tensor(
        fixed_actions, device=distribution.target_logits.device
    ).reshape(1, -1)
    first_action = (
        distribution.sample_with_fixed_actions(fixed_tensor)
        .actions[0]
        .detach()
        .cpu()
        .numpy()
    )

    total_return = 0.0
    operational_return = 0.0
    resource_cost = 0.0
    damage = 0.0
    shots = 0.0
    discount = 1.0
    action = first_action
    while True:
        _, reward, terminated, truncated, info = env.step(action)
        breakdown = info["reward_breakdown"]
        total_return += discount * float(reward)
        operational_return += discount * float(
            sum(
                value
                for key, value in breakdown.items()
                if key not in {"cost", "damage"}
            )
        )
        resource_cost += discount * max(0.0, -float(breakdown["cost"]))
        damage += discount * float(info["damage_this_step"])
        shots += float(info["shots"])
        if terminated or truncated:
            break
        discount *= gamma
        action = _sample_policy_action(model, env)

    high_threat_leaks = _high_threat_leak_count(env) - initial_high_leaks
    env.close()
    return {
        "total_return_samples": total_return,
        "operational_return_samples": operational_return,
        "resource_cost_samples": resource_cost,
        "damage_samples": damage,
        "high_threat_leak_samples": float(high_threat_leaks),
        "shot_samples": shots,
    }


def _snapshot_groups(
    *,
    model: Any,
    snapshot: dict[str, Any],
    gamma: float,
    rollouts: int,
    base_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    metadata: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    base_action = snapshot["base_action"]
    num_units = len(base_action)
    num_actions = snapshot["probabilities"].shape[1]
    num_targets = num_actions - 1
    for unit_index in range(num_units):
        legal_mask = snapshot["conditional_masks"][unit_index].astype(bool)
        target_actions = np.flatnonzero(legal_mask[:num_targets])
        if len(target_actions) == 0:
            continue
        prefix_occupancy = np.zeros(num_targets, dtype=np.float32)
        fixed_prefix = np.full(num_units, -1, dtype=np.int64)
        for earlier_unit in range(unit_index):
            earlier_action = int(base_action[earlier_unit])
            fixed_prefix[earlier_unit] = earlier_action
            if earlier_action != num_targets:
                prefix_occupancy[earlier_action] = 1.0

        probabilities = snapshot["probabilities"][unit_index, target_actions].astype(
            np.float64
        )
        probabilities /= np.sum(probabilities)
        choice_rng = np.random.default_rng(base_seed + unit_index * 100_000 + 91_337)
        selected_targets = choice_rng.choice(
            target_actions, size=rollouts, replace=True, p=probabilities
        )
        branch_components = {
            key: np.empty((2, rollouts), dtype=np.float32) for key in COMPONENT_KEYS
        }
        for rollout_index, selected_target in enumerate(selected_targets):
            random_seed = base_seed + unit_index * 100_000 + rollout_index
            fixed_noop = fixed_prefix.copy()
            fixed_noop[unit_index] = num_targets
            fixed_engage = fixed_prefix.copy()
            fixed_engage[unit_index] = int(selected_target)
            noop = _conditioned_branch_outcome(
                model=model,
                snapshot=snapshot,
                fixed_actions=fixed_noop,
                gamma=gamma,
                random_seed=random_seed,
            )
            engage = _conditioned_branch_outcome(
                model=model,
                snapshot=snapshot,
                fixed_actions=fixed_engage,
                gamma=gamma,
                random_seed=random_seed,
            )
            for key in COMPONENT_KEYS:
                branch_components[key][0, rollout_index] = noop[key]
                branch_components[key][1, rollout_index] = engage[key]

        group_id = f"task14u/{snapshot['state_id']}/unit{unit_index}"
        harm_delta = (
            30.0
            * (branch_components["damage_samples"][1] - branch_components["damage_samples"][0])
            + 20.0
            * (
                branch_components["high_threat_leak_samples"][1]
                - branch_components["high_threat_leak_samples"][0]
            )
        )
        metadata.append(
            {
                "group_id": group_id,
                "state_id": f"task14u/{snapshot['state_id']}",
                "source_seed": snapshot["source_seed"],
                "scenario": snapshot["scenario"],
                "unit_index": unit_index,
                "rollouts": rollouts,
                "mean_total_return_delta": float(
                    np.mean(
                        branch_components["total_return_samples"][1]
                        - branch_components["total_return_samples"][0]
                    )
                ),
                "mean_harm_delta": float(np.mean(harm_delta)),
                "mean_cost_delta": float(
                    np.mean(
                        branch_components["resource_cost_samples"][1]
                        - branch_components["resource_cost_samples"][0]
                    )
                ),
            }
        )
        arrays.append(
            {
                "observation": snapshot["observation"].astype(np.float32),
                "unit_index": np.asarray(unit_index, dtype=np.int64),
                "prefix_occupancy": prefix_occupancy,
                "legal_action_mask": legal_mask.astype(np.float32),
                **branch_components,
            }
        )
    return metadata, arrays


def _generate_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
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
            snapshots = _collect_snapshot_pool(
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
                metadata.extend(rows)
                arrays.extend(values)
                state_counter += 1
            print(
                f"utility dataset source_seed={source_seed} scenario={scenario} "
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
    _write_csv(args.output_dir / "dataset_groups.csv", metadata)
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
        "generation_seconds": np.asarray(perf_counter() - started),
        "state_count": np.asarray(len(np.unique(state_ids))),
    }
    for key in COMPONENT_KEYS:
        dataset[key] = np.stack([row[key] for row in arrays])
    np.savez_compressed(args.output_dir / "dataset.npz", **dataset)
    return dataset


def _components(dataset: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: dataset[key][indices] for key in COMPONENT_KEYS}


def _baseline_labels(dataset: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean(dataset["total_return_samples"], axis=2).astype(np.float32)


def _candidate_configs() -> list[EngagementUtilityConfig]:
    return [
        EngagementUtilityConfig(cost, damage, high, cvar, 0.25)
        for cost in (0.5, 1.0, 1.5, 2.0)
        for damage in (20.0, 30.0, 40.0)
        for high in (0.0, 10.0, 20.0)
        for cvar in (0.0, 0.25, 0.5)
    ]


def _config_distance(config: EngagementUtilityConfig) -> float:
    return (
        abs(config.cost_weight - 1.0)
        + abs(config.damage_weight - 30.0) / 10.0
        + config.high_threat_leak_weight / 10.0
        + config.cvar_weight * 2.0
    )


def _select_utility(
    dataset: dict[str, np.ndarray], validation: np.ndarray
) -> tuple[EngagementUtilityConfig, list[dict[str, Any]]]:
    components = _components(dataset, validation)
    oracle = safety_resource_oracle(components)["labels"]
    rows: list[dict[str, Any]] = []
    ranked: list[tuple[tuple[float, ...], EngagementUtilityConfig]] = []
    for config in _candidate_configs():
        labels, _ = engagement_utility_labels(components, config)
        metrics = utility_oracle_metrics(labels, oracle)
        balanced = float(metrics["balanced_accuracy"])
        minimum_recall = min(
            float(metrics["engage_recall"]), float(metrics["noop_recall"])
        )
        if not np.isfinite(balanced):
            balanced = minimum_recall = -1.0
        score = (balanced, minimum_recall, -_config_distance(config))
        ranked.append((score, config))
        rows.append({**config.signature(), **metrics, "selection_score": balanced})
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1], rows


def _model_inputs(
    dataset: dict[str, np.ndarray], indices: np.ndarray, device: str
) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(dataset["observations"][indices], device=device),
        torch.as_tensor(dataset["unit_indices"][indices], device=device),
        torch.as_tensor(dataset["prefix_occupancy"][indices], device=device),
        torch.as_tensor(dataset["legal_action_masks"][indices], device=device),
    )


def _train_model(
    *,
    dataset: dict[str, np.ndarray],
    labels: np.ndarray,
    layout: AirDefenseV1ObservationLayout,
    method: str,
    train_seed: int,
    args: argparse.Namespace,
) -> tuple[RiskAwareEngagementCritic, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    _seed_everything(train_seed)
    train = np.flatnonzero(dataset["splits"] == "train")
    validation = np.flatnonzero(dataset["splits"] == "validation")
    test = np.flatnonzero(dataset["splits"] == "test")
    mean = float(np.mean(labels[train]))
    scale = max(float(np.std(labels[train])), 1e-6)
    normalized = torch.as_tensor(
        (labels[train] - mean) / scale, device=args.device, dtype=torch.float32
    )
    group_codes = torch.arange(len(train), device=args.device).repeat_interleave(2)
    model = RiskAwareEngagementCritic(layout).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    curves: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        prediction = model(*_model_inputs(dataset, train, args.device))
        loss, parts = q_critic_training_loss(
            prediction.reshape(-1),
            normalized.reshape(-1),
            group_codes,
            centered_weight=1.0,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = (
                model(*_model_inputs(dataset, validation, args.device)).cpu().numpy()
                * scale
                + mean
            )
        validation_metrics = validation_difference_score(
            labels[validation].reshape(-1),
            validation_prediction.reshape(-1),
            np.repeat(dataset["group_ids"][validation], 2),
            scale=scale,
        )
        curves.append(
            {
                "method": method,
                "train_seed": train_seed,
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "absolute_loss": float(parts["absolute"].detach().cpu()),
                "centered_loss": float(parts["centered"].detach().cpu()),
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
        )
        if validation_metrics["score"] < best_score - 1e-6:
            best_score = float(validation_metrics["score"])
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
        raise RuntimeError("Engagement utility training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    started = perf_counter()
    with torch.no_grad():
        test_prediction = (
            model(*_model_inputs(dataset, test, args.device)).cpu().numpy() * scale + mean
        )
    inference_seconds = perf_counter() - started
    record = {
        "method": method,
        "train_seed": train_seed,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "label_mean": mean,
        "label_scale": scale,
        "test_inference_seconds": inference_seconds,
        "parameter_count": model.parameter_count(),
    }
    return model, test_prediction, curves, record


def _scenario_metrics(
    dataset: dict[str, np.ndarray], indices: np.ndarray, predictions: np.ndarray
) -> dict[str, dict[str, float | int]]:
    values: dict[str, dict[str, float | int]] = {}
    for scenario in np.unique(dataset["scenarios"][indices]):
        selected = dataset["scenarios"][indices] == scenario
        oracle = safety_resource_oracle(_components(dataset, indices[selected]))["labels"]
        labels = (predictions[selected, 1] > predictions[selected, 0]).astype(np.int64)
        values[str(scenario)] = oracle_classification_metrics(oracle, labels)
    return values


def _model_gate(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, bool]:
    return {
        "balanced_accuracy": candidate["balanced_accuracy"] >= 0.70,
        "improvement": (
            candidate["balanced_accuracy"] - baseline["balanced_accuracy"] >= 0.10
        ),
        "false_noop_noninferior": (
            candidate["false_noop_rate"] <= baseline["false_noop_rate"]
        ),
        "wasteful_engage_noninferior": (
            candidate["wasteful_engage_rate"] <= baseline["wasteful_engage_rate"]
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)
    dataset_path = args.output_dir / "dataset.npz"
    dataset = (
        _load_npz(dataset_path)
        if args.reuse_dataset
        else _generate_dataset(args)
    )
    environment = AirDefenseResourceAssignmentEnvV1()
    layout = AirDefenseV1ObservationLayout.infer(
        environment.observation_space, environment.action_space
    )
    environment.close()

    overlaps: dict[str, int] = {}
    for previous_path in PREVIOUS_TESTS:
        if previous_path.exists():
            previous = _load_npz(previous_path)
            previous_selected = (
                previous["splits"] == "test"
                if "splits" in previous
                else np.ones(len(previous["observations"]), dtype=bool)
            )
            overlaps[str(previous_path)] = _observation_overlap_count(
                dataset["observations"], previous["observations"][previous_selected]
            )

    train = np.flatnonzero(dataset["splits"] == "train")
    validation = np.flatnonzero(dataset["splits"] == "validation")
    test = np.flatnonzero(dataset["splits"] == "test")
    selected_config, grid_rows = _select_utility(dataset, validation)
    _write_csv(args.output_dir / "utility_grid.csv", grid_rows)
    baseline_labels = _baseline_labels(dataset)
    candidate_labels, _ = engagement_utility_labels(
        _components(dataset, np.arange(len(dataset["group_ids"]))), selected_config
    )
    test_oracle = safety_resource_oracle(_components(dataset, test))["labels"]
    baseline_utility_metrics = utility_oracle_metrics(
        baseline_labels[test], test_oracle
    )
    candidate_utility_metrics = utility_oracle_metrics(
        candidate_labels[test], test_oracle
    )

    all_curves: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    model_results: dict[int, dict[str, Any]] = {}
    for train_seed in args.train_seeds:
        model_results[train_seed] = {}
        for method, labels in (
            ("mean_return", baseline_labels),
            ("risk_constraint", candidate_labels),
        ):
            model, predictions, curves, record = _train_model(
                dataset=dataset,
                labels=labels,
                layout=layout,
                method=method,
                train_seed=train_seed,
                args=args,
            )
            metrics = oracle_classification_metrics(
                test_oracle,
                (predictions[:, 1] > predictions[:, 0]).astype(np.int64),
            )
            scenarios = _scenario_metrics(dataset, test, predictions)
            model_results[train_seed][method] = {
                "metrics": metrics,
                "scenario_metrics": scenarios,
                "record": record,
            }
            model_rows.append({**record, **metrics})
            all_curves.extend(curves)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_signature": model.signature(),
                    "utility_config": (
                        selected_config.signature()
                        if method == "risk_constraint"
                        else {"type": "environment_mean_return"}
                    ),
                    "normalization": {
                        "mean": record["label_mean"],
                        "scale": record["label_scale"],
                    },
                },
                args.output_dir / "models" / f"{method}_seed{train_seed}.pt",
            )
    _write_csv(args.output_dir / "training_curves.csv", all_curves)
    _write_csv(args.output_dir / "model_metrics.csv", model_rows)

    test_oracle_details = safety_resource_oracle(_components(dataset, test))
    diagnostic_rows: list[dict[str, Any]] = []
    for local_index, dataset_index in enumerate(test):
        diagnostic_rows.append(
            {
                "group_id": str(dataset["group_ids"][dataset_index]),
                "scenario": str(dataset["scenarios"][dataset_index]),
                "source_seed": int(dataset["source_seeds"][dataset_index]),
                "oracle_label": int(test_oracle_details["labels"][local_index]),
                "harm_delta": float(test_oracle_details["harm_delta"][local_index]),
                "harm_standard_error": float(
                    test_oracle_details["harm_standard_error"][local_index]
                ),
                "cost_delta": float(test_oracle_details["cost_delta"][local_index]),
                "cost_standard_error": float(
                    test_oracle_details["cost_standard_error"][local_index]
                ),
                "baseline_utility_delta": float(
                    baseline_labels[dataset_index, 1] - baseline_labels[dataset_index, 0]
                ),
                "candidate_utility_delta": float(
                    candidate_labels[dataset_index, 1] - candidate_labels[dataset_index, 0]
                ),
            }
        )
    _write_csv(args.output_dir / "test_group_diagnostics.csv", diagnostic_rows)

    valid_oracle = test_oracle >= 0
    engage_count = int(np.sum(test_oracle == 1))
    noop_count = int(np.sum(test_oracle == 0))
    scenario_counts = {
        str(scenario): int(
            np.sum(valid_oracle & (dataset["scenarios"][test] == scenario))
        )
        for scenario in np.unique(dataset["scenarios"][test])
    }
    power_sufficient = (
        int(np.sum(valid_oracle)) >= 20
        and engage_count >= 8
        and noop_count >= 8
        and all(count >= 5 for count in scenario_counts.values())
    )
    reconstructed_returns = (
        dataset["operational_return_samples"]
        - dataset["resource_cost_samples"]
        - 30.0 * dataset["damage_samples"]
    )
    reconstruction_max_error = float(
        np.max(np.abs(dataset["total_return_samples"] - reconstructed_returns))
    )
    state_split_leakage = any(
        len(set(dataset["splits"][dataset["state_ids"] == state_id])) != 1
        for state_id in np.unique(dataset["state_ids"])
    )
    all_overlaps_zero = all(count == 0 for count in overlaps.values())
    data_integrity = (
        not state_split_leakage
        and all_overlaps_zero
        and reconstruction_max_error <= 1e-4
    )
    utility_gate = {
        "balanced_accuracy": candidate_utility_metrics["balanced_accuracy"] >= 0.70,
        "improvement": (
            candidate_utility_metrics["balanced_accuracy"]
            - baseline_utility_metrics["balanced_accuracy"]
            >= 0.10
        ),
        "false_noop_noninferior": (
            candidate_utility_metrics["false_noop_rate"]
            <= baseline_utility_metrics["false_noop_rate"]
        ),
        "wasteful_engage_noninferior": (
            candidate_utility_metrics["wasteful_engage_rate"]
            <= baseline_utility_metrics["wasteful_engage_rate"]
        ),
    }
    passed_seeds = 0
    per_seed_gates: dict[str, Any] = {}
    for train_seed in args.train_seeds:
        baseline = model_results[train_seed]["mean_return"]["metrics"]
        candidate = model_results[train_seed]["risk_constraint"]["metrics"]
        checks = _model_gate(candidate, baseline)
        passed = all(checks.values())
        passed_seeds += int(passed)
        per_seed_gates[str(train_seed)] = {**checks, "passed": passed}
    utility_passed = power_sufficient and all(utility_gate.values())
    estimator_passed = power_sufficient and passed_seeds >= 2
    stage_passed = data_integrity and utility_passed and estimator_passed
    summary = {
        "schema_version": 1,
        "isolation_audit": {
            "previous_observation_overlaps": overlaps,
            "all_previous_observation_overlaps_zero": all_overlaps_zero,
            "state_split_leakage": state_split_leakage,
            "return_reconstruction_max_error": reconstruction_max_error,
            "data_integrity_passed": data_integrity,
        },
        "dataset": {
            "states": int(dataset["state_count"]),
            "groups": int(len(dataset["group_ids"])),
            "train_groups": int(len(train)),
            "validation_groups": int(len(validation)),
            "test_groups": int(len(test)),
            "rollouts": int(dataset["total_return_samples"].shape[2]),
            "generation_seconds": float(dataset["generation_seconds"]),
        },
        "selected_utility": selected_config.signature(),
        "test_oracle": {
            "valid_count": int(np.sum(valid_oracle)),
            "engage_count": engage_count,
            "noop_count": noop_count,
            "scenario_counts": scenario_counts,
            "power_sufficient": power_sufficient,
        },
        "utility_metrics": {
            "mean_return": baseline_utility_metrics,
            "risk_constraint": candidate_utility_metrics,
            "gate": utility_gate,
            "passed": utility_passed,
        },
        "model_results": model_results,
        "model_gate": {
            "per_seed": per_seed_gates,
            "passed_seed_count": passed_seeds,
            "required_passed_seed_count": 2,
            "passed": estimator_passed,
        },
        "task14_engagement_utility_passed": stage_passed,
        "resume_mch_ppo": stage_passed,
        "enter_gnn": False,
    }
    with (args.output_dir / "gate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    config = {
        "schema_version": 1,
        "model_dir": str(args.model_dir.resolve()),
        "source_seeds": list(args.source_seeds),
        "scenarios": list(args.scenarios),
        "episodes_per_stratum": args.episodes_per_stratum,
        "states_per_stratum": args.states_per_stratum,
        "rollouts": args.rollouts,
        "gamma": args.gamma,
        "eval_seed": args.eval_seed,
        "split_seed": args.split_seed,
        "validation_fraction": 0.2,
        "test_fraction": 0.4,
        "train_seeds": list(args.train_seeds),
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "selected_utility": selected_config.signature(),
    }
    with (args.output_dir / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
