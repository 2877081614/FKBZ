from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.algorithms.policy_gradient import FactorizedEngagementMaskablePPO
from rein_learning.common import (
    engagement_sign_accuracy,
    grouped_state_split,
    pairwise_ranking_accuracy,
    regression_metrics,
    top_action_accuracy,
)
from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    get_air_defense_v1_scenario,
)
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    MaskedActionQCritic,
    MaskedActionQCriticConfig,
)


DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task12_factorized_screening_30k_3seeds"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task14_q_critic"
)
METHOD = "factorized_engagement_ar_ppo_order_012"
SCENARIOS = ("medium", "time_pressure", "heterogeneity_pressure")
VARIANTS = {
    "full": MaskedActionQCriticConfig(),
    "no_prefix": MaskedActionQCriticConfig(include_prefix_occupancy=False),
    "no_mask": MaskedActionQCriticConfig(include_legal_mask=False),
    "observation_action_only": MaskedActionQCriticConfig(
        include_entity_features=False,
        include_prefix_occupancy=False,
        include_legal_mask=False,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate the Task 14 masked action Q-Critic."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-seeds", nargs="+", type=int, default=(8, 10))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--episodes-per-stratum", type=int, default=12)
    parser.add_argument("--states-per-stratum", type=int, default=10)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--eval-seed", type=int, default=41_000)
    parser.add_argument("--split-seed", type=int, default=14)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=(14, 15, 16))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-dataset", action="store_true")
    parser.add_argument("--dataset-only", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_model(model_dir: Path, seed: int, device: str) -> Any:
    path = model_dir / "models" / "medium" / f"{METHOD}_seed{seed}.zip"
    if not path.exists():
        raise FileNotFoundError(path)
    return FactorizedEngagementMaskablePPO.load(path, device=device)


def _distribution_and_value(
    model: Any,
    observation: np.ndarray,
    action_mask: np.ndarray,
) -> tuple[Any, float]:
    observation_tensor, _ = model.policy.obs_to_tensor(observation)
    distribution = model.policy.get_distribution(
        observation_tensor, action_masks=action_mask
    )
    value = float(model.policy.predict_values(observation_tensor).detach().cpu().item())
    return distribution, value


def _sample_policy_action(model: Any, env: AirDefenseResourceAssignmentEnvV1) -> np.ndarray:
    distribution, _ = _distribution_and_value(
        model, env._get_observation(), env.action_masks()
    )
    return distribution.sample(deterministic=False).actions[0].detach().cpu().numpy()


def _collect_snapshot_pool(
    *,
    model: Any,
    source_seed: int,
    scenario: str,
    episodes: int,
    state_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    snapshots: list[dict[str, Any]] = []
    eligible_count = 0
    for episode_index in range(episodes):
        env = AirDefenseResourceAssignmentEnvV1(get_air_defense_v1_scenario(scenario))
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
                record = {
                    "state_id": (
                        f"seed{source_seed}/{scenario}/episode{episode_index}/"
                        f"step{step_index}"
                    ),
                    "source_seed": source_seed,
                    "scenario": scenario,
                    "observation": observation.copy(),
                    "env": deepcopy(env),
                    "base_action": action.copy(),
                    "probabilities": probabilities[0].detach().cpu().numpy(),
                    "conditional_masks": conditional_masks,
                    "critic_v": value,
                }
                eligible_count += 1
                if len(snapshots) < state_count:
                    snapshots.append(record)
                else:
                    replacement = int(rng.integers(eligible_count))
                    if replacement < state_count:
                        snapshots[replacement] = record
            observation, _, terminated, truncated, _ = env.step(action)
            step_index += 1
        env.close()
    if len(snapshots) < state_count:
        raise RuntimeError(
            f"Only collected {len(snapshots)} states for seed={source_seed}, "
            f"scenario={scenario}; requested {state_count}"
        )
    return snapshots


def _conditioned_branch_return(
    *,
    model: Any,
    snapshot: dict[str, Any],
    fixed_actions: np.ndarray,
    gamma: float,
    random_seed: int,
) -> tuple[float, float]:
    env = deepcopy(snapshot["env"])
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
    _, reward, terminated, truncated, _ = env.step(first_action)
    one_step_reward = float(reward)
    total_return = one_step_reward
    discount = gamma
    while not (terminated or truncated):
        action = _sample_policy_action(model, env)
        _, reward, terminated, truncated, _ = env.step(action)
        total_return += discount * float(reward)
        discount *= gamma
    env.close()
    return total_return, one_step_reward


def _snapshot_samples(
    *,
    model: Any,
    snapshot: dict[str, Any],
    gamma: float,
    rollouts: int,
    base_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    metadata_rows: list[dict[str, Any]] = []
    array_rows: list[dict[str, np.ndarray]] = []
    action = snapshot["base_action"]
    num_units = len(action)
    num_actions = snapshot["probabilities"].shape[1]
    num_targets = num_actions - 1
    for unit_index in range(num_units):
        legal_mask = snapshot["conditional_masks"][unit_index].astype(bool)
        candidates = np.flatnonzero(legal_mask).tolist()
        prefix_occupancy = np.zeros(num_targets, dtype=np.float32)
        fixed_prefix = np.full(num_units, -1, dtype=np.int64)
        for earlier_unit in range(unit_index):
            earlier_action = int(action[earlier_unit])
            fixed_prefix[earlier_unit] = earlier_action
            if earlier_action != num_targets:
                prefix_occupancy[earlier_action] = 1.0

        returns_by_action: dict[int, np.ndarray] = {}
        immediate_by_action: dict[int, np.ndarray] = {}
        for candidate in candidates:
            fixed = fixed_prefix.copy()
            fixed[unit_index] = candidate
            returns: list[float] = []
            immediate_rewards: list[float] = []
            for rollout_index in range(rollouts):
                random_seed = (
                    base_seed
                    + unit_index * 100_000
                    + rollout_index
                )
                total_return, immediate = _conditioned_branch_return(
                    model=model,
                    snapshot=snapshot,
                    fixed_actions=fixed,
                    gamma=gamma,
                    random_seed=random_seed,
                )
                returns.append(total_return)
                immediate_rewards.append(immediate)
            returns_by_action[candidate] = np.asarray(returns, dtype=np.float64)
            immediate_by_action[candidate] = np.asarray(
                immediate_rewards, dtype=np.float64
            )

        noop_returns = returns_by_action[num_targets]
        engage_probability = float(
            snapshot["probabilities"][unit_index, :num_targets].sum()
        )
        for candidate in candidates:
            returns = returns_by_action[candidate]
            immediate = immediate_by_action[candidate]
            paired = returns - noop_returns
            policy_probability = float(
                snapshot["probabilities"][unit_index, candidate]
            )
            conditional_target_probability = (
                policy_probability / max(engage_probability, 1e-20)
                if candidate != num_targets
                else 0.0
            )
            sample_id = f"{snapshot['state_id']}/unit{unit_index}/action{candidate}"
            metadata_rows.append(
                {
                    "sample_id": sample_id,
                    "state_id": snapshot["state_id"],
                    "source_seed": snapshot["source_seed"],
                    "scenario": snapshot["scenario"],
                    "unit_index": unit_index,
                    "candidate_action": candidate,
                    "candidate_type": (
                        "noop" if candidate == num_targets else "target"
                    ),
                    "policy_probability": policy_probability,
                    "conditional_target_probability": conditional_target_probability,
                    "mc_q_mean": float(np.mean(returns)),
                    "mc_q_std": float(np.std(returns, ddof=1)),
                    "mc_q_se": float(
                        np.std(returns, ddof=1) / np.sqrt(rollouts)
                    ),
                    "paired_delta_vs_noop_mean": float(np.mean(paired)),
                    "paired_delta_vs_noop_se": float(
                        np.std(paired, ddof=1) / np.sqrt(rollouts)
                    ),
                    "one_step_reward_mean": float(np.mean(immediate)),
                    "frozen_v": snapshot["critic_v"],
                    "rollouts": rollouts,
                }
            )
            array_rows.append(
                {
                    "observation": snapshot["observation"].astype(np.float32),
                    "unit_index": np.asarray(unit_index, dtype=np.int64),
                    "candidate_action": np.asarray(candidate, dtype=np.int64),
                    "prefix_occupancy": prefix_occupancy.copy(),
                    "legal_action_mask": legal_mask.astype(np.float32),
                    "return_samples": returns.astype(np.float32),
                }
            )
    return metadata_rows, array_rows


def _generate_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
    all_metadata: list[dict[str, Any]] = []
    all_arrays: list[dict[str, np.ndarray]] = []
    generation_started = perf_counter()
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
                rows, arrays = _snapshot_samples(
                    model=model,
                    snapshot=snapshot,
                    gamma=args.gamma,
                    rollouts=args.rollouts,
                    base_seed=args.eval_seed + state_counter * 1_000_000,
                )
                all_metadata.extend(rows)
                all_arrays.extend(arrays)
                state_counter += 1
            print(
                f"dataset source_seed={source_seed} scenario={scenario} "
                f"states={len(snapshots)} rows={len(all_metadata)}",
                flush=True,
            )

    state_ids = np.asarray([row["state_id"] for row in all_metadata])
    strata = np.asarray(
        [f"seed{row['source_seed']}/{row['scenario']}" for row in all_metadata]
    )
    splits = grouped_state_split(
        state_ids,
        strata=strata,
        seed=args.split_seed,
    )
    for row, split in zip(all_metadata, splits.tolist()):
        row["split"] = split
    _write_csv(args.output_dir / "dataset_samples.csv", all_metadata)

    dataset = {
        "observations": np.stack([row["observation"] for row in all_arrays]),
        "unit_indices": np.asarray(
            [row["unit_index"] for row in all_arrays], dtype=np.int64
        ),
        "candidate_actions": np.asarray(
            [row["candidate_action"] for row in all_arrays], dtype=np.int64
        ),
        "prefix_occupancy": np.stack(
            [row["prefix_occupancy"] for row in all_arrays]
        ),
        "legal_action_masks": np.stack(
            [row["legal_action_mask"] for row in all_arrays]
        ),
        "q_labels": np.asarray(
            [row["mc_q_mean"] for row in all_metadata], dtype=np.float32
        ),
        "return_samples": np.stack(
            [row["return_samples"] for row in all_arrays]
        ),
        "q_standard_errors": np.asarray(
            [row["mc_q_se"] for row in all_metadata], dtype=np.float32
        ),
        "one_step_rewards": np.asarray(
            [row["one_step_reward_mean"] for row in all_metadata], dtype=np.float32
        ),
        "frozen_values": np.asarray(
            [row["frozen_v"] for row in all_metadata], dtype=np.float32
        ),
        "conditional_target_probabilities": np.asarray(
            [row["conditional_target_probability"] for row in all_metadata],
            dtype=np.float32,
        ),
        "state_ids": state_ids,
        "scenarios": np.asarray([row["scenario"] for row in all_metadata]),
        "source_seeds": np.asarray(
            [row["source_seed"] for row in all_metadata], dtype=np.int64
        ),
        "splits": splits,
        "generation_seconds": np.asarray(
            perf_counter() - generation_started, dtype=np.float64
        ),
        "state_count": np.asarray(len(np.unique(state_ids)), dtype=np.int64),
    }
    np.savez_compressed(args.output_dir / "dataset.npz", **dataset)
    return dataset


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _model_inputs(
    dataset: dict[str, np.ndarray], indices: np.ndarray, device: str
) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(dataset["observations"][indices], device=device),
        torch.as_tensor(dataset["unit_indices"][indices], device=device),
        torch.as_tensor(dataset["candidate_actions"][indices], device=device),
        torch.as_tensor(dataset["prefix_occupancy"][indices], device=device),
        torch.as_tensor(dataset["legal_action_masks"][indices], device=device),
    )


def _train_model(
    *,
    dataset: dict[str, np.ndarray],
    layout: AirDefenseV1ObservationLayout,
    variant: str,
    config: MaskedActionQCriticConfig,
    train_seed: int,
    args: argparse.Namespace,
) -> tuple[MaskedActionQCritic, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    _seed_everything(train_seed)
    model = MaskedActionQCritic(layout, config).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_indices = np.flatnonzero(dataset["splits"] == "train")
    validation_indices = np.flatnonzero(dataset["splits"] == "validation")
    q_mean = float(np.mean(dataset["q_labels"][train_indices]))
    q_std = float(np.std(dataset["q_labels"][train_indices]))
    q_std = max(q_std, 1e-6)
    labels = torch.as_tensor(
        (dataset["q_labels"] - q_mean) / q_std,
        device=args.device,
        dtype=torch.float32,
    )
    rng = np.random.default_rng(train_seed)
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    curve_rows: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        shuffled = train_indices.copy()
        rng.shuffle(shuffled)
        batch_losses: list[float] = []
        for start in range(0, len(shuffled), args.batch_size):
            batch = shuffled[start : start + args.batch_size]
            prediction = model(*_model_inputs(dataset, batch, args.device))
            loss = torch.nn.functional.mse_loss(prediction, labels[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_prediction = (
                model(*_model_inputs(dataset, validation_indices, args.device))
                .cpu()
                .numpy()
                * q_std
                + q_mean
            )
        validation_mae = float(
            np.mean(
                np.abs(
                    validation_prediction
                    - dataset["q_labels"][validation_indices]
                )
            )
        )
        curve_rows.append(
            {
                "variant": variant,
                "train_seed": train_seed,
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)),
                "validation_mae": validation_mae,
            }
        )
        if validation_mae < best_validation - 1e-6:
            best_validation = validation_mae
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Q-Critic training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    all_indices = np.arange(len(dataset["q_labels"]))
    inference_started = perf_counter()
    with torch.no_grad():
        predictions = (
            model(*_model_inputs(dataset, all_indices, args.device)).cpu().numpy()
            * q_std
            + q_mean
        )
    inference_seconds = perf_counter() - inference_started
    training_record = {
        "variant": variant,
        "train_seed": train_seed,
        "best_epoch": best_epoch,
        "best_validation_mae": best_validation,
        "q_mean": q_mean,
        "q_std": q_std,
        "parameter_count": model.parameter_count(),
        "inference_seconds": inference_seconds,
    }
    return model, predictions, curve_rows, training_record


def _metric_rows(
    *,
    dataset: dict[str, np.ndarray],
    predictions: np.ndarray,
    variant: str,
    train_seed: int,
    layout: AirDefenseV1ObservationLayout,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    test = np.flatnonzero(dataset["splits"] == "test")
    labels = dataset["q_labels"][test]
    predicted = predictions[test]
    standard_errors = dataset["q_standard_errors"][test]
    return_samples = dataset["return_samples"][test]
    actions = dataset["candidate_actions"][test]
    group_ids = np.asarray(
        [
            f"{state_id}/unit{unit_index}"
            for state_id, unit_index in zip(
                dataset["state_ids"][test], dataset["unit_indices"][test]
            )
        ]
    )
    conditional_probabilities = dataset["conditional_target_probabilities"][test]
    q_metrics = regression_metrics(labels, predicted)
    v_metrics = regression_metrics(labels, dataset["frozen_values"][test])
    one_step_metrics = regression_metrics(labels, dataset["one_step_rewards"][test])
    ranking = pairwise_ranking_accuracy(
        labels,
        predicted,
        group_ids,
        standard_errors=standard_errors,
        return_samples=return_samples,
    )
    target_ranking = pairwise_ranking_accuracy(
        labels,
        predicted,
        group_ids,
        standard_errors=standard_errors,
        return_samples=return_samples,
        candidate_actions=actions,
        noop_action=layout.num_targets,
        target_only=True,
    )
    top = top_action_accuracy(
        labels,
        predicted,
        group_ids,
        standard_errors=standard_errors,
        return_samples=return_samples,
    )
    engage = engagement_sign_accuracy(
        labels,
        predicted,
        group_ids,
        actions,
        conditional_probabilities,
        noop_action=layout.num_targets,
        standard_errors=standard_errors,
        return_samples=return_samples,
    )
    values: dict[str, Any] = {
        "q_mae": q_metrics["mae"],
        "q_rmse": q_metrics["rmse"],
        "v_mae": v_metrics["mae"],
        "one_step_mae": one_step_metrics["mae"],
        "mae_improvement_vs_v": 1.0 - q_metrics["mae"] / v_metrics["mae"],
        "ranking_accuracy": ranking["accuracy"],
        "ranking_count": ranking["count"],
        "target_ranking_accuracy": target_ranking["accuracy"],
        "target_ranking_count": target_ranking["count"],
        "top_action_accuracy": top["accuracy"],
        "top_action_count": top["count"],
        "engagement_sign_accuracy": engage["accuracy"],
        "engagement_sign_count": engage["count"],
    }
    rows = [
        {
            "variant": variant,
            "train_seed": train_seed,
            "scope": "overall",
            "metric": name,
            "value": value,
        }
        for name, value in values.items()
    ]
    for scenario in np.unique(dataset["scenarios"][test]):
        selected = dataset["scenarios"][test] == scenario
        scenario_ranking = pairwise_ranking_accuracy(
            labels[selected],
            predicted[selected],
            group_ids[selected],
            standard_errors=standard_errors[selected],
            return_samples=return_samples[selected],
        )
        values[f"scenario_{scenario}_ranking_accuracy"] = scenario_ranking[
            "accuracy"
        ]
        values[f"scenario_{scenario}_ranking_count"] = scenario_ranking["count"]
        rows.extend(
            (
                {
                    "variant": variant,
                    "train_seed": train_seed,
                    "scope": str(scenario),
                    "metric": "ranking_accuracy",
                    "value": scenario_ranking["accuracy"],
                },
                {
                    "variant": variant,
                    "train_seed": train_seed,
                    "scope": str(scenario),
                    "metric": "ranking_count",
                    "value": scenario_ranking["count"],
                },
            )
        )
    return rows, values


def _gate_record(
    metrics: dict[str, Any],
    *,
    inference_seconds: float,
    dataset: dict[str, np.ndarray],
) -> dict[str, Any]:
    state_count = int(dataset["state_count"])
    mc_seconds_per_state = float(dataset["generation_seconds"]) / state_count
    test_state_count = len(
        np.unique(dataset["state_ids"][dataset["splits"] == "test"])
    )
    q_seconds_per_state = inference_seconds / max(test_state_count, 1)
    gates = {
        "mae_improvement": metrics["mae_improvement_vs_v"] >= 0.10,
        "ranking": (
            metrics["ranking_count"] >= 30
            and metrics["ranking_accuracy"] >= 0.70
        ),
        "engagement_sign": (
            metrics["engagement_sign_count"] >= 30
            and metrics["engagement_sign_accuracy"] >= 0.70
        ),
        "target_ranking": (
            metrics["target_ranking_count"] >= 30
            and metrics["target_ranking_accuracy"] >= 0.65
        ),
        "top_action": (
            metrics["top_action_count"] >= 30
            and metrics["top_action_accuracy"] >= 0.50
        ),
        "scenario_ranking": all(
            metrics[f"scenario_{scenario}_ranking_count"] >= 30
            and metrics[f"scenario_{scenario}_ranking_accuracy"] >= 0.60
            for scenario in SCENARIOS
            if f"scenario_{scenario}_ranking_count" in metrics
        ),
        "efficiency": q_seconds_per_state < mc_seconds_per_state,
    }
    return {
        "gates": gates,
        "passed_gate_count": sum(gates.values()),
        "total_gate_count": len(gates),
        "passed": all(gates.values()),
        "q_seconds_per_test_state": q_seconds_per_state,
        "mc_generation_seconds_per_state": mc_seconds_per_state,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "dataset.npz"
    dataset = (
        _load_dataset(dataset_path)
        if args.reuse_dataset
        else _generate_dataset(args)
    )
    if args.dataset_only:
        print(f"dataset={dataset_path.resolve()}")
        return

    env = AirDefenseResourceAssignmentEnvV1(
        get_air_defense_v1_scenario("medium")
    )
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space, env.action_space
    )
    env.close()
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    curve_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    full_gate_records: list[dict[str, Any]] = []
    for variant, config in VARIANTS.items():
        for train_seed in args.train_seeds:
            model, predictions, curves, training_record = _train_model(
                dataset=dataset,
                layout=layout,
                variant=variant,
                config=config,
                train_seed=train_seed,
                args=args,
            )
            curve_rows.extend(curves)
            training_records.append(training_record)
            rows, values = _metric_rows(
                dataset=dataset,
                predictions=predictions,
                variant=variant,
                train_seed=train_seed,
                layout=layout,
            )
            metric_rows.extend(rows)
            for index, prediction in enumerate(predictions):
                prediction_rows.append(
                    {
                        "variant": variant,
                        "train_seed": train_seed,
                        "state_id": str(dataset["state_ids"][index]),
                        "split": str(dataset["splits"][index]),
                        "scenario": str(dataset["scenarios"][index]),
                        "source_seed": int(dataset["source_seeds"][index]),
                        "unit_index": int(dataset["unit_indices"][index]),
                        "candidate_action": int(dataset["candidate_actions"][index]),
                        "mc_q": float(dataset["q_labels"][index]),
                        "q_prediction": float(prediction),
                        "frozen_v": float(dataset["frozen_values"][index]),
                    }
                )
            payload = {
                "state_dict": model.state_dict(),
                "signature": model.signature(),
                "q_mean": training_record["q_mean"],
                "q_std": training_record["q_std"],
                "variant": variant,
                "train_seed": train_seed,
            }
            torch.save(payload, model_dir / f"{variant}_seed{train_seed}.pt")
            if variant == "full":
                gate = _gate_record(
                    values,
                    inference_seconds=training_record["inference_seconds"],
                    dataset=dataset,
                )
                full_gate_records.append(
                    {"train_seed": train_seed, "metrics": values, **gate}
                )
            print(
                f"train variant={variant} seed={train_seed} "
                f"mae={values['q_mae']:.3f} "
                f"rank={values['ranking_accuracy']:.3f}",
                flush=True,
            )

    split_leakage = any(
        len(set(dataset["splits"][dataset["state_ids"] == state_id])) != 1
        for state_id in np.unique(dataset["state_ids"])
    )
    passed_seeds = sum(record["passed"] for record in full_gate_records)
    final_report = {
        "schema_version": 1,
        "dataset": {
            "rows": int(len(dataset["q_labels"])),
            "states": int(dataset["state_count"]),
            "split_counts": {
                split: int(np.sum(dataset["splits"] == split))
                for split in ("train", "validation", "test")
            },
            "state_split_leakage": split_leakage,
            "generation_seconds": float(dataset["generation_seconds"]),
        },
        "full_model_seeds": full_gate_records,
        "passed_seed_count": passed_seeds,
        "required_passed_seed_count": 2,
        "task14_passed": (not split_leakage and passed_seeds >= 2),
        "resume_mch_ppo": (not split_leakage and passed_seeds >= 2),
        "enter_gnn": False,
    }
    _write_csv(args.output_dir / "training_curves.csv", curve_rows)
    _write_csv(args.output_dir / "metrics.csv", metric_rows)
    _write_csv(args.output_dir / "predictions.csv", prediction_rows)
    _write_csv(args.output_dir / "training_records.csv", training_records)
    (args.output_dir / "gate_summary.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config_record = {
        "schema_version": 1,
        "source_model_dir": str(args.model_dir.resolve()),
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
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "variants": {
            name: config.signature() for name, config in VARIANTS.items()
        },
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final_report, ensure_ascii=False, indent=2))
    print(f"output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
