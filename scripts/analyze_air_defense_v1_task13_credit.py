from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.algorithms.policy_gradient import (
    FactorizedEngagementMaskablePPO,
    RoleConditionedAutoregressiveMaskablePPO,
)
from rein_learning.common import hierarchical_counterfactual_advantages
from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    get_air_defense_v1_scenario,
)


DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task12_factorized_screening_30k_3seeds"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task13_credit_diagnostics"
)
METHOD_CLASSES = {
    "role_conditioned_ar_ppo_order_012": RoleConditionedAutoregressiveMaskablePPO,
    "factorized_engagement_ar_ppo_order_012": FactorizedEngagementMaskablePPO,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 13 hierarchical credit and counterfactual diagnostics."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHOD_CLASSES), default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=("medium", "time_pressure", "heterogeneity_pressure"),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=31_000)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--gradient-samples", type=int, default=100)
    parser.add_argument("--counterfactual-states", type=int, default=12)
    parser.add_argument("--counterfactual-rollouts", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _discover_seeds(model_dir: Path) -> list[int]:
    seeds = {
        int(path.stem.rsplit("_seed", maxsplit=1)[1])
        for path in (model_dir / "models" / "medium").glob("*_seed*.zip")
    }
    if not seeds:
        raise FileNotFoundError(f"No models found under {model_dir}")
    return sorted(seeds)


def _load_model(model_dir: Path, method: str, seed: int, device: str) -> Any:
    model_path = model_dir / "models" / "medium" / f"{method}_seed{seed}.zip"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return METHOD_CLASSES[method].load(model_path, device=device)


def _distribution_and_value(
    model: Any, observation: np.ndarray, action_mask: np.ndarray
) -> tuple[Any, torch.Tensor, float]:
    observation_tensor, _ = model.policy.obs_to_tensor(observation)
    distribution = model.policy.get_distribution(
        observation_tensor, action_masks=action_mask
    )
    value = float(model.policy.predict_values(observation_tensor).detach().cpu().item())
    return distribution, observation_tensor, value


def _diagnostics(distribution: Any, actions: torch.Tensor) -> dict[str, torch.Tensor]:
    if hasattr(distribution, "hierarchical_diagnostics"):
        return distribution.hierarchical_diagnostics(actions)
    return distribution.diagnostics(actions=actions)


def _branch_gradient_statistics(
    model: Any,
    diagnostics: dict[str, torch.Tensor],
    unit_index: int,
    advantage: float,
) -> tuple[float, float, float]:
    parameters = [
        parameter
        for parameter in model.policy.action_net.parameters()
        if parameter.requires_grad
    ]

    def gradients(value: torch.Tensor, retain_graph: bool) -> list[torch.Tensor | None]:
        return list(
            torch.autograd.grad(
                value * advantage,
                parameters,
                retain_graph=retain_graph,
                allow_unused=True,
            )
        )

    engagement = diagnostics["engagement_log_prob"][0, unit_index]
    engage_gradients = gradients(engagement, retain_graph=True)
    if not bool(diagnostics["selected_engage"][0, unit_index]):
        engage_norm = _gradient_norm(engage_gradients)
        return engage_norm, 0.0, float("nan")
    target = diagnostics["target_log_prob"][0, unit_index]
    # All units at the same environment step share one actor graph.
    target_gradients = gradients(target, retain_graph=True)
    engage_norm = _gradient_norm(engage_gradients)
    target_norm = _gradient_norm(target_gradients)
    dot = sum(
        float(torch.sum(left * right).detach().cpu())
        for left, right in zip(engage_gradients, target_gradients)
        if left is not None and right is not None
    )
    cosine = dot / (engage_norm * target_norm) if engage_norm and target_norm else float("nan")
    return engage_norm, target_norm, cosine


def _gradient_norm(gradients: list[torch.Tensor | None]) -> float:
    squared = sum(
        float(torch.sum(gradient.detach() ** 2).cpu())
        for gradient in gradients
        if gradient is not None
    )
    return float(np.sqrt(squared))


def _collect_rollouts(
    *,
    model: Any,
    method: str,
    train_seed: int,
    scenario: str,
    episodes: int,
    eval_seed: int,
    gamma: float,
    gradient_budget: list[int],
    counterfactual_budget: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        env = AirDefenseResourceAssignmentEnvV1(get_air_defense_v1_scenario(scenario))
        observation, _ = env.reset(seed=eval_seed + episode_index)
        trajectory: list[dict[str, Any]] = []
        terminated = False
        truncated = False
        step_index = 0
        while not (terminated or truncated):
            action_mask = env.action_masks()
            distribution, _, value = _distribution_and_value(
                model, observation, action_mask
            )
            evaluation = distribution.sample(deterministic=False)
            diagnostics = _diagnostics(distribution, evaluation.actions)
            probabilities, conditional_masks = distribution.conditional_probabilities(
                evaluation.actions
            )
            action = evaluation.actions[0].detach().cpu().numpy()
            if counterfactual_budget[0] > 0:
                snapshots.append(
                    {
                        "state_id": (
                            f"{method}/seed{train_seed}/{scenario}/"
                            f"episode{episode_index}/step{step_index}"
                        ),
                        "eval_scenario": scenario,
                        "env": deepcopy(env),
                        "action": action.copy(),
                        "probabilities": probabilities[0].detach().cpu().numpy(),
                        "conditional_masks": conditional_masks[0].detach().cpu().numpy(),
                        "value": value,
                    }
                )
                counterfactual_budget[0] -= 1
            next_observation, reward, terminated, truncated, _ = env.step(action)
            terminal = terminated or truncated
            if terminal:
                next_value = 0.0
            else:
                _, _, next_value = _distribution_and_value(
                    model, next_observation, env.action_masks()
                )
            trajectory.append(
                {
                    "episode_index": episode_index,
                    "step_index": step_index,
                    "action": action.copy(),
                    "reward": float(reward),
                    "value": value,
                    "next_value": next_value,
                    "terminal": terminal,
                    "diagnostics": diagnostics,
                }
            )
            observation = next_observation
            step_index += 1

        return_to_go = 0.0
        for transition in reversed(trajectory):
            return_to_go = transition["reward"] + gamma * return_to_go
            transition["return"] = return_to_go
        for transition in trajectory:
            advantage = transition["return"] - transition["value"]
            td_error = (
                transition["reward"]
                + gamma * transition["next_value"]
                - transition["value"]
            )
            diagnostics = transition["diagnostics"]
            for unit_index, action in enumerate(transition["action"]):
                engage_norm = target_norm = cosine = float("nan")
                if gradient_budget[0] > 0:
                    engage_norm, target_norm, cosine = _branch_gradient_statistics(
                        model, diagnostics, unit_index, advantage
                    )
                    gradient_budget[0] -= 1
                target_probability = float(
                    diagnostics["target_probability"][0, unit_index].detach().cpu()
                )
                rows.append(
                    {
                        "method": method,
                        "train_seed": train_seed,
                        "eval_scenario": scenario,
                        "episode_index": transition["episode_index"],
                        "step_index": transition["step_index"],
                        "unit_index": unit_index,
                        "resource_type": env.defense_units[unit_index].resource_type,
                        "selected_action": int(action),
                        "selected_target": (
                            int(action) if int(action) != env.num_targets else None
                        ),
                        "selected_engage": int(
                            diagnostics["selected_engage"][0, unit_index].detach().cpu()
                        ),
                        "engage_probability": float(
                            diagnostics["engage_probability"][0, unit_index].detach().cpu()
                        ),
                        "target_probability": target_probability,
                        "return": transition["return"],
                        "reward": transition["reward"],
                        "value": transition["value"],
                        "value_bias": transition["value"] - transition["return"],
                        "joint_advantage": advantage,
                        "td_error": td_error,
                        "engagement_log_prob": float(
                            diagnostics["engagement_log_prob"][0, unit_index].detach().cpu()
                        ),
                        "target_log_prob": float(
                            diagnostics["target_log_prob"][0, unit_index].detach().cpu()
                        ),
                        "engagement_gradient_norm": engage_norm,
                        "target_gradient_norm": target_norm,
                        "branch_gradient_cosine": cosine,
                    }
                )
        env.close()
    return rows, snapshots


def _sample_policy_action(model: Any, env: AirDefenseResourceAssignmentEnvV1) -> np.ndarray:
    distribution, _, _ = _distribution_and_value(
        model, env._get_observation(), env.action_masks()
    )
    return distribution.sample(deterministic=False).actions[0].detach().cpu().numpy()


def _branch_return(
    *,
    model: Any,
    snapshot: AirDefenseResourceAssignmentEnvV1,
    first_action: np.ndarray,
    gamma: float,
    random_seed: int,
) -> tuple[float, float]:
    env = deepcopy(snapshot)
    env.np_random = np.random.default_rng(random_seed)
    torch.manual_seed(random_seed)
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


def _counterfactual_diagnostics(
    *,
    model: Any,
    snapshots: list[dict[str, Any]],
    gamma: float,
    rollouts: int,
    base_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    branch_rows: list[dict[str, Any]] = []
    advantage_rows: list[dict[str, Any]] = []
    for state_index, snapshot in enumerate(snapshots):
        env = snapshot["env"]
        observed_action = snapshot["action"]
        num_targets = env.num_targets
        for unit_index in range(env.num_defense_units):
            assigned_elsewhere = {
                int(action)
                for index, action in enumerate(observed_action)
                if index != unit_index and int(action) != num_targets
            }
            legal_mask = snapshot["conditional_masks"][unit_index, :num_targets].copy()
            for target in assigned_elsewhere:
                legal_mask[target] = False
            candidates = [num_targets] + np.flatnonzero(legal_mask).tolist()
            q_values: dict[int, float] = {}
            return_samples: dict[int, np.ndarray] = {}
            pending_branch_rows: list[dict[str, Any]] = []
            for candidate in candidates:
                action = observed_action.copy()
                action[unit_index] = candidate
                returns: list[float] = []
                immediate_rewards: list[float] = []
                for rollout_index in range(rollouts):
                    random_seed = base_seed + state_index * 10_000 + rollout_index
                    total_return, immediate = _branch_return(
                        model=model,
                        snapshot=env,
                        first_action=action,
                        gamma=gamma,
                        random_seed=random_seed,
                    )
                    returns.append(total_return)
                    immediate_rewards.append(immediate)
                q_values[candidate] = float(np.mean(returns))
                return_samples[candidate] = np.asarray(returns, dtype=np.float64)
                pending_branch_rows.append(
                    {
                        "state_id": snapshot["state_id"],
                        "eval_scenario": snapshot["eval_scenario"],
                        "unit_index": unit_index,
                        "candidate_action": candidate,
                        "candidate_type": "noop" if candidate == num_targets else "target",
                        "observed_action": int(observed_action[unit_index]),
                        "legal_candidate_count": len(candidates),
                        "mc_q_mean": q_values[candidate],
                        "mc_q_std": float(np.std(returns)),
                        "one_step_reward_mean": float(np.mean(immediate_rewards)),
                        "critic_v": snapshot["value"],
                        "critic_error_vs_branch": snapshot["value"] - q_values[candidate],
                        "rollouts": rollouts,
                    }
                )
            noop_samples = return_samples[num_targets]
            for row in pending_branch_rows:
                candidate = int(row["candidate_action"])
                paired_differences = return_samples[candidate] - noop_samples
                row.update(
                    {
                        "paired_delta_vs_noop_mean": float(
                            np.mean(paired_differences)
                        ),
                        "paired_delta_vs_noop_std": float(
                            np.std(paired_differences)
                        ),
                        "paired_delta_vs_noop_se": float(
                            np.std(paired_differences) / np.sqrt(rollouts)
                        ),
                    }
                )
                branch_rows.append(row)
            target_q = np.zeros((1, 1, num_targets), dtype=np.float64)
            for target, value in q_values.items():
                if target != num_targets:
                    target_q[0, 0, target] = value
            action_probabilities = snapshot["probabilities"][unit_index]
            engage_probability = float(action_probabilities[:num_targets].sum())
            target_probabilities = action_probabilities[:num_targets] / max(
                engage_probability, 1e-20
            )
            decomposition = hierarchical_counterfactual_advantages(
                q_noop=np.array([[q_values[num_targets]]]),
                q_targets=target_q,
                engage_probabilities=np.array([[engage_probability]]),
                target_probabilities=target_probabilities.reshape(1, 1, -1),
                legal_target_mask=legal_mask.reshape(1, 1, -1),
                selected_actions=np.array([[observed_action[unit_index]]]),
            )
            advantage_rows.append(
                {
                    "state_id": snapshot["state_id"],
                    "eval_scenario": snapshot["eval_scenario"],
                    "unit_index": unit_index,
                    "observed_action": int(observed_action[unit_index]),
                    "engage_probability": engage_probability,
                    "q_noop": q_values[num_targets],
                    "q_engage": float(decomposition["q_engage"].item()),
                    "counterfactual_baseline": float(
                        decomposition["counterfactual_baseline"].item()
                    ),
                    "engagement_advantage": float(
                        decomposition["selected_engagement_advantage"].item()
                    ),
                    "target_advantage": float(
                        decomposition["selected_target_advantage"].item()
                    ),
                    "total_counterfactual_advantage": float(
                        decomposition["selected_total_advantage"].item()
                    ),
                    "joint_value_advantage_proxy": float(
                        q_values[int(observed_action[unit_index])]
                        - snapshot["value"]
                    ),
                }
            )
    return branch_rows, advantage_rows


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["method"],
            row["train_seed"],
            row["eval_scenario"],
            row["selected_engage"],
        )
        grouped.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    metrics = ("joint_advantage", "td_error", "value_bias", "engage_probability")
    for key, selected in grouped.items():
        record = {
            "method": key[0],
            "train_seed": key[1],
            "eval_scenario": key[2],
            "selected_engage": key[3],
            "count": len(selected),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = float(np.std(values))
            record[f"{metric}_positive_rate"] = float(np.mean(values > 0.0))
        summary.append(record)
    return summary


def main() -> None:
    args = parse_args()
    methods = list(args.methods or METHOD_CLASSES)
    seeds = list(args.seeds or _discover_seeds(args.model_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)

    rollout_rows: list[dict[str, Any]] = []
    all_branch_rows: list[dict[str, Any]] = []
    all_advantage_rows: list[dict[str, Any]] = []
    for method in methods:
        for train_seed in seeds:
            model = _load_model(args.model_dir, method, train_seed, args.device)
            gradient_budget = [args.gradient_samples]
            model_snapshots: list[dict[str, Any]] = []
            for scenario_index, scenario in enumerate(args.scenarios):
                counterfactual_budget = [args.counterfactual_states]
                rows, snapshots = _collect_rollouts(
                    model=model,
                    method=method,
                    train_seed=train_seed,
                    scenario=scenario,
                    episodes=args.episodes,
                    eval_seed=args.eval_seed + scenario_index * 10_000,
                    gamma=args.gamma,
                    gradient_budget=gradient_budget,
                    counterfactual_budget=counterfactual_budget,
                )
                rollout_rows.extend(rows)
                model_snapshots.extend(snapshots)
                print(
                    f"credit method={method} seed={train_seed} scenario={scenario} "
                    f"rows={len(rows)} snapshots={len(snapshots)}",
                    flush=True,
                )
            branch_rows, advantage_rows = _counterfactual_diagnostics(
                model=model,
                snapshots=model_snapshots,
                gamma=args.gamma,
                rollouts=args.counterfactual_rollouts,
                base_seed=args.eval_seed + train_seed * 1_000_000,
            )
            for row in branch_rows:
                row.update({"method": method, "train_seed": train_seed})
            for row in advantage_rows:
                row.update({"method": method, "train_seed": train_seed})
            all_branch_rows.extend(branch_rows)
            all_advantage_rows.extend(advantage_rows)

    _write_csv(args.output_dir / "credit_samples.csv", rollout_rows)
    _write_csv(args.output_dir / "credit_summary.csv", _summary(rollout_rows))
    _write_csv(args.output_dir / "counterfactual_branches.csv", all_branch_rows)
    _write_csv(
        args.output_dir / "counterfactual_advantages.csv", all_advantage_rows
    )
    config = {
        "schema_version": 1,
        "source_models": str(args.model_dir.resolve()),
        "methods": methods,
        "train_seeds": seeds,
        "scenarios": list(args.scenarios),
        "episodes": args.episodes,
        "eval_seed": args.eval_seed,
        "gamma": args.gamma,
        "gradient_samples_per_model": args.gradient_samples,
        "counterfactual_states_per_model_scenario": args.counterfactual_states,
        "counterfactual_rollouts": args.counterfactual_rollouts,
        "critic_scope": (
            "The frozen PPO critic estimates V(s), not Q(s,a). Branch error is "
            "reported diagnostically and is not interpreted as a learned Q error."
        ),
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
