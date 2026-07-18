from __future__ import annotations

import argparse
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
from rein_learning.envs import get_air_defense_v1_scenario
from rein_learning.trainers.air_defense_v1_ppo import evaluate_air_defense_v1_model


METHOD_CLASSES = {
    "role_conditioned_ar_ppo_order_012": (
        RoleConditionedAutoregressiveMaskablePPO
    ),
    "factorized_engagement_ar_ppo_order_012": FactorizedEngagementMaskablePPO,
}
DEFAULT_SCREENING = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task12_factorized_screening_30k_3seeds"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stochastic evaluation for frozen Task 12 screening models."
    )
    parser.add_argument("--screening-dir", type=Path, default=DEFAULT_SCREENING)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=200)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    deterministic_rows = _read_csv(args.screening_dir / "runs.csv")
    seeds = sorted({int(row["train_seed"]) for row in deterministic_rows})
    scenarios = list(
        json.loads(
            (args.screening_dir / "experiment_config.json").read_text(
                encoding="utf-8"
            )
        )["benchmark"]["eval_scenarios"]
    )
    stochastic_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for run_index, train_seed in enumerate(seeds):
        for method, model_class in METHOD_CLASSES.items():
            model_path = (
                args.screening_dir
                / "models"
                / "medium"
                / f"{method}_seed{train_seed}.zip"
            )
            model = model_class.load(model_path, device="cpu")
            for scenario_index, scenario in enumerate(scenarios):
                evaluation_seed = (
                    args.eval_seed
                    + run_index * len(scenarios) * args.episodes
                    + scenario_index * args.episodes
                )
                sampling_seed = 70_000 + run_index * 100 + scenario_index
                random.seed(sampling_seed)
                np.random.seed(sampling_seed)
                torch.manual_seed(sampling_seed)
                metrics = evaluate_air_defense_v1_model(
                    model,
                    env_config=get_air_defense_v1_scenario(scenario),
                    episodes=args.episodes,
                    seed=evaluation_seed,
                    deterministic=False,
                    use_action_masks=True,
                )
                row = {
                    "method": method,
                    "train_scenario": "medium",
                    "eval_scenario": scenario,
                    "run_index": run_index,
                    "train_seed": train_seed,
                    "evaluation_seed": evaluation_seed,
                    "sampling_seed": sampling_seed,
                    **metrics,
                }
                stochastic_rows.append(row)
                deterministic = next(
                    item
                    for item in deterministic_rows
                    if item["method"] == method
                    and item["eval_scenario"] == scenario
                    and int(item["train_seed"]) == train_seed
                )
                gap_rows.append(
                    {
                        "method": method,
                        "eval_scenario": scenario,
                        "train_seed": train_seed,
                        "deterministic_actionable_engagement_rate": float(
                            deterministic["actionable_engagement_rate"]
                        ),
                        "stochastic_actionable_engagement_rate": metrics[
                            "actionable_engagement_rate"
                        ],
                        "stochastic_engagement_gap": metrics[
                            "actionable_engagement_rate"
                        ]
                        - float(deterministic["actionable_engagement_rate"]),
                        "deterministic_all_noop_episode_rate": float(
                            deterministic["all_noop_episode_rate"]
                        ),
                        "stochastic_all_noop_episode_rate": metrics[
                            "all_noop_episode_rate"
                        ],
                    }
                )
                print(
                    f"method={method} seed={train_seed} scenario={scenario} "
                    f"gap={gap_rows[-1]['stochastic_engagement_gap']:.4f}",
                    flush=True,
                )
    _write_csv(args.screening_dir / "stochastic_runs.csv", stochastic_rows)
    _write_csv(args.screening_dir / "stochastic_engagement_gaps.csv", gap_rows)


if __name__ == "__main__":
    main()
