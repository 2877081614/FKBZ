from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.common import (
    engagement_sign_accuracy,
    pairwise_ranking_accuracy,
    top_action_accuracy,
)

DEFAULT_DATASET = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task14_q_critic" / "dataset.npz"
)
SCENARIOS = ("medium", "time_pressure", "heterogeneity_pressure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit paired-comparison power in a Task 14 Q-Critic dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--projected-rollouts", nargs="+", type=int, default=(8, 16, 32, 64)
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def projected_pair_counts(
    dataset: dict[str, np.ndarray], projected_rollouts: tuple[int, ...]
) -> list[dict[str, object]]:
    test = np.flatnonzero(dataset["splits"] == "test")
    samples = dataset["return_samples"][test]
    observed_rollouts = samples.shape[1]
    group_ids = np.asarray(
        [
            f"{state_id}/unit{unit_index}"
            for state_id, unit_index in zip(
                dataset["state_ids"][test], dataset["unit_indices"][test]
            )
        ]
    )
    scenarios = dataset["scenarios"][test]
    rows: list[dict[str, object]] = []
    for scenario in ("all", *SCENARIOS):
        selected = (
            np.ones(len(test), dtype=bool)
            if scenario == "all"
            else scenarios == scenario
        )
        for rollout_count in projected_rollouts:
            # Projection holds the observed effect and variance fixed and only
            # changes the standard-error denominator.
            critical_t = 1.96 * np.sqrt(observed_rollouts / rollout_count)
            count = 0
            total = 0
            for group_id in np.unique(group_ids[selected]):
                indices = np.flatnonzero(selected & (group_ids == group_id))
                for left, right in combinations(indices.tolist(), 2):
                    paired = samples[left] - samples[right]
                    standard_error = np.std(paired, ddof=1) / np.sqrt(
                        observed_rollouts
                    )
                    if standard_error == 0.0:
                        effect_t = np.inf if np.mean(paired) != 0.0 else 0.0
                    else:
                        effect_t = abs(float(np.mean(paired))) / standard_error
                    count += int(effect_t > critical_t)
                    total += 1
            rows.append(
                {
                    "scenario": scenario,
                    "projected_rollouts": rollout_count,
                    "high_confidence_pairs": count,
                    "total_pairs": total,
                }
            )
    return rows


def projected_gate_counts(
    dataset: dict[str, np.ndarray], projected_rollouts: tuple[int, ...]
) -> list[dict[str, object]]:
    test = np.flatnonzero(dataset["splits"] == "test")
    samples = dataset["return_samples"][test]
    observed_rollouts = samples.shape[1]
    labels = dataset["q_labels"][test]
    actions = dataset["candidate_actions"][test]
    group_ids = np.asarray(
        [
            f"{state_id}/unit{unit_index}"
            for state_id, unit_index in zip(
                dataset["state_ids"][test], dataset["unit_indices"][test]
            )
        ]
    )
    probabilities = dataset["conditional_target_probabilities"][test]
    scenarios = dataset["scenarios"][test]
    noop_action = dataset["legal_action_masks"].shape[1] - 1
    rows: list[dict[str, object]] = []
    for rollout_count in projected_rollouts:
        projected_z = 1.96 * np.sqrt(observed_rollouts / rollout_count)
        evaluations = {
            "overall_pair": pairwise_ranking_accuracy(
                labels,
                labels,
                group_ids,
                return_samples=samples,
                uncertainty_z=projected_z,
            ),
            "target_pair": pairwise_ranking_accuracy(
                labels,
                labels,
                group_ids,
                return_samples=samples,
                candidate_actions=actions,
                noop_action=noop_action,
                target_only=True,
                uncertainty_z=projected_z,
            ),
            "top_action": top_action_accuracy(
                labels,
                labels,
                group_ids,
                return_samples=samples,
                uncertainty_z=projected_z,
            ),
            "engagement_sign": engagement_sign_accuracy(
                labels,
                labels,
                group_ids,
                actions,
                probabilities,
                noop_action=noop_action,
                return_samples=samples,
                uncertainty_z=projected_z,
            ),
        }
        target_only = actions != noop_action
        evaluations["target_top"] = top_action_accuracy(
            labels[target_only],
            labels[target_only],
            group_ids[target_only],
            return_samples=samples[target_only],
            uncertainty_z=projected_z,
        )
        for metric, result in evaluations.items():
            rows.append(
                {
                    "scope": "all",
                    "metric": metric,
                    "projected_rollouts": rollout_count,
                    "high_confidence_count": int(result["count"]),
                }
            )
        for scenario in SCENARIOS:
            selected = scenarios == scenario
            scenario_target = selected & target_only
            scenario_evaluations = {
                "pair": pairwise_ranking_accuracy(
                    labels[selected],
                    labels[selected],
                    group_ids[selected],
                    return_samples=samples[selected],
                    uncertainty_z=projected_z,
                ),
                "engagement_sign": engagement_sign_accuracy(
                    labels[selected],
                    labels[selected],
                    group_ids[selected],
                    actions[selected],
                    probabilities[selected],
                    noop_action=noop_action,
                    return_samples=samples[selected],
                    uncertainty_z=projected_z,
                ),
                "target_pair": pairwise_ranking_accuracy(
                    labels[scenario_target],
                    labels[scenario_target],
                    group_ids[scenario_target],
                    return_samples=samples[scenario_target],
                    uncertainty_z=projected_z,
                ),
                "target_top": top_action_accuracy(
                    labels[scenario_target],
                    labels[scenario_target],
                    group_ids[scenario_target],
                    return_samples=samples[scenario_target],
                    uncertainty_z=projected_z,
                ),
            }
            for metric, result in scenario_evaluations.items():
                rows.append(
                    {
                        "scope": scenario,
                        "metric": metric,
                        "projected_rollouts": rollout_count,
                        "high_confidence_count": int(result["count"]),
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    if min(args.projected_rollouts) < 2:
        raise ValueError("projected rollouts must be at least two")
    with np.load(args.dataset, allow_pickle=False) as archive:
        dataset = {key: archive[key] for key in archive.files}
    rollout_counts = tuple(args.projected_rollouts)
    pair_rows = projected_pair_counts(dataset, rollout_counts)
    print("scenario,projected_rollouts,high_confidence_pairs,total_pairs")
    for row in pair_rows:
        print(
            f"{row['scenario']},{row['projected_rollouts']},"
            f"{row['high_confidence_pairs']},{row['total_pairs']}"
        )
    gate_rows = projected_gate_counts(dataset, rollout_counts)
    print("scope,metric,projected_rollouts,high_confidence_count")
    for row in gate_rows:
        print(
            f"{row['scope']},{row['metric']},{row['projected_rollouts']},"
            f"{row['high_confidence_count']}"
        )
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, output_rows in (
            ("pair_power_projection.csv", pair_rows),
            ("gate_power_projection.csv", gate_rows),
        ):
            with (args.output_dir / filename).open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
                writer.writeheader()
                writer.writerows(output_rows)


if __name__ == "__main__":
    main()
