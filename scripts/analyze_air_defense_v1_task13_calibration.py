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
from rein_learning.common import (
    binary_calibration_metrics,
    engagement_threshold_grid,
)
from rein_learning.envs import get_air_defense_v1_scenario
from rein_learning.trainers.air_defense_v1_ppo import evaluate_air_defense_v1_model


DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task12_factorized_screening_30k_3seeds"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task13_calibration"
)
DEFAULT_TASK7_REFERENCE = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task7_formal_medium_100k_5seeds"
)
METHOD_CLASSES = {
    "role_conditioned_ar_ppo_order_012": RoleConditionedAutoregressiveMaskablePPO,
    "factorized_engagement_ar_ppo_order_012": FactorizedEngagementMaskablePPO,
}
SCENARIOS = ("medium", "time_pressure", "heterogeneity_pressure")


class DiagnosticPolicyAdapter:
    """Expose threshold or stochastic actions while retaining SB3 policy metadata."""

    def __init__(self, model: Any, *, threshold: float | None) -> None:
        self.model = model
        self.policy = model.policy
        self.threshold = threshold
        self.records: list[dict[str, float | int]] = []

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
        action_masks: np.ndarray,
    ) -> tuple[np.ndarray, None]:
        del deterministic
        observation_tensor, vectorized = self.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = self.policy.get_distribution(
                observation_tensor, action_masks=action_masks
            )
            if self.threshold is None:
                evaluation = distribution.sample(deterministic=False)
            else:
                evaluation = distribution.sample_with_engagement_threshold(
                    self.threshold
                )
            if hasattr(distribution, "hierarchical_diagnostics"):
                diagnostics = distribution.hierarchical_diagnostics(
                    evaluation.actions
                )
            else:
                diagnostics = distribution.diagnostics(
                    actions=evaluation.actions
                )
        for batch_index in range(evaluation.actions.shape[0]):
            for unit_index in range(evaluation.actions.shape[1]):
                self.records.append(
                    {
                        "unit_index": unit_index,
                        "engage_probability": float(
                            diagnostics["engage_probability"][
                                batch_index, unit_index
                            ].cpu()
                        ),
                        "selected_engage": int(
                            diagnostics["selected_engage"][
                                batch_index, unit_index
                            ].cpu()
                        ),
                        "actionable": int(
                            diagnostics["actionable"][
                                batch_index, unit_index
                            ].cpu()
                        ),
                    }
                )
        actions = evaluation.actions.cpu().numpy()
        return (actions if vectorized else actions[0]), None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 13 frozen-model engagement calibration and threshold scan."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--task7-reference", type=Path, default=DEFAULT_TASK7_REFERENCE
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=13_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--methods", nargs="+", choices=tuple(METHOD_CLASSES), default=None
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--calibration-episodes", type=int, default=30)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _discover_seeds(model_dir: Path) -> list[int]:
    seeds: set[int] = set()
    for path in (model_dir / "models" / "medium").glob("*_seed*.zip"):
        seeds.add(int(path.stem.rsplit("_seed", maxsplit=1)[1]))
    if not seeds:
        raise FileNotFoundError(f"No frozen models found under {model_dir}")
    return sorted(seeds)


def _load_model(
    model_dir: Path, method: str, seed: int, device: str
) -> Any:
    path = model_dir / "models" / "medium" / f"{method}_seed{seed}.zip"
    if not path.exists():
        raise FileNotFoundError(path)
    return METHOD_CLASSES[method].load(path, device=device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _paired_evaluation_seed(base: int, scenario_index: int) -> int:
    return base + 10_000 * scenario_index


def _threshold_scan(
    *,
    model_dir: Path,
    methods: list[str],
    seeds: list[int],
    episodes: int,
    eval_seed: int,
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        for train_seed in seeds:
            model = _load_model(model_dir, method, train_seed, device)
            for threshold in engagement_threshold_grid():
                adapter = DiagnosticPolicyAdapter(model, threshold=float(threshold))
                for scenario_index, scenario in enumerate(SCENARIOS):
                    sampling_seed = (
                        130_000
                        + train_seed * 1_000
                        + scenario_index * 100
                        + int(round(threshold * 100))
                    )
                    _seed_everything(sampling_seed)
                    metrics = evaluate_air_defense_v1_model(
                        adapter,
                        env_config=get_air_defense_v1_scenario(scenario),
                        episodes=episodes,
                        seed=_paired_evaluation_seed(eval_seed, scenario_index),
                        deterministic=True,
                        use_action_masks=True,
                    )
                    row = {
                        "method": method,
                        "train_seed": train_seed,
                        "eval_scenario": scenario,
                        "engagement_threshold": float(threshold),
                        "episodes": episodes,
                        "evaluation_seed": _paired_evaluation_seed(
                            eval_seed, scenario_index
                        ),
                        **metrics,
                    }
                    rows.append(row)
                    print(
                        f"scan method={method} seed={train_seed} "
                        f"scenario={scenario} threshold={threshold:.2f} "
                        f"reward={metrics['avg_reward']:.2f} "
                        f"engage={metrics['actionable_engagement_rate']:.3f}",
                        flush=True,
                    )
    return rows


def _sampling_calibration(
    *,
    model_dir: Path,
    methods: list[str],
    seeds: list[int],
    episodes: int,
    eval_seed: int,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for method in methods:
        for train_seed in seeds:
            model = _load_model(model_dir, method, train_seed, device)
            for scenario_index, scenario in enumerate(SCENARIOS):
                adapter = DiagnosticPolicyAdapter(model, threshold=None)
                sampling_seed = 230_000 + train_seed * 100 + scenario_index
                _seed_everything(sampling_seed)
                evaluate_air_defense_v1_model(
                    adapter,
                    env_config=get_air_defense_v1_scenario(scenario),
                    episodes=episodes,
                    seed=_paired_evaluation_seed(eval_seed + 50_000, scenario_index),
                    deterministic=False,
                    use_action_masks=True,
                )
                actionable = [row for row in adapter.records if row["actionable"]]
                calibration = binary_calibration_metrics(
                    [row["engage_probability"] for row in actionable],
                    [row["selected_engage"] for row in actionable],
                )
                common = {
                    "method": method,
                    "train_seed": train_seed,
                    "eval_scenario": scenario,
                    "episodes": episodes,
                    "calibration_target": "sampled_engagement_action",
                }
                summary_rows.append(
                    {
                        **common,
                        **{key: value for key, value in calibration.items() if key != "bins"},
                    }
                )
                for row in calibration["bins"]:
                    bin_rows.append({**common, **row})
    return summary_rows, bin_rows


def _uniform_threshold_assessment(
    rows: list[dict[str, Any]], task7_reference: Path
) -> dict[str, Any]:
    reference_rows = [
        row
        for row in _read_csv(task7_reference / "runs.csv")
        if row["method"] == "maskable_ppo"
    ]
    references: dict[str, dict[str, float]] = {}
    for scenario in SCENARIOS:
        selected = [
            row for row in reference_rows if row["eval_scenario"] == scenario
        ]
        if not selected:
            raise ValueError(f"Task 7 reference is missing scenario {scenario}")
        references[scenario] = {
            metric: float(np.mean([float(row[metric]) for row in selected]))
            for metric in (
                "avg_reward",
                "avg_total_damage",
                "high_threat_leak_rate",
                "avg_resource_cost",
            )
        }
    threshold_results: list[dict[str, Any]] = []
    for threshold in engagement_threshold_grid():
        selected = [
            row
            for row in rows
            if abs(float(row["engagement_threshold"]) - threshold) < 1e-9
        ]
        per_run_passes: list[bool] = []
        for row in selected:
            reference = references[str(row["eval_scenario"])]
            per_run_passes.append(
                float(row["all_noop_episode_rate"]) <= 0.02
                and float(row["avg_reward"])
                >= reference["avg_reward"] - 5.0
                and float(row["avg_total_damage"])
                <= reference["avg_total_damage"] + 0.10
                and float(row["high_threat_leak_rate"])
                <= reference["high_threat_leak_rate"] + 0.02
                and float(row["avg_resource_cost"])
                <= reference["avg_resource_cost"] + 0.50
            )
        threshold_results.append(
            {
                "threshold": float(threshold),
                "passed_runs": int(sum(per_run_passes)),
                "total_runs": len(per_run_passes),
                "uniformly_valid": bool(per_run_passes and all(per_run_passes)),
            }
        )
    valid = [row["threshold"] for row in threshold_results if row["uniformly_valid"]]
    return {
        "criteria": {
            "all_noop_episode_rate": "<= 0.02",
            "reference": "Task 7 frozen 100k Maskable PPO scenario mean",
            "reward_delta_vs_reference": ">= -5.0",
            "damage_delta_vs_reference": "<= 0.10",
            "high_threat_leak_delta_vs_reference": "<= 0.02",
            "resource_cost_delta_vs_reference": "<= 0.50",
        },
        "reference_means": references,
        "thresholds": threshold_results,
        "uniform_valid_thresholds": valid,
        "uniform_threshold_exists": bool(valid),
    }


def main() -> None:
    args = parse_args()
    methods = list(args.methods or METHOD_CLASSES)
    seeds = list(args.seeds or _discover_seeds(args.model_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_existing:
        threshold_rows = _read_csv(args.output_dir / "threshold_runs.csv")
        calibration_rows = _read_csv(args.output_dir / "action_calibration.csv")
        bin_rows = _read_csv(args.output_dir / "reliability_bins.csv")
    else:
        threshold_rows = _threshold_scan(
            model_dir=args.model_dir,
            methods=methods,
            seeds=seeds,
            episodes=args.episodes,
            eval_seed=args.eval_seed,
            device=args.device,
        )
        calibration_rows, bin_rows = _sampling_calibration(
            model_dir=args.model_dir,
            methods=methods,
            seeds=seeds,
            episodes=args.calibration_episodes,
            eval_seed=args.eval_seed,
            device=args.device,
        )
    assessment = _uniform_threshold_assessment(
        threshold_rows, args.task7_reference
    )

    _write_csv(args.output_dir / "threshold_runs.csv", threshold_rows)
    _write_csv(args.output_dir / "action_calibration.csv", calibration_rows)
    _write_csv(args.output_dir / "reliability_bins.csv", bin_rows)
    (args.output_dir / "threshold_assessment.json").write_text(
        json.dumps(assessment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = {
        "schema_version": 1,
        "source_models": str(args.model_dir.resolve()),
        "task7_reference": str(args.task7_reference.resolve()),
        "methods": methods,
        "train_seeds": seeds,
        "scenarios": list(SCENARIOS),
        "thresholds": engagement_threshold_grid().tolist(),
        "episodes": int(float(threshold_rows[0]["episodes"])),
        "calibration_episodes": int(float(calibration_rows[0]["episodes"])),
        "eval_seed": args.eval_seed,
        "calibration_scope": (
            "Brier/ECE tests policy probability against stochastic action realization; "
            "mission utility is assessed by paired threshold-return curves."
        ),
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(assessment, ensure_ascii=False, indent=2))
    print(f"output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
