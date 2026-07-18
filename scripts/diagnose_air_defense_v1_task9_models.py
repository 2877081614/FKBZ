from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.algorithms.policy_gradient.autoregressive_ppo import (
    AutoregressiveMaskablePPO,
)
from rein_learning.common import (
    aggregate_decision_rows,
    aggregate_leak_attributions,
)
from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    get_air_defense_v1_scenario_profile,
)
from rein_learning.experiments.air_defense_v1_benchmark import (
    DECISION_FIELDNAMES,
    DECISION_SUMMARY_FIELDNAMES,
    EPISODE_FIELDNAMES,
    LEAK_ATTRIBUTION_FIELDNAMES,
    LEAK_ATTRIBUTION_SUMMARY_FIELDNAMES,
    METRIC_NAMES,
    RUN_FIELDNAMES,
    SUMMARY_FIELDNAMES,
    summarize_rows,
)
from rein_learning.trainers.air_defense_v1_ppo import (
    evaluate_air_defense_v1_model,
)


METHOD = "autoregressive_ppo_order_012"
METHOD_TYPE = "learning_frozen_replay"
TRAIN_SCENARIO = "medium"
DEFAULT_SCENARIOS = ("medium", "time_pressure", "heterogeneity_pressure")
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task9_autoregressive_screening_30k_3seeds"
    / "models"
    / TRAIN_SCENARIO
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task10_frozen_model_diagnostics"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay frozen Task-9 autoregressive models and record Task-10 "
            "unit-order decision diagnostics."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--eval-scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=30_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--high-threat-threshold", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)
        json_file.write("\n")


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def run_diagnostics(args: argparse.Namespace) -> dict[str, int]:
    if args.eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_profiles = {
        name: get_air_defense_v1_scenario_profile(name)
        for name in args.eval_scenarios
    }
    run_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    leak_rows: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []

    for run_index, train_seed in enumerate(args.seeds):
        model_path = (
            args.model_dir
            / f"autoregressive_maskable_ppo_seed{train_seed}.zip"
        ).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Frozen Task-9 model not found: {model_path}")
        print(f"Loading frozen model: {model_path}")
        model = AutoregressiveMaskablePPO.load(
            model_path,
            device=args.device,
        )
        signature = dict(model.action_generator_signature)
        if signature.get("unit_order") != [0, 1, 2]:
            raise ValueError(
                f"Task-9 model must use frozen order [0, 1, 2], got {signature}"
            )
        model_records.append(
            {
                "run_index": run_index,
                "train_seed": train_seed,
                "model_path": str(model_path),
                "action_generator_signature": signature,
            }
        )

        for scenario_index, (scenario_name, profile) in enumerate(
            scenario_profiles.items()
        ):
            evaluation_seed = (
                args.eval_seed + scenario_index * args.eval_episodes
            )
            metadata_row = {
                "method": METHOD,
                "method_type": METHOD_TYPE,
                "train_scenario": TRAIN_SCENARIO,
                "eval_scenario": scenario_name,
                "run_index": run_index,
                "train_seed": train_seed,
                "evaluation_seed": evaluation_seed,
            }
            episode_counter = 0

            def record_episode(
                row: dict[str, Any],
                *,
                metadata_row: dict[str, Any] = metadata_row,
            ) -> None:
                nonlocal episode_counter
                episode_index = episode_counter
                episode_counter += 1
                episode_rows.append(
                    {
                        **metadata_row,
                        "episode_index": episode_index,
                        "episode_seed": evaluation_seed + episode_index,
                        **row,
                    }
                )

            def record_decision(
                episode_index: int,
                row: dict[str, Any],
                *,
                metadata_row: dict[str, Any] = metadata_row,
            ) -> None:
                decision_rows.append(
                    {
                        **metadata_row,
                        "episode_index": episode_index,
                        "episode_seed": evaluation_seed + episode_index,
                        **row,
                    }
                )

            def record_leak(
                episode_index: int,
                row: dict[str, Any],
                *,
                metadata_row: dict[str, Any] = metadata_row,
            ) -> None:
                leak_rows.append(
                    {
                        **metadata_row,
                        "episode_index": episode_index,
                        "episode_seed": evaluation_seed + episode_index,
                        **row,
                    }
                )

            print(
                f"Evaluating seed={train_seed}, scenario={scenario_name}, "
                f"episodes={args.eval_episodes}, eval_seed={evaluation_seed}"
            )
            metrics = evaluate_air_defense_v1_model(
                model,
                env_factory=lambda config=profile.config: (
                    AirDefenseResourceAssignmentEnvV1(config=config)
                ),
                episodes=args.eval_episodes,
                seed=evaluation_seed,
                deterministic=True,
                use_action_masks=True,
                high_threat_threshold=args.high_threat_threshold,
                episode_metrics_callback=record_episode,
                decision_trace_callback=record_decision,
                leak_attribution_callback=record_leak,
            )
            run_rows.append(
                {
                    **metadata_row,
                    "requested_timesteps": 30_000,
                    "training_timesteps": 30_000,
                    "training_seconds": 0.0,
                    "model_path": str(model_path),
                    **metrics,
                }
            )

    summary_rows = summarize_rows(
        run_rows,
        group_keys=("method", "method_type", "train_scenario", "eval_scenario"),
        metrics=METRIC_NAMES,
        confidence_level=args.confidence_level,
    )
    decision_summary_rows = aggregate_decision_rows(
        decision_rows,
        group_keys=(
            "method",
            "method_type",
            "train_scenario",
            "eval_scenario",
            "run_index",
            "train_seed",
            "unit_order",
            "unit_index",
            "resource_type",
            "unit_order_position",
        ),
    )
    leak_summary_rows = aggregate_leak_attributions(
        leak_rows,
        group_keys=(
            "method",
            "method_type",
            "train_scenario",
            "eval_scenario",
            "run_index",
            "train_seed",
        ),
    )

    outputs = {
        "runs.csv": (run_rows, RUN_FIELDNAMES),
        "episodes.csv": (episode_rows, EPISODE_FIELDNAMES),
        "summary.csv": (summary_rows, SUMMARY_FIELDNAMES),
        "decisions.csv": (decision_rows, DECISION_FIELDNAMES),
        "decision_summary.csv": (
            decision_summary_rows,
            DECISION_SUMMARY_FIELDNAMES,
        ),
        "leak_attributions.csv": (leak_rows, LEAK_ATTRIBUTION_FIELDNAMES),
        "leak_attribution_summary.csv": (
            leak_summary_rows,
            LEAK_ATTRIBUTION_SUMMARY_FIELDNAMES,
        ),
    }
    for filename, (rows, fieldnames) in outputs.items():
        _write_csv(output_dir / filename, rows, fieldnames)

    result_counts = {
        filename.removesuffix(".csv") + "_rows": len(rows)
        for filename, (rows, _) in outputs.items()
    }
    _write_json(
        output_dir / "experiment_config.json",
        {
            "schema_version": 6,
            "experiment_type": "task10_frozen_model_decision_diagnostics",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": METHOD,
            "train_scenario": TRAIN_SCENARIO,
            "model_source": str(args.model_dir.resolve()),
            "models": model_records,
            "protocol": {
                "train_seeds": list(args.seeds),
                "eval_scenarios": list(scenario_profiles),
                "eval_episodes": args.eval_episodes,
                "eval_seed": args.eval_seed,
                "scenario_seed_formula": (
                    "eval_seed + scenario_index * eval_episodes + episode_index; "
                    "identical episode seeds are paired across frozen models"
                ),
                "deterministic": True,
                "high_threat_threshold": args.high_threat_threshold,
                "confidence_level": args.confidence_level,
                "record_decisions": True,
                "retraining": False,
            },
            "scenarios": {
                name: asdict(profile.config)
                for name, profile in scenario_profiles.items()
            },
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "device": args.device,
                "packages": {
                    package: _version(package)
                    for package in (
                        "gymnasium",
                        "numpy",
                        "stable-baselines3",
                        "sb3-contrib",
                        "torch",
                    )
                },
            },
            "result_counts": result_counts,
        },
    )
    return result_counts


def main() -> None:
    args = parse_args()
    counts = run_diagnostics(args)
    print(f"Task-10 frozen diagnostics written to: {args.output_dir.resolve()}")
    for name, count in counts.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
