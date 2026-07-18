from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.algorithms.policy_gradient import (
    RoleConditionedAutoregressiveMaskablePPO,
)
from rein_learning.common import PolicyProbeCorpus, evaluate_policy_probe
from rein_learning.envs import get_air_defense_v1_scenario
from rein_learning.trainers.air_defense_v1_ppo import evaluate_air_defense_v1_model


DEFAULT_MODELS_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task11_role_conditioned_screening_30k_3seeds"
    / "models"
    / "medium"
)
DEFAULT_PROBE_DIR = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task12_probe_corpus"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task12_task11_frozen_replay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay frozen Task 11 models to diagnose no-op collapse."
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--probe-corpus", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=("medium", "time_pressure", "heterogeneity_pressure"),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=50_000)
    return parser.parse_args()


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in materialized for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    corpus = PolicyProbeCorpus.load(args.probe_corpus)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    for train_seed in args.seeds:
        model_path = (
            args.models_dir
            / f"role_conditioned_ar_ppo_order_012_seed{train_seed}.zip"
        )
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = RoleConditionedAutoregressiveMaskablePPO.load(
            model_path, device="cpu"
        )
        for deterministic in (True, False):
            mode = "deterministic" if deterministic else "stochastic"
            diagnostic_seed = 60_000 + train_seed * 10 + int(not deterministic)
            random.seed(diagnostic_seed)
            np.random.seed(diagnostic_seed)
            torch.manual_seed(diagnostic_seed)
            for row in evaluate_policy_probe(
                model, corpus, deterministic=deterministic
            ):
                row = dict(row)
                row["probe_engagement_rate"] = row.pop(
                    "deterministic_engagement_rate"
                )
                probe_rows.append(
                    {
                        "train_seed": train_seed,
                        "evaluation_mode": mode,
                        "diagnostic_seed": diagnostic_seed,
                        **row,
                    }
                )
            for scenario_index, scenario in enumerate(args.scenarios):
                evaluation_seed = (
                    args.eval_seed + scenario_index * args.episodes
                )
                current_episodes: list[dict[str, Any]] = []

                def record_episode(raw: dict[str, Any]) -> None:
                    current_episodes.append(
                        {
                            "train_seed": train_seed,
                            "evaluation_mode": mode,
                            "eval_scenario": scenario,
                            "evaluation_seed": evaluation_seed,
                            "episode_index": len(current_episodes),
                            **raw,
                        }
                    )

                random.seed(diagnostic_seed)
                np.random.seed(diagnostic_seed)
                torch.manual_seed(diagnostic_seed)
                metrics = evaluate_air_defense_v1_model(
                    model,
                    env_config=get_air_defense_v1_scenario(scenario),
                    episodes=args.episodes,
                    seed=evaluation_seed,
                    deterministic=deterministic,
                    use_action_masks=True,
                    episode_metrics_callback=record_episode,
                )
                episode_rows.extend(current_episodes)
                run_rows.append(
                    {
                        "train_seed": train_seed,
                        "evaluation_mode": mode,
                        "eval_scenario": scenario,
                        "evaluation_seed": evaluation_seed,
                        "episodes": args.episodes,
                        "model_path": str(model_path.resolve()),
                        **metrics,
                    }
                )
                print(
                    f"seed={train_seed} mode={mode} scenario={scenario} "
                    f"engage={metrics['actionable_engagement_rate']:.4f} "
                    f"all_noop={metrics['all_noop_episode_rate']:.4f}",
                    flush=True,
                )

    _write_csv(output_dir / "runs.csv", run_rows)
    _write_csv(output_dir / "episodes.csv", episode_rows)
    _write_csv(output_dir / "probe_diagnostics.csv", probe_rows)
    config = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "task12_task11_frozen_noop_replay",
        "models_dir": str(args.models_dir.resolve()),
        "probe_corpus": str(args.probe_corpus.resolve()),
        "probe_sha256": corpus.content_sha256(),
        "train_seeds": list(args.seeds),
        "scenarios": list(args.scenarios),
        "episodes": args.episodes,
        "eval_seed": args.eval_seed,
        "evaluation_modes": ["deterministic", "stochastic"],
        "run_rows": len(run_rows),
        "episode_rows": len(episode_rows),
        "probe_rows": len(probe_rows),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
