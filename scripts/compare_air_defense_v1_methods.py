from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.experiments import (
    AirDefenseV1BenchmarkConfig,
    run_air_defense_v1_benchmark,
)
from rein_learning.trainers.air_defense_v1_ppo import AirDefenseV1PPOConfig


DISPLAY_METRICS = (
    "avg_reward",
    "success_rate",
    "intercept_rate",
    "leak_rate",
    "avg_total_damage",
    "avg_invalid_actions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible multi-seed AirDefense v1.0 benchmark with "
            "rule baselines, PPO, Maskable PPO, confidence intervals, and curves."
        )
    )
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=(128, 128),
    )
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(0, 1, 2, 3, 4),
        help="Training seeds; rule baselines use the paired evaluation block for each run.",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="Deprecated single-seed alias retained for earlier commands.",
    )
    parser.add_argument("--eval-seed", type=int, default=200)
    parser.add_argument("--curve-eval-freq", type=int, default=5_000)
    parser.add_argument("--curve-eval-episodes", type=int, default=10)
    parser.add_argument("--curve-eval-seed", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="Run all five rule baselines without PPO training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "air_defense_v1",
        help="Parent directory for timestamped experiment folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Deprecated compatibility option. Its parent and stem become the "
            "experiment output directory."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Output folder name; defaults to benchmark_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def _resolve_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    if args.train_seed is not None:
        if tuple(args.seeds) != (0, 1, 2, 3, 4):
            raise ValueError("Use either --seeds or --train-seed, not both")
        return (args.train_seed,)
    return tuple(args.seeds)


def _experiment_output_dir(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output.parent / args.output.stem
    experiment_name = args.experiment_name
    if experiment_name is None:
        experiment_name = datetime.now().strftime("benchmark_%Y%m%d_%H%M%S")
    return args.output_dir / experiment_name


def _print_summary(summary_rows: tuple[dict[str, object], ...]) -> None:
    indexed = {
        (str(row["method"]), str(row["metric"])): row
        for row in summary_rows
    }
    methods = sorted({str(row["method"]) for row in summary_rows})
    print(
        "method                 n    reward [CI]       success   intercept "
        " leak     damage   invalid"
    )
    for method in methods:
        reward = indexed[(method, "avg_reward")]
        values = {
            metric: float(indexed[(method, metric)]["mean"])
            for metric in DISPLAY_METRICS
        }
        print(
            f"{method:<22}"
            f"{int(reward['n_runs']):>3}  "
            f"{values['avg_reward']:>7.2f} "
            f"[{float(reward['ci_low']):>7.2f}, {float(reward['ci_high']):>7.2f}]"
            f"{values['success_rate']:>10.2f}"
            f"{values['intercept_rate']:>12.2f}"
            f"{values['leak_rate']:>7.2f}"
            f"{values['avg_total_damage']:>9.2f}"
            f"{values['avg_invalid_actions']:>10.2f}"
        )


def main() -> None:
    args = parse_args()
    seeds = _resolve_seeds(args)
    output_dir = _experiment_output_dir(args)
    protocol = AirDefenseV1BenchmarkConfig(
        train_seeds=seeds,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        curve_eval_freq=args.curve_eval_freq,
        curve_eval_episodes=args.curve_eval_episodes,
        curve_eval_seed=args.curve_eval_seed,
        confidence_level=args.confidence_level,
        include_learning=not args.rules_only,
        save_models=not args.no_save_models,
        create_plot=not args.no_plot,
    )
    training = AirDefenseV1PPOConfig(
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        net_arch=tuple(args.net_arch),
        device=args.device,
        verbose=args.verbose,
        progress_bar=args.progress_bar,
    )
    result = run_air_defense_v1_benchmark(
        output_dir=output_dir,
        benchmark_config=protocol,
        train_config=training,
        progress_callback=lambda message: print(f"[benchmark] {message}", flush=True),
    )

    _print_summary(result.summary_rows)
    print(f"experiment_dir={result.artifacts.output_dir}")
    print(f"config={result.artifacts.config}")
    print(f"runs={result.artifacts.runs}")
    print(f"summary={result.artifacts.summary}")
    if result.curve_rows:
        print(f"learning_curves={result.artifacts.learning_curves}")
        for figure_path in result.figure_paths:
            print(f"curve_figure={figure_path}")


if __name__ == "__main__":
    main()
