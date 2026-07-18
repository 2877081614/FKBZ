from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.experiments import (
    ALL_BENCHMARK_METHODS,
    AirDefenseV1BenchmarkConfig,
    RULE_POLICY_FACTORIES,
    run_air_defense_v1_benchmark,
)
from rein_learning.envs import AIR_DEFENSE_V1_SCENARIO_NAMES
from rein_learning.trainers.air_defense_v1_ppo import AirDefenseV1PPOConfig


DISPLAY_METRICS = (
    "avg_reward",
    "success_rate",
    "intercept_rate",
    "leak_rate",
    "avg_total_damage",
    "avg_invalid_actions",
    "avg_decision_time_ms",
    "high_threat_leak_rate",
    "assignment_conflict_rate",
    "overkill_rate",
    "damage_reduction_per_ammo",
    "avg_resource_cost",
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
    parser.add_argument("--high-threat-threshold", type=float, default=0.8)
    parser.add_argument(
        "--train-scenario",
        dest="train_scenarios",
        nargs="+",
        choices=AIR_DEFENSE_V1_SCENARIO_NAMES,
        default=("medium",),
        help="One or more named scenarios used to train each learning method.",
    )
    parser.add_argument(
        "--eval-scenarios",
        nargs="+",
        choices=AIR_DEFENSE_V1_SCENARIO_NAMES,
        default=("medium",),
        help="Named scenarios evaluated with paired scenario seeds.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_BENCHMARK_METHODS,
        default=None,
        help="Methods to include; defaults to all rule and learning methods.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="Run all six rule baselines without PPO training.",
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
    parser.add_argument(
        "--record-decisions",
        action="store_true",
        help="Write unit-level final-evaluation decisions and leak attributions.",
    )
    parser.add_argument(
        "--record-training-dynamics",
        action="store_true",
        help="Record PPO optimization statistics at fixed training intervals.",
    )
    parser.add_argument(
        "--diagnostics-freq",
        type=int,
        default=1_000,
        help="Training-step interval for optimization and policy-probe diagnostics.",
    )
    parser.add_argument(
        "--probe-corpus",
        type=Path,
        default=None,
        help="Directory or probe_states.npz used for frozen-state diagnostics.",
    )
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
        (
            str(row["method"]),
            str(row["train_scenario"]),
            str(row["eval_scenario"]),
            str(row["metric"]),
        ): row
        for row in summary_rows
    }
    evaluation_groups = sorted(
        {
            (
                str(row["method"]),
                str(row["train_scenario"]),
                str(row["eval_scenario"]),
            )
            for row in summary_rows
        }
    )
    print(
        "method                            train          eval           n    reward [CI]       success   intercept "
        " leak     damage   invalid  decision_ms  high_leak  conflict  "
        "overkill  dmg/ammo  cost"
    )
    for method, train_scenario, eval_scenario in evaluation_groups:
        reward = indexed[(method, train_scenario, eval_scenario, "avg_reward")]
        values = {
            metric: float(
                indexed[(method, train_scenario, eval_scenario, metric)]["mean"]
            )
            for metric in DISPLAY_METRICS
        }
        print(
            f"{method:<34}"
            f"{train_scenario:<15}"
            f"{eval_scenario:<15}"
            f"{int(reward['n_runs']):>3}  "
            f"{values['avg_reward']:>7.2f} "
            f"[{float(reward['ci_low']):>7.2f}, {float(reward['ci_high']):>7.2f}]"
            f"{values['success_rate']:>10.2f}"
            f"{values['intercept_rate']:>12.2f}"
            f"{values['leak_rate']:>7.2f}"
            f"{values['avg_total_damage']:>9.2f}"
            f"{values['avg_invalid_actions']:>10.2f}"
            f"{values['avg_decision_time_ms']:>13.3f}"
            f"{values['high_threat_leak_rate']:>11.2f}"
            f"{values['assignment_conflict_rate']:>10.2f}"
            f"{values['overkill_rate']:>10.2f}"
            f"{values['damage_reduction_per_ammo']:>10.2f}"
            f"{values['avg_resource_cost']:>7.2f}"
        )


def main() -> None:
    args = parse_args()
    if args.rules_only and args.methods is not None:
        raise ValueError("Use either --rules-only or --methods, not both")
    seeds = _resolve_seeds(args)
    output_dir = _experiment_output_dir(args)
    selected_methods = None
    if args.rules_only:
        selected_methods = tuple(RULE_POLICY_FACTORIES)
    elif args.methods is not None:
        selected_methods = tuple(args.methods)
    protocol = AirDefenseV1BenchmarkConfig(
        train_seeds=seeds,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        curve_eval_freq=args.curve_eval_freq,
        curve_eval_episodes=args.curve_eval_episodes,
        curve_eval_seed=args.curve_eval_seed,
        confidence_level=args.confidence_level,
        high_threat_threshold=args.high_threat_threshold,
        train_scenarios=tuple(args.train_scenarios),
        eval_scenarios=tuple(args.eval_scenarios),
        methods=selected_methods,
        include_learning=not args.rules_only,
        save_models=not args.no_save_models,
        create_plot=not args.no_plot,
        record_decisions=args.record_decisions,
        record_training_dynamics=(
            args.record_training_dynamics or args.probe_corpus is not None
        ),
        diagnostics_freq=args.diagnostics_freq,
        probe_corpus_path=(
            str(args.probe_corpus) if args.probe_corpus is not None else None
        ),
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
    print(f"episodes={result.artifacts.episodes}")
    print(f"summary={result.artifacts.summary}")
    print(f"paired_differences={result.artifacts.paired_differences}")
    print(f"generalization_matrix={result.artifacts.generalization_matrix}")
    if result.curve_rows:
        print(f"learning_curves={result.artifacts.learning_curves}")
    if result.training_dynamics_rows:
        print(f"training_dynamics={result.artifacts.training_dynamics}")
    if result.probe_dynamics_rows:
        print(f"probe_dynamics={result.artifacts.probe_dynamics}")
    for figure_path in result.figure_paths:
        print(f"figure={figure_path}")


if __name__ == "__main__":
    main()
