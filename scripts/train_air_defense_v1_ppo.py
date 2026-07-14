from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.trainers.air_defense_v1_ppo import (
    AirDefenseV1PPOConfig,
    evaluate_air_defense_v1_model,
    train_maskable_ppo,
    train_ppo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO or Maskable PPO on AirDefenseResourceAssignmentEnv v1.0."
    )
    parser.add_argument(
        "--algorithm",
        choices=("ppo", "maskable-ppo", "both"),
        default="maskable-ppo",
        help="Which learning baseline to train.",
    )
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=1_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "air_defense_v1",
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def make_train_config(args: argparse.Namespace) -> AirDefenseV1PPOConfig:
    return AirDefenseV1PPOConfig(
        total_timesteps=args.timesteps,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        tensorboard_log=str(PROJECT_ROOT / "runs" / "air_defense_v1"),
    )


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name:<14}"
        f"{metrics['avg_reward']:>10.2f}"
        f"{metrics['success_rate']:>9.2f}"
        f"{metrics['intercept_rate']:>11.2f}"
        f"{metrics['leak_rate']:>7.2f}"
        f"{metrics['avg_total_damage']:>9.2f}"
        f"{metrics['avg_ammo_used']:>7.2f}"
        f"{metrics['avg_shots']:>7.2f}"
        f"{metrics['hit_rate_per_shot']:>10.2f}"
        f"{metrics['avg_invalid_actions']:>9.2f}"
    )


def main() -> None:
    args = parse_args()
    config = make_train_config(args)
    algorithms = (
        ("ppo", train_ppo, False),
        ("maskable_ppo", train_maskable_ppo, True),
    )
    if args.algorithm == "ppo":
        algorithms = algorithms[:1]
    elif args.algorithm == "maskable-ppo":
        algorithms = algorithms[1:]

    print(
        "method        avg_reward  success  intercept  leak   damage  "
        "ammo  shots  hit/shot  invalid"
    )
    for name, train_fn, use_action_masks in algorithms:
        save_path = None
        if not args.no_save:
            save_path = args.save_dir / f"{name}_seed{args.seed}.zip"
        model = train_fn(train_config=config, save_path=save_path)
        metrics = evaluate_air_defense_v1_model(
            model,
            episodes=args.eval_episodes,
            seed=args.eval_seed,
            use_action_masks=use_action_masks,
        )
        print_metrics(name, metrics)
        if save_path is not None:
            print(f"saved_model={save_path}")


if __name__ == "__main__":
    main()
