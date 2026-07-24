from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.experiments import (
    AirDefenseV1BenchmarkConfig,
    run_air_defense_v1_benchmark,
)
from rein_learning.trainers.air_defense_v1_ppo import AirDefenseV1PPOConfig


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "mch_ppo_mechanism_stress_test"
)
DEFAULT_Q_CRITICS = tuple(
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_hierarchical_q"
    / "models"
    / f"hierarchical_seed{seed}.pt"
    for seed in (14, 15, 16)
)
METHODS = (
    "factorized_engagement_ar_ppo_order_012",
    "mch_ppo_order_012",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered MCH-PPO mechanism stress test."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=(8, 9, 10))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=("time_pressure", "heterogeneity_pressure"),
    )
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--q-critics", type=Path, nargs="+", default=DEFAULT_Q_CRITICS
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 256 steps, one seed, one scenario and two evaluation episodes.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Regenerate the gate report from an existing runs.csv.",
    )
    return parser.parse_args()


def _paired_summary(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    baseline_name, candidate_name = METHODS
    on_scenario = [
        row for row in rows if row["train_scenario"] == row["eval_scenario"]
    ]
    lookup = {
        (row["method"], row["train_scenario"], int(row["train_seed"])): row
        for row in on_scenario
    }
    scenario_names = sorted(
        {str(row["train_scenario"]) for row in on_scenario}
    )
    scenario_summaries: dict[str, Any] = {}
    all_structural_zero = True
    all_noop_gate = True
    collapsed_candidate_runs = 0
    any_high_threat_improvement = False
    cost_gate = True
    safety_gate = True
    for scenario in scenario_names:
        seeds = sorted(
            int(row["train_seed"])
            for row in on_scenario
            if row["method"] == candidate_name
            and row["train_scenario"] == scenario
        )
        paired_rows = []
        for seed in seeds:
            baseline = lookup[(baseline_name, scenario, seed)]
            candidate = lookup[(candidate_name, scenario, seed)]
            paired_rows.append(
                {
                    "train_seed": seed,
                    "all_noop_delta": float(candidate["all_noop_episode_rate"])
                    - float(baseline["all_noop_episode_rate"]),
                    "high_threat_leak_delta": float(
                        candidate["high_threat_leak_rate"]
                    )
                    - float(baseline["high_threat_leak_rate"]),
                    "resource_cost_delta": float(candidate["avg_resource_cost"])
                    - float(baseline["avg_resource_cost"]),
                    "reward_delta": float(candidate["avg_reward"])
                    - float(baseline["avg_reward"]),
                    "damage_delta": float(candidate["avg_total_damage"])
                    - float(baseline["avg_total_damage"]),
                }
            )
            structural = max(
                float(candidate["avg_invalid_actions"]),
                float(candidate["assignment_conflict_rate"]),
                float(candidate["overkill_rate"]),
            )
            all_structural_zero = all_structural_zero and structural == 0.0
            if (
                float(candidate["all_noop_episode_rate"]) >= 0.98
                or float(candidate["actionable_engagement_rate"]) < 0.01
            ):
                collapsed_candidate_runs += 1

        noop_noninferior = sum(
            row["all_noop_delta"] <= 0.0 for row in paired_rows
        )
        mean_high_threat_delta = sum(
            row["high_threat_leak_delta"] for row in paired_rows
        ) / len(paired_rows)
        baseline_cost = sum(
            float(lookup[(baseline_name, scenario, seed)]["avg_resource_cost"])
            for seed in seeds
        ) / len(seeds)
        candidate_cost = sum(
            float(lookup[(candidate_name, scenario, seed)]["avg_resource_cost"])
            for seed in seeds
        ) / len(seeds)
        cost_ratio = candidate_cost / baseline_cost if baseline_cost > 0.0 else 1.0
        mean_reward_delta = sum(row["reward_delta"] for row in paired_rows) / len(
            paired_rows
        )
        mean_damage_delta = sum(row["damage_delta"] for row in paired_rows) / len(
            paired_rows
        )
        scenario_all_noop_passed = noop_noninferior >= max(1, len(seeds) - 1)
        scenario_cost_passed = cost_ratio <= 1.10
        scenario_safety_passed = mean_reward_delta >= -10.0 and mean_damage_delta <= 0.20
        all_noop_gate = all_noop_gate and scenario_all_noop_passed
        any_high_threat_improvement = (
            any_high_threat_improvement or mean_high_threat_delta < 0.0
        )
        cost_gate = cost_gate and scenario_cost_passed
        safety_gate = safety_gate and scenario_safety_passed
        scenario_summaries[scenario] = {
            "paired_runs": paired_rows,
            "all_noop_noninferior_seed_count": noop_noninferior,
            "all_noop_gate_passed": scenario_all_noop_passed,
            "mean_high_threat_leak_delta": mean_high_threat_delta,
            "resource_cost_ratio": cost_ratio,
            "cost_gate_passed": scenario_cost_passed,
            "mean_reward_delta": mean_reward_delta,
            "mean_damage_delta": mean_damage_delta,
            "safety_gate_passed": scenario_safety_passed,
        }

    gates = {
        "structural_zero": all_structural_zero,
        "all_noop_noninferiority": all_noop_gate,
        "no_collapsed_candidate_runs": collapsed_candidate_runs == 0,
        "high_threat_improvement": any_high_threat_improvement,
        "resource_cost": cost_gate,
        "reward_damage_safety": safety_gate,
    }
    return {
        "schema_version": 1,
        "methods": list(METHODS),
        "scenarios": scenario_summaries,
        "collapsed_candidate_run_count": collapsed_candidate_runs,
        "gates": gates,
        "mechanism_gate_passed": all(gates.values()),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# AirDefense v1 MCH-PPO 机制压力实验",
        "",
        "## 门控结论",
        "",
        f"- 总门控：`{str(summary['mechanism_gate_passed']).lower()}`",
        f"- 候选塌缩场景种子数：`{summary['collapsed_candidate_run_count']}`",
    ]
    for name, passed in summary["gates"].items():
        lines.append(f"- `{name}`：`{str(passed).lower()}`")
    for scenario, record in summary["scenarios"].items():
        lines.extend(
            [
                "",
                f"## {scenario}",
                "",
                f"- all-noop 非劣种子数：{record['all_noop_noninferior_seed_count']}",
                f"- 高威胁突防率均值差：{record['mean_high_threat_leak_delta']:.6f}",
                f"- 资源成本比：{record['resource_cost_ratio']:.6f}",
                f"- 奖励均值差：{record['mean_reward_delta']:.6f}",
                f"- 损伤均值差：{record['mean_damage_delta']:.6f}",
                "",
                "| seed | all-noop 差 | 高威胁突防差 | 成本差 | 奖励差 | 损伤差 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in record["paired_runs"]:
            lines.append(
                f"| {row['train_seed']} | {row['all_noop_delta']:.6f} | "
                f"{row['high_threat_leak_delta']:.6f} | "
                f"{row['resource_cost_delta']:.6f} | "
                f"{row['reward_delta']:.6f} | {row['damage_delta']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "本结果是冻结困难场景上的机制筛选。通过门控只允许进入更大预算正式实验；未通过时不得挑选单个优势种子宣称 MCH-PPO 普遍优越。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.smoke:
        timesteps = 256
        seeds = (8,)
        scenarios = ("time_pressure",)
        eval_episodes = 2
        output_dir = args.output_dir.parent / f"{args.output_dir.name}_smoke"
    else:
        timesteps = args.timesteps
        seeds = tuple(args.seeds)
        scenarios = tuple(args.scenarios)
        eval_episodes = args.eval_episodes
        output_dir = args.output_dir
    if args.analyze_only:
        with (output_dir / "runs.csv").open(newline="", encoding="utf-8") as handle:
            rows = tuple(csv.DictReader(handle))
        summary = _paired_summary(rows)
        _write_summary(output_dir, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"output_dir={output_dir.resolve()}")
        return
    missing = [str(path) for path in args.q_critics if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Q-Critic checkpoints: {missing}")

    protocol = AirDefenseV1BenchmarkConfig(
        train_seeds=seeds,
        eval_episodes=eval_episodes,
        curve_eval_freq=max(128, timesteps // 2),
        curve_eval_episodes=max(1, min(5, eval_episodes)),
        train_scenarios=scenarios,
        eval_scenarios=scenarios,
        methods=METHODS,
        save_models=True,
        create_plot=not args.smoke,
        record_decisions=True,
    )
    training = AirDefenseV1PPOConfig(
        total_timesteps=timesteps,
        n_steps=min(256, timesteps),
        batch_size=64,
        n_epochs=args.n_epochs,
        seed=seeds[0],
        device=args.device,
        verbose=0,
        mch_q_critic_paths=tuple(str(path.resolve()) for path in args.q_critics),
    )
    result = run_air_defense_v1_benchmark(
        output_dir=output_dir,
        benchmark_config=protocol,
        train_config=training,
        progress_callback=lambda message: print(message, flush=True),
    )
    summary = _paired_summary(result.run_rows)
    _write_summary(output_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir={output_dir.resolve()}")


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    summary_path = output_dir / "mch_stress_summary.json"
    report_path = output_dir / "mch_stress_summary.md"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
