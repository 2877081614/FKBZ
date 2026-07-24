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


RESULTS_ROOT = PROJECT_ROOT / "results" / "air_defense_v1"
DEFAULT_OUTPUT = RESULTS_ROOT / "sa_rg_mch_ppo_mechanism_stress_test"
DEFAULT_MCH_REFERENCE = RESULTS_ROOT / "mch_ppo_mechanism_stress_test"
DEFAULT_RG_REFERENCE = RESULTS_ROOT / "rg_mch_ppo_mechanism_stress_test"
DEFAULT_SUPPORT_DATASET = RESULTS_ROOT / "task14_q_critic" / "dataset.npz"
DEFAULT_Q_CRITICS = tuple(
    RESULTS_ROOT
    / "task14_hierarchical_q"
    / "models"
    / f"hierarchical_seed{seed}.pt"
    for seed in (14, 15, 16)
)
BASELINE = "factorized_engagement_ar_ppo_order_012"
MCH_V0 = "mch_ppo_order_012"
RG_MCH = "rg_mch_ppo_order_012"
CANDIDATE = "sa_rg_mch_ppo_order_012"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered SA-RG-MCH-PPO stress test."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mch-reference-dir", type=Path, default=DEFAULT_MCH_REFERENCE
    )
    parser.add_argument(
        "--rg-reference-dir", type=Path, default=DEFAULT_RG_REFERENCE
    )
    parser.add_argument(
        "--support-dataset", type=Path, default=DEFAULT_SUPPORT_DATASET
    )
    parser.add_argument("--q-critics", type=Path, nargs="+", default=DEFAULT_Q_CRITICS)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=(8, 9, 10))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=("time_pressure", "heterogeneity_pressure"),
    )
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _number(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(_number(row, key) for row in rows) / len(rows)


def _build_summary(
    candidate_rows: tuple[dict[str, Any], ...],
    mch_reference_rows: tuple[dict[str, Any], ...],
    rg_reference_rows: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    all_rows = candidate_rows + mch_reference_rows + rg_reference_rows
    on_scenario = [
        row
        for row in all_rows
        if row["train_scenario"] == row["eval_scenario"]
    ]
    lookup = {
        (row["method"], row["train_scenario"], int(row["train_seed"])): row
        for row in on_scenario
    }
    scenarios = sorted(
        {
            row["train_scenario"]
            for row in on_scenario
            if row["method"] == CANDIDATE
        }
    )
    scenario_records: dict[str, Any] = {}
    collapsed_count = 0
    structural_zero = True
    all_noop_gate = True
    safety_gate = True
    cost_gate = True
    high_threat_gate = False
    mch_v0_gate = True
    rg_noncatastrophic_gate = True

    for scenario in scenarios:
        seeds = sorted(
            int(row["train_seed"])
            for row in on_scenario
            if row["method"] == CANDIDATE
            and row["train_scenario"] == scenario
        )
        candidates: list[dict[str, Any]] = []
        baselines: list[dict[str, Any]] = []
        mch_rows: list[dict[str, Any]] = []
        rg_rows: list[dict[str, Any]] = []
        paired: list[dict[str, Any]] = []
        for seed in seeds:
            candidate = lookup[(CANDIDATE, scenario, seed)]
            baseline = lookup[(BASELINE, scenario, seed)]
            mch_v0 = lookup[(MCH_V0, scenario, seed)]
            rg_mch = lookup[(RG_MCH, scenario, seed)]
            candidates.append(candidate)
            baselines.append(baseline)
            mch_rows.append(mch_v0)
            rg_rows.append(rg_mch)
            collapsed = (
                _number(candidate, "all_noop_episode_rate") >= 0.98
                or _number(candidate, "actionable_engagement_rate") < 0.01
            )
            collapsed_count += int(collapsed)
            structural_zero = structural_zero and max(
                _number(candidate, "avg_invalid_actions"),
                _number(candidate, "assignment_conflict_rate"),
                _number(candidate, "overkill_rate"),
            ) == 0.0
            paired.append(
                {
                    "train_seed": seed,
                    "candidate_collapsed": collapsed,
                    "all_noop_delta_vs_baseline": _number(
                        candidate, "all_noop_episode_rate"
                    )
                    - _number(baseline, "all_noop_episode_rate"),
                    "high_threat_leak_delta_vs_baseline": _number(
                        candidate, "high_threat_leak_rate"
                    )
                    - _number(baseline, "high_threat_leak_rate"),
                    "reward_delta_vs_baseline": _number(candidate, "avg_reward")
                    - _number(baseline, "avg_reward"),
                    "damage_delta_vs_baseline": _number(
                        candidate, "avg_total_damage"
                    )
                    - _number(baseline, "avg_total_damage"),
                    "cost_delta_vs_baseline": _number(
                        candidate, "avg_resource_cost"
                    )
                    - _number(baseline, "avg_resource_cost"),
                    "reward_delta_vs_rg_mch": _number(candidate, "avg_reward")
                    - _number(rg_mch, "avg_reward"),
                    "damage_delta_vs_rg_mch": _number(
                        candidate, "avg_total_damage"
                    )
                    - _number(rg_mch, "avg_total_damage"),
                }
            )

        noop_count = sum(
            row["all_noop_delta_vs_baseline"] <= 0.0 for row in paired
        )
        reward_delta = _mean(candidates, "avg_reward") - _mean(
            baselines, "avg_reward"
        )
        damage_delta = _mean(candidates, "avg_total_damage") - _mean(
            baselines, "avg_total_damage"
        )
        high_threat_delta = _mean(
            candidates, "high_threat_leak_rate"
        ) - _mean(baselines, "high_threat_leak_rate")
        candidate_cost = _mean(candidates, "avg_resource_cost")
        baseline_cost = _mean(baselines, "avg_resource_cost")
        cost_ratio = candidate_cost / baseline_cost if baseline_cost > 0.0 else 1.0
        reward_delta_mch = _mean(candidates, "avg_reward") - _mean(
            mch_rows, "avg_reward"
        )
        damage_delta_mch = _mean(candidates, "avg_total_damage") - _mean(
            mch_rows, "avg_total_damage"
        )
        reward_delta_rg = _mean(candidates, "avg_reward") - _mean(
            rg_rows, "avg_reward"
        )
        damage_delta_rg = _mean(candidates, "avg_total_damage") - _mean(
            rg_rows, "avg_total_damage"
        )
        scenario_noop_gate = noop_count >= max(1, len(seeds) - 1)
        scenario_safety_gate = reward_delta >= -10.0 and damage_delta <= 0.20
        scenario_cost_gate = cost_ratio <= 1.10
        scenario_mch_gate = reward_delta_mch > 0.0 and damage_delta_mch < 0.0
        scenario_rg_noncatastrophic = not (
            reward_delta_rg < -10.0 and damage_delta_rg > 0.20
        )
        all_noop_gate = all_noop_gate and scenario_noop_gate
        safety_gate = safety_gate and scenario_safety_gate
        cost_gate = cost_gate and scenario_cost_gate
        high_threat_gate = high_threat_gate or high_threat_delta < 0.0
        mch_v0_gate = mch_v0_gate and scenario_mch_gate
        rg_noncatastrophic_gate = (
            rg_noncatastrophic_gate and scenario_rg_noncatastrophic
        )
        scenario_records[scenario] = {
            "paired_runs": paired,
            "all_noop_noninferior_seed_count": noop_count,
            "all_noop_gate_passed": scenario_noop_gate,
            "mean_reward_delta_vs_baseline": reward_delta,
            "mean_damage_delta_vs_baseline": damage_delta,
            "mean_high_threat_leak_delta_vs_baseline": high_threat_delta,
            "resource_cost_ratio_vs_baseline": cost_ratio,
            "safety_gate_passed": scenario_safety_gate,
            "cost_gate_passed": scenario_cost_gate,
            "mean_reward_delta_vs_mch_v0": reward_delta_mch,
            "mean_damage_delta_vs_mch_v0": damage_delta_mch,
            "mch_v0_gate_passed": scenario_mch_gate,
            "mean_reward_delta_vs_rg_mch": reward_delta_rg,
            "mean_damage_delta_vs_rg_mch": damage_delta_rg,
            "rg_noncatastrophic_gate_passed": scenario_rg_noncatastrophic,
        }

    candidate_training = [
        row for row in on_scenario if row["method"] == CANDIDATE
    ]
    baseline_training = [
        row for row in on_scenario if row["method"] == BASELINE
    ]
    diagnostics = {
        key: _mean(candidate_training, key)
        for key in (
            "mch_engagement_reliability",
            "mch_target_reliability",
            "mch_engagement_support",
            "mch_target_support",
            "mch_engagement_residual_abs",
            "mch_target_residual_abs",
            "mch_anchor_kl",
            "mch_anchor_penalty",
            "mch_anchor_excess_rate",
        )
    }
    diagnostics["training_time_ratio_vs_baseline"] = _mean(
        candidate_training, "training_seconds"
    ) / _mean(baseline_training, "training_seconds")
    gates = {
        "structural_zero": structural_zero,
        "no_collapsed_candidate_runs": collapsed_count == 0,
        "all_noop_noninferiority": all_noop_gate,
        "reward_damage_safety": safety_gate,
        "high_threat_improvement": high_threat_gate,
        "resource_cost": cost_gate,
        "improves_mch_v0_both_scenarios": mch_v0_gate,
        "reduces_rg_mch_collapse_count": collapsed_count < 2,
        "noncatastrophic_vs_rg_mch": rg_noncatastrophic_gate,
    }
    return {
        "schema_version": 1,
        "methods": [BASELINE, MCH_V0, RG_MCH, CANDIDATE],
        "scenarios": scenario_records,
        "collapsed_candidate_run_count": collapsed_count,
        "rg_mch_collapsed_run_count": 2,
        "training_diagnostics": diagnostics,
        "gates": gates,
        "mechanism_gate_passed": all(gates.values()),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# AirDefense v1 SA-RG-MCH-PPO 机制压力实验",
        "",
        "## 总门控",
        "",
        f"- 通过：`{str(summary['mechanism_gate_passed']).lower()}`",
        f"- 候选塌缩数：{summary['collapsed_candidate_run_count']}",
        f"- RG-MCH 塌缩数：{summary['rg_mch_collapsed_run_count']}",
    ]
    for name, passed in summary["gates"].items():
        lines.append(f"- `{name}`：`{str(passed).lower()}`")
    lines.extend(["", "## 训练诊断", ""])
    for name, value in summary["training_diagnostics"].items():
        lines.append(f"- `{name}`：{value:.6f}")
    for scenario, record in summary["scenarios"].items():
        lines.extend(
            [
                "",
                f"## {scenario}",
                "",
                f"- 奖励差 vs baseline：{record['mean_reward_delta_vs_baseline']:.6f}",
                f"- 损伤差 vs baseline：{record['mean_damage_delta_vs_baseline']:.6f}",
                f"- 突防差 vs baseline：{record['mean_high_threat_leak_delta_vs_baseline']:.6f}",
                f"- 奖励差 vs RG-MCH：{record['mean_reward_delta_vs_rg_mch']:.6f}",
                f"- 损伤差 vs RG-MCH：{record['mean_damage_delta_vs_rg_mch']:.6f}",
                "",
                "| seed | collapsed | all-noop差 | 突防差 | 奖励差 | 损伤差 | 成本差 |",
                "| ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in record["paired_runs"]:
            lines.append(
                f"| {row['train_seed']} | {str(row['candidate_collapsed']).lower()} | "
                f"{row['all_noop_delta_vs_baseline']:.6f} | "
                f"{row['high_threat_leak_delta_vs_baseline']:.6f} | "
                f"{row['reward_delta_vs_baseline']:.6f} | "
                f"{row['damage_delta_vs_baseline']:.6f} | "
                f"{row['cost_delta_vs_baseline']:.6f} |"
            )
    return "\n".join(lines) + "\n"


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "sa_rg_mch_stress_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sa_rg_mch_stress_summary.md").write_text(
        _markdown(summary), encoding="utf-8"
    )


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
    mch_reference_rows = _read_csv(args.mch_reference_dir / "runs.csv")
    rg_reference_rows = _read_csv(args.rg_reference_dir / "runs.csv")
    if args.analyze_only:
        summary = _build_summary(
            _read_csv(output_dir / "runs.csv"),
            mch_reference_rows,
            rg_reference_rows,
        )
        _write_summary(output_dir, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    required_files = tuple(args.q_critics) + (args.support_dataset,)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")

    protocol = AirDefenseV1BenchmarkConfig(
        train_seeds=seeds,
        eval_episodes=eval_episodes,
        curve_eval_freq=max(128, timesteps // 2),
        curve_eval_episodes=max(1, min(5, eval_episodes)),
        train_scenarios=scenarios,
        eval_scenarios=scenarios,
        methods=(CANDIDATE,),
        save_models=True,
        create_plot=not args.smoke,
        record_decisions=True,
    )
    training = AirDefenseV1PPOConfig(
        total_timesteps=timesteps,
        n_steps=min(256, timesteps),
        batch_size=64,
        n_epochs=2,
        seed=seeds[0],
        device=args.device,
        verbose=0,
        mch_q_critic_paths=tuple(str(path.resolve()) for path in args.q_critics),
        sa_rg_mch_support_dataset_path=str(args.support_dataset.resolve()),
    )
    result = run_air_defense_v1_benchmark(
        output_dir=output_dir,
        benchmark_config=protocol,
        train_config=training,
        progress_callback=lambda message: print(message, flush=True),
    )
    summary = _build_summary(
        result.run_rows, mch_reference_rows, rg_reference_rows
    )
    _write_summary(output_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
