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
    / "bpce_ppo_mechanism_stress_test"
)
DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "mch_ppo_mechanism_stress_test"
)
BASELINE = "factorized_engagement_ar_ppo_order_012"
CANDIDATE = "bpce_ppo_order_012"
RANDOM_PROBE = "bpce_random_probe_ppo_order_012"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered BPCE-PPO v0 mechanism stress test."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=(8, 9, 10))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=("time_pressure", "heterogeneity_pressure"),
    )
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-baseline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _number(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else 0.0


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(_number(row, key) for row in rows) / len(rows)


def _summary(
    experiment_rows: tuple[dict[str, Any], ...],
    reference_rows: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    rows = tuple(reference_rows) + tuple(experiment_rows)
    on_scenario = [
        row for row in rows if row["train_scenario"] == row["eval_scenario"]
    ]
    lookup = {
        (row["method"], row["train_scenario"], int(row["train_seed"])): row
        for row in on_scenario
        if row.get("train_seed") not in {"", None}
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
    all_noop_noninferior = True
    safety_noninferior = True
    resource_cost_passed = True
    improves_high_threat = False
    boundary_beats_random = True

    for scenario in scenarios:
        seeds = sorted(
            int(row["train_seed"])
            for row in on_scenario
            if row["method"] == CANDIDATE
            and row["train_scenario"] == scenario
        )
        candidates: list[dict[str, Any]] = []
        random_rows: list[dict[str, Any]] = []
        baselines: list[dict[str, Any]] = []
        paired: list[dict[str, Any]] = []
        for seed in seeds:
            candidate = lookup[(CANDIDATE, scenario, seed)]
            random_probe = lookup[(RANDOM_PROBE, scenario, seed)]
            baseline = lookup[(BASELINE, scenario, seed)]
            candidates.append(candidate)
            random_rows.append(random_probe)
            baselines.append(baseline)
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
                    "collapsed": collapsed,
                    "all_noop_delta_vs_baseline": _number(
                        candidate, "all_noop_episode_rate"
                    )
                    - _number(baseline, "all_noop_episode_rate"),
                    "reward_delta_vs_baseline": _number(candidate, "avg_reward")
                    - _number(baseline, "avg_reward"),
                    "damage_delta_vs_baseline": _number(
                        candidate, "avg_total_damage"
                    )
                    - _number(baseline, "avg_total_damage"),
                    "high_threat_delta_vs_baseline": _number(
                        candidate, "high_threat_leak_rate"
                    )
                    - _number(baseline, "high_threat_leak_rate"),
                    "reward_delta_vs_random": _number(candidate, "avg_reward")
                    - _number(random_probe, "avg_reward"),
                    "damage_delta_vs_random": _number(
                        candidate, "avg_total_damage"
                    )
                    - _number(random_probe, "avg_total_damage"),
                }
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
        cost_ratio = _mean(candidates, "avg_resource_cost") / max(
            1e-8, _mean(baselines, "avg_resource_cost")
        )
        random_reward_delta = _mean(candidates, "avg_reward") - _mean(
            random_rows, "avg_reward"
        )
        random_damage_delta = _mean(
            candidates, "avg_total_damage"
        ) - _mean(random_rows, "avg_total_damage")
        noop_count = sum(
            record["all_noop_delta_vs_baseline"] <= 0.0 for record in paired
        )
        scenario_noop = noop_count >= max(1, len(seeds) - 1)
        scenario_safety = reward_delta >= -10.0 and damage_delta <= 0.20
        scenario_cost = cost_ratio <= 1.10
        scenario_boundary = (
            random_reward_delta > 0.0 and random_damage_delta <= 0.0
        )
        all_noop_noninferior &= scenario_noop
        safety_noninferior &= scenario_safety
        resource_cost_passed &= scenario_cost
        improves_high_threat |= high_threat_delta < 0.0
        boundary_beats_random &= scenario_boundary
        scenario_records[scenario] = {
            "paired_runs": paired,
            "all_noop_noninferior_seed_count": noop_count,
            "mean_reward_delta_vs_baseline": reward_delta,
            "mean_damage_delta_vs_baseline": damage_delta,
            "mean_high_threat_leak_delta_vs_baseline": high_threat_delta,
            "resource_cost_ratio_vs_baseline": cost_ratio,
            "mean_reward_delta_vs_random_probe": random_reward_delta,
            "mean_damage_delta_vs_random_probe": random_damage_delta,
            "boundary_probe_gate_passed": scenario_boundary,
        }

    candidate_keys = sorted(
        (
            row["train_scenario"],
            int(row["train_seed"]),
        )
        for row in on_scenario
        if row["method"] == CANDIDATE
    )
    candidate_training = [
        lookup[(CANDIDATE, scenario, seed)]
        for scenario, seed in candidate_keys
    ]
    baseline_training = [
        lookup[(BASELINE, scenario, seed)]
        for scenario, seed in candidate_keys
    ]
    training_time_ratio = _mean(
        candidate_training, "training_seconds"
    ) / max(1e-8, _mean(baseline_training, "training_seconds"))
    diagnostics = {
        key: _mean(candidate_training, key)
        for key in (
            "bpce_probe_cumulative_probe_rollouts",
            "bpce_probe_cumulative_selected_count",
            "bpce_probe_cumulative_accepted_count",
            "bpce_probe_cumulative_acceptance_rate",
            "bpce_probe_cumulative_positive_count",
            "bpce_probe_cumulative_negative_count",
            "bpce_probe_cumulative_mean_abs_delta",
            "bpce_probe_cumulative_mean_sign_agreement",
            "bpce_probe_cumulative_effect_pass_rate",
            "bpce_probe_cumulative_agreement_pass_rate",
            "bpce_probe_cumulative_mean_informative_repeats",
            "bpce_probe_cumulative_mean_opposite_repeats",
            "bpce_probe_cumulative_extra_transitions",
            "bpce_train_cumulative_auxiliary_train_calls",
            "bpce_train_cumulative_mean_auxiliary_loss",
        )
    }
    diagnostics["training_time_ratio_vs_baseline"] = training_time_ratio
    gates = {
        "structural_zero": structural_zero,
        "no_collapsed_candidate_runs": collapsed_count == 0,
        "all_noop_noninferiority": all_noop_noninferior,
        "high_threat_improvement": improves_high_threat,
        "reward_damage_safety": safety_noninferior,
        "resource_cost": resource_cost_passed,
        "boundary_beats_equal_budget_random": boundary_beats_random,
        "training_time_within_2x": training_time_ratio <= 2.0,
    }
    return {
        "schema_version": 1,
        "methods": [BASELINE, RANDOM_PROBE, CANDIDATE],
        "scenarios": scenario_records,
        "collapsed_candidate_run_count": collapsed_count,
        "training_diagnostics": diagnostics,
        "gates": gates,
        "mechanism_gate_passed": all(gates.values()),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# AirDefense v1 BPCE-PPO v0 机制压力实验",
        "",
        "## 总结",
        "",
        f"- 总门控：`{str(summary['mechanism_gate_passed']).lower()}`",
        f"- 候选塌缩场景种子数：{summary['collapsed_candidate_run_count']}",
    ]
    lines.extend(
        f"- `{name}`：`{str(passed).lower()}`"
        for name, passed in summary["gates"].items()
    )
    lines.extend(["", "## 探测与训练诊断", ""])
    lines.extend(
        f"- `{name}`：{value:.6f}"
        for name, value in summary["training_diagnostics"].items()
    )
    for scenario, record in summary["scenarios"].items():
        lines.extend(
            [
                "",
                f"## {scenario}",
                "",
                f"- 奖励差 vs baseline：{record['mean_reward_delta_vs_baseline']:.6f}",
                f"- 损伤差 vs baseline：{record['mean_damage_delta_vs_baseline']:.6f}",
                f"- 高威胁突防差 vs baseline：{record['mean_high_threat_leak_delta_vs_baseline']:.6f}",
                f"- 奖励差 vs random probe：{record['mean_reward_delta_vs_random_probe']:.6f}",
                f"- 损伤差 vs random probe：{record['mean_damage_delta_vs_random_probe']:.6f}",
                "",
                "| seed | collapsed | all-noop差 | 奖励差 | 损伤差 | vs随机奖励 | vs随机损伤 |",
                "| ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in record["paired_runs"]:
            lines.append(
                f"| {row['train_seed']} | {str(row['collapsed']).lower()} | "
                f"{row['all_noop_delta_vs_baseline']:.6f} | "
                f"{row['reward_delta_vs_baseline']:.6f} | "
                f"{row['damage_delta_vs_baseline']:.6f} | "
                f"{row['reward_delta_vs_random']:.6f} | "
                f"{row['damage_delta_vs_random']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "该结果使用冻结种子、场景和门控。随机探测与边界探测使用相同分支预算；不得选择单个优势种子替代总体结论。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "bpce_stress_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "bpce_stress_summary.md").write_text(
        _markdown(summary),
        encoding="utf-8",
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
    reference_rows = _read_csv(args.reference_dir / "runs.csv")
    if args.analyze_only:
        summary = _summary(_read_csv(output_dir / "runs.csv"), reference_rows)
        _write_summary(output_dir, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    methods = (CANDIDATE, RANDOM_PROBE)
    if args.smoke or args.train_baseline:
        methods = (BASELINE, *methods)
    protocol = AirDefenseV1BenchmarkConfig(
        train_seeds=seeds,
        eval_episodes=eval_episodes,
        curve_eval_freq=max(128, timesteps // 2),
        curve_eval_episodes=max(1, min(5, eval_episodes)),
        train_scenarios=scenarios,
        eval_scenarios=scenarios,
        methods=methods,
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
    )
    result = run_air_defense_v1_benchmark(
        output_dir=output_dir,
        benchmark_config=protocol,
        train_config=training,
        progress_callback=lambda message: print(message, flush=True),
    )
    summary = _summary(result.run_rows, reference_rows)
    _write_summary(output_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
