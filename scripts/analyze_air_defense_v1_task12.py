from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task12_task11_frozen_replay"
)
DEFAULT_DIAGNOSTIC = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task12_role_diagnostic_10k_5seeds"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task12_analysis"
)
DEFAULT_TASK10 = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task10_order_screening_30k_3seeds"
)
DEFAULT_TASK7 = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task7_formal_medium_100k_5seeds"
)
DEFAULT_TASK8 = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task8_conflict_free_screening_30k_3seeds"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Task 12 no-op diagnostics.")
    parser.add_argument("--frozen-replay", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--training-diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--screening", type=Path, default=None)
    parser.add_argument("--task10-reference", type=Path, default=DEFAULT_TASK10)
    parser.add_argument("--task7-reference", type=Path, default=DEFAULT_TASK7)
    parser.add_argument("--task8-reference", type=Path, default=DEFAULT_TASK8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _final_rows(
    rows: list[dict[str, str]], *, scenario: str
) -> list[dict[str, str]]:
    selected = [row for row in rows if row["probe_scenario"] == scenario]
    final: dict[str, dict[str, str]] = {}
    for row in selected:
        seed = row["train_seed"]
        if seed not in final or int(row["timesteps"]) > int(final[seed]["timesteps"]):
            final[seed] = row
    return [final[seed] for seed in sorted(final, key=int)]


def _first_stable_checkpoint(
    rows: list[dict[str, str]], train_seed: str
) -> int | None:
    seed_rows = sorted(
        (
            row
            for row in rows
            if row["train_seed"] == train_seed and row["probe_scenario"] == "all"
        ),
        key=lambda row: int(row["timesteps"]),
    )
    labels = [
        _number(row, "deterministic_engagement_rate") >= 0.5
        for row in seed_rows
    ]
    for index, label in enumerate(labels):
        if all(candidate == label for candidate in labels[index:]):
            return int(seed_rows[index]["timesteps"])
    return None


def _frozen_summary(directory: Path) -> dict[str, Any]:
    runs = _read_csv(directory / "runs.csv")
    probes = _read_csv(directory / "probe_diagnostics.csv")
    seed_one_runs = [row for row in runs if row["train_seed"] == "1"]
    deterministic = {
        row["eval_scenario"]: row
        for row in seed_one_runs
        if row["evaluation_mode"] == "deterministic"
    }
    stochastic = {
        row["eval_scenario"]: row
        for row in seed_one_runs
        if row["evaluation_mode"] == "stochastic"
    }
    gaps = {
        scenario: _number(stochastic[scenario], "actionable_engagement_rate")
        - _number(deterministic[scenario], "actionable_engagement_rate")
        for scenario in deterministic
    }
    seed_one_probe = next(
        row
        for row in probes
        if row["train_seed"] == "1"
        and row["evaluation_mode"] == "deterministic"
        and row["probe_scenario"] == "all"
    )
    return {
        "seed_1_deterministic_all_noop_rate": {
            scenario: _number(row, "all_noop_episode_rate")
            for scenario, row in deterministic.items()
        },
        "seed_1_stochastic_all_noop_rate": {
            scenario: _number(row, "all_noop_episode_rate")
            for scenario, row in stochastic.items()
        },
        "seed_1_stochastic_engagement_gap": gaps,
        "seed_1_probe_engage_probability": _number(
            seed_one_probe, "engage_probability_mean"
        ),
        "seed_1_probe_noop_probability": _number(
            seed_one_probe, "noop_probability_mean"
        ),
        "inference": (
            "deterministic_argmax_amplification"
            if min(gaps.values()) >= 0.25
            else "probability_level_noop_collapse"
        ),
    }


def _training_summary(directory: Path) -> dict[str, Any]:
    runs = _read_csv(directory / "runs.csv")
    probes = _read_csv(directory / "probe_dynamics.csv")
    final_probes = _final_rows(probes, scenario="all")
    final_runs = {row["train_seed"]: row for row in runs}
    seed_rows = []
    for probe in final_probes:
        seed = probe["train_seed"]
        run = final_runs[seed]
        collapsed = (
            _number(run, "all_noop_episode_rate") >= 0.98
            or _number(run, "actionable_engagement_rate") < 0.01
        )
        seed_rows.append(
            {
                "train_seed": int(seed),
                "timesteps": int(probe["timesteps"]),
                "collapsed": collapsed,
                "avg_reward": _number(run, "avg_reward"),
                "actionable_engagement_rate": _number(
                    run, "actionable_engagement_rate"
                ),
                "all_noop_episode_rate": _number(
                    run, "all_noop_episode_rate"
                ),
                "probe_engage_probability": _number(
                    probe, "engage_probability_mean"
                ),
                "probe_noop_margin": _number(probe, "noop_margin_mean"),
                "probe_engagement_entropy": _number(
                    probe, "engagement_entropy_mean"
                ),
                "first_stable_branch_timestep": _first_stable_checkpoint(
                    probes, seed
                ),
            }
        )
    return {
        "num_seeds": len(seed_rows),
        "collapsed_seed_count": sum(row["collapsed"] for row in seed_rows),
        "successful_seed_count": sum(not row["collapsed"] for row in seed_rows),
        "seeds": seed_rows,
    }


def _screening_summary(
    directory: Path,
    *,
    task10_reference: Path,
    task7_reference: Path,
    task8_reference: Path,
) -> dict[str, Any]:
    runs = _read_csv(directory / "runs.csv")
    decisions = _read_csv(directory / "decision_summary.csv")
    leak_rows = _read_csv(directory / "leak_attribution_summary.csv")
    gaps = _read_csv(directory / "stochastic_engagement_gaps.csv")
    parameter_record = json.loads(
        (directory / "model_parameter_counts.json").read_text(encoding="utf-8")
    )["models"]
    candidate = "factorized_engagement_ar_ppo_order_012"
    baseline = "role_conditioned_ar_ppo_order_012"
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in runs:
        grouped.setdefault((row["method"], row["eval_scenario"]), []).append(row)
    means: dict[str, Any] = {}
    for (method, scenario), rows in grouped.items():
        means[f"{method}/{scenario}"] = {
            metric: sum(_number(row, metric) for row in rows) / len(rows)
            for metric in (
                "avg_reward",
                "avg_total_damage",
                "avg_resource_cost",
                "high_threat_leak_rate",
                "actionable_engagement_rate",
                "all_noop_episode_rate",
                "avg_decision_time_ms",
            )
        }
    gates: list[dict[str, Any]] = []

    def add_gate(
        name: str,
        value: float | int | bool,
        threshold: str,
        passed: bool,
        category: str,
    ) -> None:
        gates.append(
            {
                "name": name,
                "category": category,
                "value": value,
                "threshold": threshold,
                "passed": bool(passed),
            }
        )

    candidate_runs = [row for row in runs if row["method"] == candidate]
    max_invalid = max(_number(row, "avg_invalid_actions") for row in candidate_runs)
    max_conflict = max(
        _number(row, "assignment_conflict_rate") for row in candidate_runs
    )
    max_overkill = max(_number(row, "overkill_rate") for row in candidate_runs)
    add_gate(
        "structural_zero_violations",
        max(max_invalid, max_conflict, max_overkill),
        "== 0",
        max(max_invalid, max_conflict, max_overkill) == 0.0,
        "structure",
    )
    candidate_parameters = next(
        row for row in parameter_record if row["method"] == candidate
    )
    baseline_parameters = next(
        row for row in parameter_record if row["method"] == baseline
    )
    actor_ratio = float(candidate_parameters["actor_parameters"]) / float(
        baseline_parameters["actor_parameters"]
    )
    add_gate(
        "actor_parameter_ratio",
        actor_ratio,
        "0.90 <= ratio <= 1.10",
        0.90 <= actor_ratio <= 1.10,
        "structure",
    )
    critic_equal = int(candidate_parameters["critic_parameters"]) == int(
        baseline_parameters["critic_parameters"]
    )
    add_gate(
        "critic_parameters_equal",
        critic_equal,
        "true",
        critic_equal,
        "structure",
    )

    collapsed_groups = {
        (row["eval_scenario"], row["train_seed"])
        for row in decisions
        if row["method"] == candidate
        and row["collapsed_unit"].strip().lower() == "true"
    }
    add_gate(
        "candidate_collapsed_scenario_seeds",
        len(collapsed_groups),
        "== 0",
        not collapsed_groups,
        "stability",
    )
    scenario_all_noop_means = {
        scenario: means[f"{candidate}/{scenario}"]["all_noop_episode_rate"]
        for scenario in ("medium", "time_pressure", "heterogeneity_pressure")
    }
    max_all_noop_mean = max(scenario_all_noop_means.values())
    add_gate(
        "max_scenario_mean_all_noop_episode_rate",
        max_all_noop_mean,
        "<= 0.02",
        max_all_noop_mean <= 0.02,
        "stability",
    )

    def weighted_unassigned(method: str) -> float:
        selected = [
            row
            for row in leak_rows
            if row["method"] == method
            and row["eval_scenario"] == "heterogeneity_pressure"
            and row["attribution"] == "unassigned"
        ]
        total_unassigned = sum(int(row["count"]) for row in selected)
        total_leaks = sum(int(row["total_high_threat_leaks"]) for row in selected)
        return total_unassigned / total_leaks if total_leaks else 0.0

    unassigned_reduction = weighted_unassigned(baseline) - weighted_unassigned(
        candidate
    )
    add_gate(
        "heterogeneity_unassigned_leak_reduction",
        unassigned_reduction,
        ">= 0.15",
        unassigned_reduction >= 0.15,
        "mission",
    )
    candidate_hetero = grouped[(candidate, "heterogeneity_pressure")]
    baseline_hetero = grouped[(baseline, "heterogeneity_pressure")]
    baseline_by_seed = {row["train_seed"]: row for row in baseline_hetero}
    high_leak_reductions = [
        _number(baseline_by_seed[row["train_seed"]], "high_threat_leak_rate")
        - _number(row, "high_threat_leak_rate")
        for row in candidate_hetero
    ]
    high_leak_mean_reduction = sum(high_leak_reductions) / len(
        high_leak_reductions
    )
    high_leak_improved_seeds = sum(value > 0.0 for value in high_leak_reductions)
    add_gate(
        "heterogeneity_high_threat_leak_mean_reduction",
        high_leak_mean_reduction,
        ">= 0.02 and >= 2/3 seeds",
        high_leak_mean_reduction >= 0.02 and high_leak_improved_seeds >= 2,
        "mission",
    )

    paired_gates = (
        ("medium_reward_delta", "medium", "avg_reward", -5.0, "min"),
        ("medium_damage_delta", "medium", "avg_total_damage", 0.10, "max"),
        (
            "time_pressure_resource_cost_delta",
            "time_pressure",
            "avg_resource_cost",
            0.50,
            "max",
        ),
        (
            "heterogeneity_damage_delta",
            "heterogeneity_pressure",
            "avg_total_damage",
            0.10,
            "max",
        ),
    )
    for name, scenario, metric, threshold, direction in paired_gates:
        delta = means[f"{candidate}/{scenario}"][metric] - means[
            f"{baseline}/{scenario}"
        ][metric]
        passed = delta >= threshold if direction == "min" else delta <= threshold
        comparator = ">=" if direction == "min" else "<="
        add_gate(name, delta, f"{comparator} {threshold}", passed, "noninferiority")

    candidate_gaps = [
        abs(_number(row, "stochastic_engagement_gap"))
        for row in gaps
        if row["method"] == candidate
    ]
    max_stochastic_gap = max(candidate_gaps)
    add_gate(
        "max_absolute_stochastic_engagement_gap",
        max_stochastic_gap,
        "<= 0.05",
        max_stochastic_gap <= 0.05,
        "calibration",
    )

    task10_runs = [
        row
        for row in _read_csv(task10_reference / "runs.csv")
        if row["method"] == "autoregressive_ppo_order_012"
        and row["eval_scenario"]
        in {"medium", "time_pressure", "heterogeneity_pressure"}
    ]
    candidate_latency = sum(
        _number(row, "avg_decision_time_ms") for row in candidate_runs
    ) / len(candidate_runs)
    task10_latency = sum(
        _number(row, "avg_decision_time_ms") for row in task10_runs
    ) / len(task10_runs)
    latency_increase = candidate_latency / task10_latency - 1.0
    add_gate(
        "decision_latency_increase_vs_task10",
        latency_increase,
        "<= 0.25",
        latency_increase <= 0.25,
        "efficiency",
    )

    task7_runs = [
        row
        for row in _read_csv(task7_reference / "runs.csv")
        if row["method"] == "maskable_ppo"
    ]
    task7_grouped: dict[str, list[dict[str, str]]] = {}
    for row in task7_runs:
        task7_grouped.setdefault(row["eval_scenario"], []).append(row)

    def reference_mean(
        rows_by_scenario: dict[str, list[dict[str, str]]],
        scenario: str,
        metric: str,
    ) -> float:
        values = rows_by_scenario[scenario]
        return sum(_number(row, metric) for row in values) / len(values)

    external_gates = (
        ("external_medium_reward", "medium", "avg_reward", -5.0, "min"),
        (
            "external_time_pressure_reward",
            "time_pressure",
            "avg_reward",
            -5.0,
            "min",
        ),
        (
            "external_medium_damage",
            "medium",
            "avg_total_damage",
            0.10,
            "max",
        ),
        (
            "external_heterogeneity_damage",
            "heterogeneity_pressure",
            "avg_total_damage",
            0.10,
            "max",
        ),
        (
            "external_time_pressure_cost",
            "time_pressure",
            "avg_resource_cost",
            0.50,
            "max",
        ),
    )
    for name, scenario, metric, threshold, direction in external_gates:
        delta = means[f"{candidate}/{scenario}"][metric] - reference_mean(
            task7_grouped, scenario, metric
        )
        passed = delta >= threshold if direction == "min" else delta <= threshold
        comparator = ">=" if direction == "min" else "<="
        add_gate(name, delta, f"{comparator} {threshold}", passed, "external")

    task8_runs = [
        row
        for row in _read_csv(task8_reference / "runs.csv")
        if row["method"] == "conflict_free_maskable_ppo"
        and row["eval_scenario"] == "time_pressure"
    ]
    discrete_time_cost = sum(
        _number(row, "avg_resource_cost") for row in task8_runs
    ) / len(task8_runs)
    candidate_time_cost = means[f"{candidate}/time_pressure"]["avg_resource_cost"]
    add_gate(
        "time_pressure_cost_below_discrete_136",
        candidate_time_cost - discrete_time_cost,
        "< 0",
        candidate_time_cost < discrete_time_cost,
        "external",
    )
    return {
        "means": means,
        "scenario_all_noop_means": scenario_all_noop_means,
        "collapsed_scenario_seeds": sorted(
            [list(item) for item in collapsed_groups]
        ),
        "heterogeneity_unassigned_rates": {
            "baseline": weighted_unassigned(baseline),
            "candidate": weighted_unassigned(candidate),
        },
        "heterogeneity_high_leak_improved_seeds": high_leak_improved_seeds,
        "gates": gates,
        "passed_gate_count": sum(row["passed"] for row in gates),
        "failed_gate_count": sum(not row["passed"] for row in gates),
        "screening_passed": all(row["passed"] for row in gates),
        "run_100k": all(row["passed"] for row in gates),
    }


def _markdown(report: dict[str, Any]) -> str:
    frozen = report["frozen_replay"]
    training = report["training_diagnostic"]
    lines = [
        "# Task 12 no-op 稳定性诊断摘要",
        "",
        "## 冻结模型回放",
        "",
        f"- 机制判定：`{frozen['inference']}`",
        f"- 种子 1 固定探针平均交战概率：{frozen['seed_1_probe_engage_probability']:.4f}",
        f"- 种子 1 固定探针平均 no-op 概率：{frozen['seed_1_probe_noop_probability']:.4f}",
        "",
        "| 场景 | deterministic all-noop | stochastic all-noop | 交战率差 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for scenario, gap in frozen["seed_1_stochastic_engagement_gap"].items():
        lines.append(
            f"| {scenario} | "
            f"{frozen['seed_1_deterministic_all_noop_rate'][scenario]:.3f} | "
            f"{frozen['seed_1_stochastic_all_noop_rate'][scenario]:.3f} | "
            f"{gap:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 10k 训练分叉",
            "",
            f"- 成功种子：{training['successful_seed_count']}",
            f"- 塌缩种子：{training['collapsed_seed_count']}",
            "",
            "| seed | collapsed | reward | actionable engage | all-noop | probe p(engage) | no-op margin | stable step |",
            "| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if "screening" in report:
        screening = report["screening"]
        lines.extend(
            [
                "## 30k 配对筛选",
                "",
                f"- 通过门槛：{screening['passed_gate_count']}",
                f"- 失败门槛：{screening['failed_gate_count']}",
                f"- 是否运行 100k：{str(screening['run_100k']).lower()}",
                "",
                "| 类别 | 门槛 | 数值 | 要求 | 通过 |",
                "| --- | --- | ---: | --- | :---: |",
            ]
        )
        for gate in screening["gates"]:
            value = gate["value"]
            rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
            lines.append(
                f"| {gate['category']} | {gate['name']} | {rendered} | "
                f"{gate['threshold']} | {str(gate['passed']).lower()} |"
            )
        lines.append("")
    for row in training["seeds"]:
        lines.append(
            f"| {row['train_seed']} | {str(row['collapsed']).lower()} | "
            f"{row['avg_reward']:.2f} | {row['actionable_engagement_rate']:.3f} | "
            f"{row['all_noop_episode_rate']:.3f} | "
            f"{row['probe_engage_probability']:.3f} | "
            f"{row['probe_noop_margin']:.3f} | "
            f"{row['first_stable_branch_timestep']} |"
        )
    lines.extend(
        [
            "",
            "## 当前结论",
            "",
            "普通 categorical 的总交战概率被多个目标分摊，deterministic argmax 会优先选择单个 no-op。训练随后进一步分叉：成功种子的 no-op margin 转负，失败种子转正并形成 all-noop。因子化候选必须使用先判定交战、再选择目标的分层确定性规则。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "schema_version": 1,
        "frozen_replay": _frozen_summary(args.frozen_replay),
        "training_diagnostic": _training_summary(args.training_diagnostic),
    }
    if args.screening is not None:
        report["screening"] = _screening_summary(
            args.screening,
            task10_reference=args.task10_reference,
            task7_reference=args.task7_reference,
            task8_reference=args.task8_reference,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task12_diagnostic_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "task12_diagnostic_summary.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
