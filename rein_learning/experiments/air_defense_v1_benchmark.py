from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.stats import t as student_t
from stable_baselines3.common.callbacks import BaseCallback

from ..baselines import (
    GreedyDamageReductionPolicy,
    HighestThreatJointPolicy,
    NearestTargetJointPolicy,
    RandomLegalJointPolicy,
    TimeToImpactJointPolicy,
    evaluate_air_defense_v1_policy,
)
from ..envs import AirDefenseResourceAssignmentEnvV1, AirDefenseV1EnvConfig
from ..trainers.air_defense_v1_ppo import (
    AirDefenseV1PPOConfig,
    evaluate_air_defense_v1_model,
    train_maskable_ppo,
    train_ppo,
)


MetricRow = dict[str, float | int | str]
PolicyFactory = Callable[[int], object]
ProgressCallback = Callable[[str], None]

METRIC_NAMES = (
    "avg_reward",
    "avg_steps",
    "success_rate",
    "intercept_rate",
    "leak_rate",
    "avg_total_damage",
    "avg_ammo_used",
    "avg_shots",
    "hit_rate_per_shot",
    "avg_invalid_actions",
)

CURVE_METRIC_NAMES = (
    "avg_reward",
    "success_rate",
    "intercept_rate",
    "avg_total_damage",
    "avg_invalid_actions",
)

METRIC_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "avg_steps": (0.0, None),
    "success_rate": (0.0, 1.0),
    "intercept_rate": (0.0, 1.0),
    "leak_rate": (0.0, 1.0),
    "avg_total_damage": (0.0, None),
    "avg_ammo_used": (0.0, None),
    "avg_shots": (0.0, None),
    "hit_rate_per_shot": (0.0, 1.0),
    "avg_invalid_actions": (0.0, None),
}

RULE_POLICY_FACTORIES: dict[str, PolicyFactory] = {
    "random_joint": lambda seed: RandomLegalJointPolicy(seed=seed),
    "nearest_joint": lambda seed: NearestTargetJointPolicy(),
    "highest_threat": lambda seed: HighestThreatJointPolicy(),
    "time_to_impact": lambda seed: TimeToImpactJointPolicy(),
    "greedy_damage": lambda seed: GreedyDamageReductionPolicy(),
}


@dataclass(frozen=True)
class AirDefenseV1BenchmarkConfig:
    """Protocol shared by rule and learning baselines."""

    train_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    eval_episodes: int = 50
    eval_seed: int = 200
    curve_eval_freq: int = 5_000
    curve_eval_episodes: int = 10
    curve_eval_seed: int = 10_000
    confidence_level: float = 0.95
    include_learning: bool = True
    save_models: bool = True
    create_plot: bool = True

    def __post_init__(self) -> None:
        if not self.train_seeds:
            raise ValueError("train_seeds must contain at least one seed")
        if len(set(self.train_seeds)) != len(self.train_seeds):
            raise ValueError("train_seeds must be unique")
        if self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive")
        if self.curve_eval_freq <= 0:
            raise ValueError("curve_eval_freq must be positive")
        if self.curve_eval_episodes <= 0:
            raise ValueError("curve_eval_episodes must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1")


@dataclass(frozen=True)
class BenchmarkArtifacts:
    output_dir: Path
    config: Path
    runs: Path
    summary: Path
    learning_curves: Path
    learning_curve_summary: Path
    curve_figure_base: Path
    models_dir: Path
    tensorboard_dir: Path


@dataclass(frozen=True)
class BenchmarkResult:
    artifacts: BenchmarkArtifacts
    run_rows: tuple[MetricRow, ...]
    summary_rows: tuple[MetricRow, ...]
    curve_rows: tuple[MetricRow, ...]
    curve_summary_rows: tuple[MetricRow, ...]
    figure_paths: tuple[Path, ...]


class EvaluationCurveCallback(BaseCallback):
    """Evaluate a model on held-out seeds at fixed training intervals."""

    def __init__(
        self,
        *,
        method: str,
        train_seed: int,
        eval_freq: int,
        eval_episodes: int,
        eval_seed: int,
        env_config: AirDefenseV1EnvConfig | None,
        use_action_masks: bool,
    ) -> None:
        super().__init__(verbose=0)
        if eval_freq <= 0:
            raise ValueError("eval_freq must be positive")
        if eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive")
        self.method = method
        self.train_seed = train_seed
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.env_config = env_config
        self.use_action_masks = use_action_masks
        self.rows: list[MetricRow] = []
        self._next_eval_timestep = eval_freq
        self._last_eval_timestep = -1
        self._started_at = 0.0

    def _on_training_start(self) -> None:
        self._started_at = perf_counter()
        self._record_evaluation(0)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval_timestep:
            self._record_evaluation(self.num_timesteps)
            while self._next_eval_timestep <= self.num_timesteps:
                self._next_eval_timestep += self.eval_freq
        return True

    def _on_training_end(self) -> None:
        if self._last_eval_timestep != self.num_timesteps:
            self._record_evaluation(self.num_timesteps)

    def _record_evaluation(self, timesteps: int) -> None:
        metrics = evaluate_air_defense_v1_model(
            self.model,
            env_config=self.env_config,
            episodes=self.eval_episodes,
            seed=self.eval_seed,
            use_action_masks=self.use_action_masks,
        )
        self.rows.append(
            {
                "method": self.method,
                "train_seed": self.train_seed,
                "timesteps": timesteps,
                "evaluation_seed": self.eval_seed,
                "elapsed_seconds": perf_counter() - self._started_at,
                **metrics,
            }
        )
        self._last_eval_timestep = timesteps


def create_artifacts(output_dir: str | Path) -> BenchmarkArtifacts:
    output_path = Path(output_dir)
    return BenchmarkArtifacts(
        output_dir=output_path,
        config=output_path / "experiment_config.json",
        runs=output_path / "runs.csv",
        summary=output_path / "summary.csv",
        learning_curves=output_path / "learning_curves.csv",
        learning_curve_summary=output_path / "learning_curve_summary.csv",
        curve_figure_base=output_path / "learning_curves",
        models_dir=output_path / "models",
        tensorboard_dir=output_path / "tensorboard",
    )


def run_air_defense_v1_benchmark(
    *,
    output_dir: str | Path,
    benchmark_config: AirDefenseV1BenchmarkConfig | None = None,
    train_config: AirDefenseV1PPOConfig | None = None,
    env_config: AirDefenseV1EnvConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BenchmarkResult:
    protocol = benchmark_config or AirDefenseV1BenchmarkConfig()
    environment = env_config or AirDefenseV1EnvConfig()
    training = train_config or AirDefenseV1PPOConfig()
    artifacts = create_artifacts(output_dir)
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    config_record = _build_config_record(
        status="running",
        started_at=started_at,
        completed_at=None,
        protocol=protocol,
        training=training,
        environment=environment,
        artifacts=artifacts,
    )
    _write_json(artifacts.config, config_record)

    run_rows: list[MetricRow] = []
    curve_rows: list[MetricRow] = []
    for run_index, train_seed in enumerate(protocol.train_seeds):
        evaluation_seed = protocol.eval_seed + run_index * protocol.eval_episodes
        curve_seed = (
            protocol.curve_eval_seed
            + run_index * protocol.curve_eval_episodes
        )
        _report_progress(
            progress_callback,
            (
                f"run {run_index + 1}/{len(protocol.train_seeds)}: "
                f"rules, evaluation_seed={evaluation_seed}"
            ),
        )
        run_rows.extend(
            _evaluate_rule_methods(
                environment=environment,
                run_index=run_index,
                evaluation_seed=evaluation_seed,
                episodes=protocol.eval_episodes,
            )
        )
        if protocol.include_learning:
            learning_rows, learning_curves = _train_learning_methods(
                environment=environment,
                base_training=training,
                artifacts=artifacts,
                protocol=protocol,
                run_index=run_index,
                train_seed=train_seed,
                evaluation_seed=evaluation_seed,
                curve_seed=curve_seed,
                progress_callback=progress_callback,
            )
            run_rows.extend(learning_rows)
            curve_rows.extend(learning_curves)

    summary_rows = summarize_rows(
        run_rows,
        group_keys=("method", "method_type"),
        metrics=METRIC_NAMES,
        confidence_level=protocol.confidence_level,
    )
    curve_summary_rows = summarize_rows(
        curve_rows,
        group_keys=("method", "timesteps"),
        metrics=CURVE_METRIC_NAMES,
        confidence_level=protocol.confidence_level,
    )

    _write_csv(artifacts.runs, run_rows, RUN_FIELDNAMES)
    _write_csv(artifacts.summary, summary_rows, SUMMARY_FIELDNAMES)
    _write_csv(artifacts.learning_curves, curve_rows, CURVE_FIELDNAMES)
    _write_csv(
        artifacts.learning_curve_summary,
        curve_summary_rows,
        CURVE_SUMMARY_FIELDNAMES,
    )

    figure_paths: tuple[Path, ...] = ()
    if protocol.create_plot and curve_summary_rows:
        figure_paths = tuple(
            plot_learning_curves(
                curve_summary_rows,
                artifacts.curve_figure_base,
                confidence_level=protocol.confidence_level,
            )
        )

    completed_at = datetime.now(timezone.utc)
    config_record = _build_config_record(
        status="completed",
        started_at=started_at,
        completed_at=completed_at,
        protocol=protocol,
        training=training,
        environment=environment,
        artifacts=artifacts,
    )
    config_record["figure_contract"] = {
        "core_conclusion": (
            "Compare PPO and Maskable PPO learning dynamics and uncertainty "
            "under the same held-out air-defense scenarios."
        ),
        "archetype": "asymmetric quantitative grid",
        "backend": "Python/matplotlib",
        "statistics": (
            f"mean and {protocol.confidence_level:.0%} Student-t confidence "
            "interval across experiment seeds"
        ),
        "source_data": artifacts.learning_curve_summary.name,
    }
    config_record["result_counts"] = {
        "run_rows": len(run_rows),
        "curve_rows": len(curve_rows),
    }
    _write_json(artifacts.config, config_record)

    return BenchmarkResult(
        artifacts=artifacts,
        run_rows=tuple(run_rows),
        summary_rows=tuple(summary_rows),
        curve_rows=tuple(curve_rows),
        curve_summary_rows=tuple(curve_summary_rows),
        figure_paths=figure_paths,
    )


def summarize_rows(
    rows: Sequence[MetricRow],
    *,
    group_keys: Sequence[str],
    metrics: Sequence[str],
    confidence_level: float = 0.95,
) -> list[MetricRow]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    grouped: dict[tuple[object, ...], list[MetricRow]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows: list[MetricRow] = []
    for key, grouped_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        group_values = dict(zip(group_keys, key))
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in grouped_rows],
                dtype=np.float64,
            )
            mean, std, sem, ci_low, ci_high = _mean_confidence_interval(
                values,
                confidence_level,
            )
            lower_bound, upper_bound = METRIC_BOUNDS.get(metric, (None, None))
            if lower_bound is not None:
                ci_low = max(lower_bound, ci_low)
            if upper_bound is not None:
                ci_high = min(upper_bound, ci_high)
            summary_rows.append(
                {
                    **group_values,
                    "metric": metric,
                    "n_runs": len(values),
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "confidence_level": confidence_level,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return summary_rows


def plot_learning_curves(
    summary_rows: Sequence[MetricRow],
    output_base: str | Path,
    *,
    confidence_level: float,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    figure = plt.figure(figsize=(7.2, 4.8), facecolor="white")
    grid = figure.add_gridspec(
        3,
        5,
        width_ratios=(1.15, 1.15, 1.15, 1.0, 1.0),
        hspace=0.55,
        wspace=0.85,
    )
    axes = {
        "avg_reward": figure.add_subplot(grid[:, :3]),
        "success_rate": figure.add_subplot(grid[0, 3:]),
        "avg_total_damage": figure.add_subplot(grid[1, 3:]),
        "avg_invalid_actions": figure.add_subplot(grid[2, 3:]),
    }
    panel_labels = {
        "avg_reward": "a",
        "success_rate": "b",
        "avg_total_damage": "c",
        "avg_invalid_actions": "d",
    }
    y_labels = {
        "avg_reward": "Average episode reward",
        "success_rate": "Success rate",
        "avg_total_damage": "Average total damage",
        "avg_invalid_actions": "Invalid actions / episode",
    }
    method_colors = {
        "ppo": "#767676",
        "maskable_ppo": "#0F4D92",
    }
    method_labels = {
        "ppo": "PPO",
        "maskable_ppo": "Maskable PPO",
    }

    for metric, axis in axes.items():
        metric_rows = [row for row in summary_rows if row["metric"] == metric]
        methods = sorted({str(row["method"]) for row in metric_rows})
        for method in methods:
            method_rows = sorted(
                (row for row in metric_rows if row["method"] == method),
                key=lambda row: int(row["timesteps"]),
            )
            x = np.asarray([int(row["timesteps"]) for row in method_rows])
            mean = np.asarray([float(row["mean"]) for row in method_rows])
            ci_low = np.asarray([float(row["ci_low"]) for row in method_rows])
            ci_high = np.asarray([float(row["ci_high"]) for row in method_rows])
            if metric == "success_rate":
                ci_low = np.clip(ci_low, 0.0, 1.0)
                ci_high = np.clip(ci_high, 0.0, 1.0)
            elif metric in {"avg_total_damage", "avg_invalid_actions"}:
                ci_low = np.maximum(ci_low, 0.0)
            color = method_colors.get(method, "#4D4D4D")
            axis.plot(
                x,
                mean,
                color=color,
                linewidth=1.7,
                label=method_labels.get(method, method),
            )
            axis.fill_between(x, ci_low, ci_high, color=color, alpha=0.16)

        axis.set_ylabel(y_labels[metric])
        axis.set_xlabel("Training timesteps")
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.65)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
        axis.text(
            -0.16,
            1.04,
            panel_labels[metric],
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        if metric == "success_rate":
            axis.set_ylim(0.0, 1.0)
        elif metric in {"avg_total_damage", "avg_invalid_actions"}:
            axis.set_ylim(bottom=0.0)

    handles, labels = axes["avg_reward"].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.01),
        ncol=max(1, len(labels)),
    )
    n_runs = max(int(row["n_runs"]) for row in summary_rows)
    figure.text(
        0.995,
        0.005,
        f"Mean and {confidence_level:.0%} CI across {n_runs} seeds",
        ha="right",
        va="bottom",
        fontsize=6,
        color="#4D4D4D",
    )
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.91)

    output_path = Path(output_base)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for extension, dpi in (("svg", 300), ("pdf", 300), ("png", 300)):
        path = output_path.with_suffix(f".{extension}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(path)
    plt.close(figure)
    return saved_paths


def _evaluate_rule_methods(
    *,
    environment: AirDefenseV1EnvConfig,
    run_index: int,
    evaluation_seed: int,
    episodes: int,
) -> list[MetricRow]:
    rows: list[MetricRow] = []
    for method, policy_factory in RULE_POLICY_FACTORIES.items():
        metrics = evaluate_air_defense_v1_policy(
            env_factory=lambda: AirDefenseResourceAssignmentEnvV1(config=environment),
            policy_factory=policy_factory,
            episodes=episodes,
            seed=evaluation_seed,
        )
        rows.append(
            {
                "method": method,
                "method_type": "rule",
                "run_index": run_index,
                "train_seed": "",
                "evaluation_seed": evaluation_seed,
                "training_timesteps": 0,
                "requested_timesteps": 0,
                "training_seconds": 0.0,
                "model_path": "",
                **metrics,
            }
        )
    return rows


def _train_learning_methods(
    *,
    environment: AirDefenseV1EnvConfig,
    base_training: AirDefenseV1PPOConfig,
    artifacts: BenchmarkArtifacts,
    protocol: AirDefenseV1BenchmarkConfig,
    run_index: int,
    train_seed: int,
    evaluation_seed: int,
    curve_seed: int,
    progress_callback: ProgressCallback | None,
) -> tuple[list[MetricRow], list[MetricRow]]:
    method_specs = (
        ("ppo", train_ppo, False),
        ("maskable_ppo", train_maskable_ppo, True),
    )
    rows: list[MetricRow] = []
    curves: list[MetricRow] = []
    training = replace(
        base_training,
        seed=train_seed,
        tensorboard_log=str(artifacts.tensorboard_dir),
    )
    for method, train_fn, use_action_masks in method_specs:
        _report_progress(
            progress_callback,
            (
                f"run {run_index + 1}/{len(protocol.train_seeds)}: "
                f"train {method}, seed={train_seed}"
            ),
        )
        callback = EvaluationCurveCallback(
            method=method,
            train_seed=train_seed,
            eval_freq=protocol.curve_eval_freq,
            eval_episodes=protocol.curve_eval_episodes,
            eval_seed=curve_seed,
            env_config=environment,
            use_action_masks=use_action_masks,
        )
        save_path = None
        if protocol.save_models:
            save_path = artifacts.models_dir / f"{method}_seed{train_seed}.zip"
        started = perf_counter()
        model = train_fn(
            env_config=environment,
            train_config=training,
            save_path=save_path,
            callback=callback,
            tb_log_name=f"{method}_seed{train_seed}",
        )
        training_seconds = perf_counter() - started
        metrics = evaluate_air_defense_v1_model(
            model,
            env_config=environment,
            episodes=protocol.eval_episodes,
            seed=evaluation_seed,
            use_action_masks=use_action_masks,
        )
        rows.append(
            {
                "method": method,
                "method_type": "learning",
                "run_index": run_index,
                "train_seed": train_seed,
                "evaluation_seed": evaluation_seed,
                "requested_timesteps": training.total_timesteps,
                "training_timesteps": int(model.num_timesteps),
                "training_seconds": training_seconds,
                "model_path": str(save_path) if save_path is not None else "",
                **metrics,
            }
        )
        for curve_row in callback.rows:
            curves.append({"run_index": run_index, **curve_row})
    return rows, curves


def _mean_confidence_interval(
    values: np.ndarray,
    confidence_level: float,
) -> tuple[float, float, float, float, float]:
    if values.size == 0:
        raise ValueError("values must not be empty")
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, 0.0, 0.0, mean, mean
    std = float(np.std(values, ddof=1))
    sem = std / float(np.sqrt(values.size))
    critical = float(
        student_t.ppf((1.0 + confidence_level) / 2.0, df=values.size - 1)
    )
    margin = critical * sem
    return mean, std, sem, mean - margin, mean + margin


def _build_config_record(
    *,
    status: str,
    started_at: datetime,
    completed_at: datetime | None,
    protocol: AirDefenseV1BenchmarkConfig,
    training: AirDefenseV1PPOConfig,
    environment: AirDefenseV1EnvConfig,
    artifacts: BenchmarkArtifacts,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat() if completed_at else None,
        "benchmark": asdict(protocol),
        "training": asdict(training),
        "environment": asdict(environment),
        "methods": {
            "rule": list(RULE_POLICY_FACTORIES),
            "learning": ["ppo", "maskable_ppo"] if protocol.include_learning else [],
        },
        "evaluation_protocol": {
            "paired_scenario_blocks": True,
            "final_evaluation_seed_formula": (
                "eval_seed + run_index * eval_episodes"
            ),
            "curve_evaluation_seed_formula": (
                "curve_eval_seed + run_index * curve_eval_episodes"
            ),
            "confidence_interval": "two-sided Student-t interval across runs",
            "training_seed_source": (
                "benchmark.train_seeds; the training template seed is overwritten "
                "for each run"
            ),
            "timestep_accounting": (
                "requested_timesteps is the configured budget; training_timesteps "
                "is the actual SB3 rollout count"
            ),
        },
        "runtime": _runtime_metadata(),
        "artifacts": {
            field: str(path)
            for field, path in asdict(artifacts).items()
        },
    }


def _runtime_metadata() -> dict[str, Any]:
    packages = {}
    for package_name in (
        "gymnasium",
        "numpy",
        "scipy",
        "stable-baselines3",
        "sb3-contrib",
        "torch",
    ):
        try:
            packages[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            packages[package_name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "command": sys.argv,
    }


def _write_csv(
    path: Path,
    rows: Iterable[MetricRow],
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


def _report_progress(
    progress_callback: ProgressCallback | None,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(message)


RUN_FIELDNAMES = (
    "method",
    "method_type",
    "run_index",
    "train_seed",
    "evaluation_seed",
    "requested_timesteps",
    "training_timesteps",
    "training_seconds",
    "model_path",
    "episodes",
    "avg_reward",
    "std_reward",
    "avg_steps",
    "success_rate",
    "intercept_rate",
    "leak_rate",
    "avg_total_damage",
    "avg_ammo_used",
    "avg_shots",
    "hit_rate_per_shot",
    "avg_invalid_actions",
)

SUMMARY_FIELDNAMES = (
    "method",
    "method_type",
    "metric",
    "n_runs",
    "mean",
    "std",
    "sem",
    "confidence_level",
    "ci_low",
    "ci_high",
)

CURVE_FIELDNAMES = (
    "method",
    "run_index",
    "train_seed",
    "timesteps",
    "evaluation_seed",
    "elapsed_seconds",
    "episodes",
    "avg_reward",
    "std_reward",
    "avg_steps",
    "success_rate",
    "intercept_rate",
    "leak_rate",
    "avg_total_damage",
    "avg_ammo_used",
    "avg_shots",
    "hit_rate_per_shot",
    "avg_invalid_actions",
)

CURVE_SUMMARY_FIELDNAMES = (
    "method",
    "timesteps",
    "metric",
    "n_runs",
    "mean",
    "std",
    "sem",
    "confidence_level",
    "ci_low",
    "ci_high",
)
