from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib import metadata
from itertools import combinations
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.stats import t as student_t
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from ..baselines import (
    GreedyDamageReductionPolicy,
    HighestThreatJointPolicy,
    HungarianDamageReductionPolicy,
    NearestTargetJointPolicy,
    RandomLegalJointPolicy,
    TimeToImpactJointPolicy,
    evaluate_air_defense_v1_policy,
)
from ..common import (
    DEFAULT_HIGH_THREAT_THRESHOLD,
    PPOTrainingDiagnosticsCallback,
    PolicyProbeCorpus,
    aggregate_decision_rows,
    aggregate_leak_attributions,
)
from ..envs import (
    AIR_DEFENSE_V1_DEFAULT_SCENARIO,
    AirDefenseResourceAssignmentEnvV1,
    AirDefenseV1EnvConfig,
    ConflictFreeJointActionWrapper,
    get_air_defense_v1_scenario_profile,
)
from ..models import (
    AirDefenseV1ObservationLayout,
    FactorizedEngagementActionHeadConfig,
    RoleConditionedActionHeadConfig,
)
from ..algorithms.policy_gradient.role_conditioned_autoregressive_ppo import (
    policy_parameter_counts,
)
from ..trainers.air_defense_v1_ppo import (
    AirDefenseV1PPOConfig,
    evaluate_air_defense_v1_model,
    train_autoregressive_maskable_ppo,
    train_bpce_ppo,
    train_conflict_free_maskable_ppo,
    train_factorized_engagement_autoregressive_ppo,
    train_maskable_ppo,
    train_mch_ppo,
    train_rg_mch_ppo,
    train_sa_rg_mch_ppo,
    train_ppo,
    train_role_conditioned_autoregressive_ppo,
)


MetricRow = dict[str, Any]
PolicyFactory = Callable[[int], object]
ProgressCallback = Callable[[str], None]
EvaluationBlock = tuple[str, AirDefenseV1EnvConfig, int]

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
    "avg_decision_time_ms",
    "high_threat_leak_rate",
    "avg_zone_weighted_damage",
    "assignment_conflict_rate",
    "overkill_rate",
    "damage_reduction_per_ammo",
    "avg_resource_cost",
    "engagement_rate",
    "actionable_engagement_rate",
    "all_noop_episode_rate",
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
    "avg_decision_time_ms": (0.0, None),
    "high_threat_leak_rate": (0.0, 1.0),
    "avg_zone_weighted_damage": (0.0, None),
    "assignment_conflict_rate": (0.0, 1.0),
    "overkill_rate": (0.0, 1.0),
    "damage_reduction_per_ammo": (0.0, None),
    "avg_resource_cost": (0.0, None),
    "engagement_rate": (0.0, 1.0),
    "actionable_engagement_rate": (0.0, 1.0),
    "all_noop_episode_rate": (0.0, 1.0),
}

RULE_POLICY_FACTORIES: dict[str, PolicyFactory] = {
    "random_joint": lambda seed: RandomLegalJointPolicy(seed=seed),
    "nearest_joint": lambda seed: NearestTargetJointPolicy(),
    "highest_threat": lambda seed: HighestThreatJointPolicy(),
    "time_to_impact": lambda seed: TimeToImpactJointPolicy(),
    "greedy_damage": lambda seed: GreedyDamageReductionPolicy(),
    "hungarian_damage": lambda seed: HungarianDamageReductionPolicy(),
}

DEFAULT_LEARNING_METHODS = (
    "ppo",
    "maskable_ppo",
    "conflict_free_maskable_ppo",
    "autoregressive_maskable_ppo",
)
AUTOREGRESSIVE_ORDER_METHODS: dict[str, tuple[int, ...]] = {
    "autoregressive_ppo_order_012": (0, 1, 2),
    "autoregressive_ppo_order_120": (1, 2, 0),
    "autoregressive_ppo_order_201": (2, 0, 1),
}
ROLE_CONDITIONED_ORDER_METHODS: dict[str, tuple[int, ...]] = {
    "role_conditioned_ar_ppo_order_012": (0, 1, 2),
    "role_conditioned_ar_ppo_order_120": (1, 2, 0),
    "role_conditioned_ar_ppo_order_201": (2, 0, 1),
}
FACTORIZED_ENGAGEMENT_METHODS: dict[str, tuple[int, ...]] = {
    "factorized_engagement_ar_ppo_order_012": (0, 1, 2),
}
MCH_PPO_METHODS: dict[str, tuple[int, ...]] = {
    "mch_ppo_order_012": (0, 1, 2),
}
RG_MCH_PPO_METHODS: dict[str, tuple[int, ...]] = {
    "rg_mch_ppo_order_012": (0, 1, 2),
}
SA_RG_MCH_PPO_METHODS: dict[str, tuple[int, ...]] = {
    "sa_rg_mch_ppo_order_012": (0, 1, 2),
}
BPCE_PPO_METHODS: dict[str, tuple[int, ...]] = {
    "bpce_ppo_order_012": (0, 1, 2),
}
BPCE_RANDOM_PROBE_PPO_METHODS: dict[str, tuple[int, ...]] = {
    "bpce_random_probe_ppo_order_012": (0, 1, 2),
}
LEARNING_METHODS = (
    DEFAULT_LEARNING_METHODS
    + tuple(AUTOREGRESSIVE_ORDER_METHODS)
    + tuple(ROLE_CONDITIONED_ORDER_METHODS)
    + tuple(FACTORIZED_ENGAGEMENT_METHODS)
    + tuple(MCH_PPO_METHODS)
    + tuple(RG_MCH_PPO_METHODS)
    + tuple(SA_RG_MCH_PPO_METHODS)
    + tuple(BPCE_PPO_METHODS)
    + tuple(BPCE_RANDOM_PROBE_PPO_METHODS)
)
ALL_BENCHMARK_METHODS = tuple(RULE_POLICY_FACTORIES) + LEARNING_METHODS
DEFAULT_BENCHMARK_METHODS = tuple(RULE_POLICY_FACTORIES) + DEFAULT_LEARNING_METHODS


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
    high_threat_threshold: float = DEFAULT_HIGH_THREAT_THRESHOLD
    train_scenarios: tuple[str, ...] = (AIR_DEFENSE_V1_DEFAULT_SCENARIO,)
    eval_scenarios: tuple[str, ...] = (AIR_DEFENSE_V1_DEFAULT_SCENARIO,)
    methods: tuple[str, ...] | None = None
    include_learning: bool = True
    save_models: bool = True
    create_plot: bool = True
    record_decisions: bool = False
    record_training_dynamics: bool = False
    diagnostics_freq: int = 1_000
    probe_corpus_path: str | None = None

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
        if not 0.0 <= self.high_threat_threshold <= 1.0:
            raise ValueError("high_threat_threshold must be between 0 and 1")
        if self.diagnostics_freq <= 0:
            raise ValueError("diagnostics_freq must be positive")
        if self.probe_corpus_path is not None and not self.record_training_dynamics:
            raise ValueError(
                "probe_corpus_path requires record_training_dynamics=True"
            )
        _validate_scenario_names("train_scenarios", self.train_scenarios)
        _validate_scenario_names("eval_scenarios", self.eval_scenarios)
        if self.methods is not None:
            if not self.methods:
                raise ValueError("methods must contain at least one method")
            if len(set(self.methods)) != len(self.methods):
                raise ValueError("methods must be unique")
            unknown_methods = set(self.methods) - set(ALL_BENCHMARK_METHODS)
            if unknown_methods:
                available = ", ".join(ALL_BENCHMARK_METHODS)
                unknown = ", ".join(sorted(unknown_methods))
                raise ValueError(
                    f"Unknown benchmark methods: {unknown}. Available: {available}"
                )


@dataclass(frozen=True)
class BenchmarkArtifacts:
    output_dir: Path
    config: Path
    runs: Path
    episodes: Path
    decisions: Path
    decision_summary: Path
    leak_attributions: Path
    leak_attribution_summary: Path
    summary: Path
    paired_differences: Path
    generalization_matrix: Path
    learning_curves: Path
    learning_curve_summary: Path
    model_parameter_counts: Path
    training_dynamics: Path
    probe_dynamics: Path
    curve_figure_base: Path
    generalization_figure_base: Path
    models_dir: Path
    tensorboard_dir: Path


@dataclass(frozen=True)
class BenchmarkResult:
    artifacts: BenchmarkArtifacts
    run_rows: tuple[MetricRow, ...]
    episode_rows: tuple[MetricRow, ...]
    decision_rows: tuple[MetricRow, ...]
    decision_summary_rows: tuple[MetricRow, ...]
    leak_attribution_rows: tuple[MetricRow, ...]
    leak_attribution_summary_rows: tuple[MetricRow, ...]
    summary_rows: tuple[MetricRow, ...]
    paired_difference_rows: tuple[MetricRow, ...]
    generalization_rows: tuple[MetricRow, ...]
    curve_rows: tuple[MetricRow, ...]
    curve_summary_rows: tuple[MetricRow, ...]
    training_dynamics_rows: tuple[MetricRow, ...]
    probe_dynamics_rows: tuple[MetricRow, ...]
    figure_paths: tuple[Path, ...]


class EvaluationCurveCallback(BaseCallback):
    """Evaluate a model on held-out seeds at fixed training intervals."""

    def __init__(
        self,
        *,
        method: str,
        train_scenario: str,
        train_seed: int,
        eval_freq: int,
        eval_episodes: int,
        eval_seed: int,
        env_config: AirDefenseV1EnvConfig | None,
        use_action_masks: bool,
        env_factory: Callable[[], Any] | None = None,
        high_threat_threshold: float = DEFAULT_HIGH_THREAT_THRESHOLD,
    ) -> None:
        super().__init__(verbose=0)
        if eval_freq <= 0:
            raise ValueError("eval_freq must be positive")
        if eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive")
        self.method = method
        self.train_scenario = train_scenario
        self.train_seed = train_seed
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.env_config = env_config
        self.use_action_masks = use_action_masks
        self.env_factory = env_factory
        self.high_threat_threshold = high_threat_threshold
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
            env_factory=self.env_factory,
            env_config=self.env_config,
            episodes=self.eval_episodes,
            seed=self.eval_seed,
            use_action_masks=self.use_action_masks,
            high_threat_threshold=self.high_threat_threshold,
        )
        self.rows.append(
            {
                "method": self.method,
                "train_scenario": self.train_scenario,
                "eval_scenario": self.train_scenario,
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
        episodes=output_path / "episodes.csv",
        decisions=output_path / "decisions.csv",
        decision_summary=output_path / "decision_summary.csv",
        leak_attributions=output_path / "leak_attributions.csv",
        leak_attribution_summary=output_path / "leak_attribution_summary.csv",
        summary=output_path / "summary.csv",
        paired_differences=output_path / "paired_differences.csv",
        generalization_matrix=output_path / "generalization_matrix.csv",
        learning_curves=output_path / "learning_curves.csv",
        learning_curve_summary=output_path / "learning_curve_summary.csv",
        model_parameter_counts=output_path / "model_parameter_counts.json",
        training_dynamics=output_path / "training_dynamics.csv",
        probe_dynamics=output_path / "probe_dynamics.csv",
        curve_figure_base=output_path / "learning_curves",
        generalization_figure_base=output_path / "generalization",
        models_dir=output_path / "models",
        tensorboard_dir=output_path / "tensorboard",
    )


def _validate_scenario_names(field_name: str, names: Sequence[str]) -> None:
    if not names:
        raise ValueError(f"{field_name} must contain at least one scenario")
    canonical_names = [
        get_air_defense_v1_scenario_profile(name).name
        for name in names
    ]
    if len(set(canonical_names)) != len(canonical_names):
        raise ValueError(f"{field_name} must contain unique canonical scenarios")


def _resolve_benchmark_methods(
    protocol: AirDefenseV1BenchmarkConfig,
) -> tuple[str, ...]:
    if protocol.methods is not None:
        return protocol.methods
    if protocol.include_learning:
        return DEFAULT_BENCHMARK_METHODS
    return tuple(RULE_POLICY_FACTORIES)


def _resolve_scenario_configs(
    protocol: AirDefenseV1BenchmarkConfig,
    env_config: AirDefenseV1EnvConfig | None,
) -> tuple[dict[str, AirDefenseV1EnvConfig], dict[str, AirDefenseV1EnvConfig]]:
    if env_config is not None:
        default_names = (AIR_DEFENSE_V1_DEFAULT_SCENARIO,)
        if (
            protocol.train_scenarios != default_names
            or protocol.eval_scenarios != default_names
        ):
            raise ValueError(
                "env_config is a legacy single-scenario override and cannot be "
                "combined with train_scenarios or eval_scenarios"
            )
        return {"custom": env_config}, {"custom": env_config}

    return (
        _named_scenario_configs(protocol.train_scenarios),
        _named_scenario_configs(protocol.eval_scenarios),
    )


def _named_scenario_configs(
    names: Sequence[str],
) -> dict[str, AirDefenseV1EnvConfig]:
    return {
        profile.name: profile.config
        for profile in (
            get_air_defense_v1_scenario_profile(name)
            for name in names
        )
    }


def _validate_scenario_space_compatibility(
    train_scenarios: dict[str, AirDefenseV1EnvConfig],
    eval_scenarios: dict[str, AirDefenseV1EnvConfig],
) -> None:
    reference_name, reference_config = next(iter(train_scenarios.items()))
    reference_signature = _scenario_space_signature(reference_config)
    for scenario_name, config in {
        **train_scenarios,
        **eval_scenarios,
    }.items():
        signature = _scenario_space_signature(config)
        if signature != reference_signature:
            raise ValueError(
                "Incompatible AirDefense v1 scenario spaces: "
                f"{reference_name!r} has observation/action signature "
                f"{reference_signature}, but {scenario_name!r} has {signature}"
            )


def _scenario_space_signature(
    config: AirDefenseV1EnvConfig,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    env = AirDefenseResourceAssignmentEnvV1(config=config)
    observation_shape = tuple(int(value) for value in env.observation_space.shape)
    action_shape = tuple(int(value) for value in env.action_space.nvec.tolist())
    env.close()
    return observation_shape, action_shape


def _create_learning_environment(
    method: str,
    config: AirDefenseV1EnvConfig,
) -> Any:
    env = AirDefenseResourceAssignmentEnvV1(config=config)
    if method == "conflict_free_maskable_ppo":
        return ConflictFreeJointActionWrapper(env)
    return env


def _method_space_signature(
    method: str,
    config: AirDefenseV1EnvConfig,
) -> dict[str, object]:
    env = _create_learning_environment(method, config)
    action_space = env.action_space
    if hasattr(action_space, "nvec"):
        action_signature: dict[str, object] = {
            "type": "MultiDiscrete",
            "nvec": [int(value) for value in action_space.nvec.tolist()],
        }
    else:
        action_signature = {
            "type": "Discrete",
            "n": int(action_space.n),
        }
    signature = {
        "observation_shape": [
            int(value) for value in env.observation_space.shape
        ],
        "action_space": action_signature,
        "action_generator": _method_action_generator_signature(
            method,
            num_units=len(config.defense_units),
            num_targets=(len(config.targets) or config.num_random_targets),
            num_zones=len(config.protected_zones),
        ),
    }
    env.close()
    return signature


def _method_action_generator_signature(
    method: str,
    *,
    num_units: int,
    num_targets: int,
    num_zones: int,
) -> dict[str, object]:
    if (
        method in FACTORIZED_ENGAGEMENT_METHODS
        or method in MCH_PPO_METHODS
        or method in RG_MCH_PPO_METHODS
        or method in SA_RG_MCH_PPO_METHODS
    ):
        unit_order = (
            FACTORIZED_ENGAGEMENT_METHODS.get(method)
            or MCH_PPO_METHODS.get(method)
            or RG_MCH_PPO_METHODS.get(method)
            or SA_RG_MCH_PPO_METHODS[method]
        )
        if len(unit_order) != num_units:
            raise ValueError(
                f"Method {method!r} requires {len(unit_order)} units, "
                f"but the environment has {num_units}"
            )
        signature = {
            "type": "factorized_engagement_autoregressive_conflict_free",
            "unit_order": list(unit_order),
            "conditional_target_mask": True,
            "joint_log_prob": "sum_of_conditional_log_probs",
            "environment_steps_per_joint_action": 1,
            "probability_schema": {
                "noop": "1-sigmoid(engage_logit)",
                "target": "sigmoid(engage_logit)*softmax(legal_target_logits)",
                "entropy": "exact_final_discrete_distribution",
                "deterministic_rule": "bernoulli_argmax_then_target_argmax",
            },
            "actor_head": FactorizedEngagementActionHeadConfig().signature(),
            "observation_layout": AirDefenseV1ObservationLayout(
                num_zones=num_zones,
                num_targets=num_targets,
                num_units=num_units,
            ).signature(),
        }
        if method in MCH_PPO_METHODS:
            signature["optimizer"] = {
                "type": "masked_counterfactual_hierarchical_ppo",
                "critic_source": "training.mch_q_critic_paths",
            }
        if method in RG_MCH_PPO_METHODS:
            signature["optimizer"] = {
                "type": "reliability_gated_mch_ppo",
                "critic_source": "training.mch_q_critic_paths",
                "base_advantage": "normalized_on_policy_gae",
            }
        if method in SA_RG_MCH_PPO_METHODS:
            signature["optimizer"] = {
                "type": "support_anchored_reliability_gated_mch_ppo",
                "critic_source": "training.mch_q_critic_paths",
                "support_source": "training.sa_rg_mch_support_dataset_path",
                "base_advantage": "normalized_on_policy_gae",
                "combined_reliability": (
                    "ensemble_agreement_times_context_support"
                ),
            }
        return signature
    if method in ROLE_CONDITIONED_ORDER_METHODS:
        unit_order = ROLE_CONDITIONED_ORDER_METHODS[method]
        if len(unit_order) != num_units:
            raise ValueError(
                f"Method {method!r} requires {len(unit_order)} units, "
                f"but the environment has {num_units}"
            )
        return {
            "type": "role_conditioned_autoregressive_conflict_free",
            "unit_order": list(unit_order),
            "conditional_target_mask": True,
            "joint_log_prob": "sum_of_conditional_log_probs",
            "environment_steps_per_joint_action": 1,
            "actor_head": RoleConditionedActionHeadConfig().signature(),
            "observation_layout": AirDefenseV1ObservationLayout(
                num_zones=num_zones,
                num_targets=num_targets,
                num_units=num_units,
            ).signature(),
        }
    if method == "autoregressive_maskable_ppo" or method in AUTOREGRESSIVE_ORDER_METHODS:
        unit_order = AUTOREGRESSIVE_ORDER_METHODS.get(
            method,
            tuple(range(num_units)),
        )
        if len(unit_order) != num_units:
            raise ValueError(
                f"Method {method!r} requires {len(unit_order)} units, "
                f"but the environment has {num_units}"
            )
        return {
            "type": "autoregressive_conflict_free",
            "unit_order": list(unit_order),
            "conditional_target_mask": True,
            "joint_log_prob": "sum_of_conditional_log_probs",
            "environment_steps_per_joint_action": 1,
        }
    if method == "conflict_free_maskable_ppo":
        return {
            "type": "enumerated_conflict_free",
            "conditional_target_mask": True,
            "environment_steps_per_joint_action": 1,
        }
    if method == "maskable_ppo":
        return {
            "type": "independent_masked",
            "conditional_target_mask": False,
            "environment_steps_per_joint_action": 1,
        }
    if method == "ppo":
        return {
            "type": "independent_unmasked",
            "conditional_target_mask": False,
            "environment_steps_per_joint_action": 1,
        }
    return {
        "type": "rule_joint_policy",
        "environment_steps_per_joint_action": 1,
    }


def _build_evaluation_blocks(
    protocol: AirDefenseV1BenchmarkConfig,
    eval_scenarios: dict[str, AirDefenseV1EnvConfig],
    *,
    run_index: int,
) -> tuple[EvaluationBlock, ...]:
    block_stride = len(eval_scenarios) * protocol.eval_episodes
    run_seed = protocol.eval_seed + run_index * block_stride
    return tuple(
        (
            scenario_name,
            config,
            run_seed + scenario_index * protocol.eval_episodes,
        )
        for scenario_index, (scenario_name, config) in enumerate(
            eval_scenarios.items()
        )
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
    training = train_config or AirDefenseV1PPOConfig()
    methods = _resolve_benchmark_methods(protocol)
    train_scenario_configs, eval_scenario_configs = _resolve_scenario_configs(
        protocol,
        env_config,
    )
    _validate_scenario_space_compatibility(
        train_scenario_configs,
        eval_scenario_configs,
    )
    artifacts = create_artifacts(output_dir)
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    config_record = _build_config_record(
        status="running",
        started_at=started_at,
        completed_at=None,
        protocol=protocol,
        training=training,
        train_scenarios=train_scenario_configs,
        eval_scenarios=eval_scenario_configs,
        methods=methods,
        artifacts=artifacts,
    )
    _write_json(artifacts.config, config_record)

    run_rows: list[MetricRow] = []
    episode_rows: list[MetricRow] = []
    decision_rows: list[MetricRow] = []
    leak_attribution_rows: list[MetricRow] = []
    curve_rows: list[MetricRow] = []
    training_dynamics_rows: list[MetricRow] = []
    probe_dynamics_rows: list[MetricRow] = []
    rule_methods = tuple(method for method in methods if method in RULE_POLICY_FACTORIES)
    learning_methods = tuple(method for method in methods if method in LEARNING_METHODS)
    for train_scenario, train_environment in train_scenario_configs.items():
        for run_index, train_seed in enumerate(protocol.train_seeds):
            evaluation_blocks = _build_evaluation_blocks(
                protocol,
                eval_scenario_configs,
                run_index=run_index,
            )
            if rule_methods:
                _report_progress(
                    progress_callback,
                    (
                        f"train_scenario={train_scenario}, "
                        f"run {run_index + 1}/{len(protocol.train_seeds)}: rules"
                    ),
                )
                (
                    rule_rows,
                    rule_episode_rows,
                    rule_decision_rows,
                    rule_leak_rows,
                ) = _evaluate_rule_methods(
                    train_scenario=train_scenario,
                    methods=rule_methods,
                    evaluation_blocks=evaluation_blocks,
                    run_index=run_index,
                    episodes=protocol.eval_episodes,
                    high_threat_threshold=protocol.high_threat_threshold,
                    record_decisions=protocol.record_decisions,
                )
                run_rows.extend(rule_rows)
                episode_rows.extend(rule_episode_rows)
                decision_rows.extend(rule_decision_rows)
                leak_attribution_rows.extend(rule_leak_rows)

            if learning_methods:
                curve_seed = (
                    protocol.curve_eval_seed
                    + run_index * protocol.curve_eval_episodes
                )
                (
                    learning_rows,
                    learning_curves,
                    learning_episode_rows,
                    learning_decision_rows,
                    learning_leak_rows,
                    learning_training_dynamics,
                    learning_probe_dynamics,
                ) = _train_learning_methods(
                    train_scenario=train_scenario,
                    train_environment=train_environment,
                    evaluation_blocks=evaluation_blocks,
                    methods=learning_methods,
                    base_training=training,
                    artifacts=artifacts,
                    protocol=protocol,
                    run_index=run_index,
                    train_seed=train_seed,
                    curve_seed=curve_seed,
                    progress_callback=progress_callback,
                )
                run_rows.extend(learning_rows)
                curve_rows.extend(learning_curves)
                episode_rows.extend(learning_episode_rows)
                decision_rows.extend(learning_decision_rows)
                leak_attribution_rows.extend(learning_leak_rows)
                training_dynamics_rows.extend(learning_training_dynamics)
                probe_dynamics_rows.extend(learning_probe_dynamics)

    summary_rows = summarize_rows(
        run_rows,
        group_keys=("method", "method_type", "train_scenario", "eval_scenario"),
        metrics=METRIC_NAMES,
        confidence_level=protocol.confidence_level,
    )
    curve_summary_rows = summarize_rows(
        curve_rows,
        group_keys=("method", "train_scenario", "eval_scenario", "timesteps"),
        metrics=CURVE_METRIC_NAMES,
        confidence_level=protocol.confidence_level,
    )
    paired_difference_rows = summarize_paired_differences(
        run_rows,
        metrics=METRIC_NAMES,
        confidence_level=protocol.confidence_level,
        method_order=methods,
    )
    generalization_rows = build_generalization_matrix_rows(
        summary_rows,
        methods=methods,
        train_scenarios=tuple(train_scenario_configs),
        eval_scenarios=tuple(eval_scenario_configs),
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
    leak_attribution_summary_rows = aggregate_leak_attributions(
        leak_attribution_rows,
        group_keys=(
            "method",
            "method_type",
            "train_scenario",
            "eval_scenario",
            "run_index",
            "train_seed",
        ),
    )

    _write_csv(artifacts.runs, run_rows, RUN_FIELDNAMES)
    _write_csv(artifacts.episodes, episode_rows, EPISODE_FIELDNAMES)
    _write_csv(artifacts.decisions, decision_rows, DECISION_FIELDNAMES)
    _write_csv(
        artifacts.decision_summary,
        decision_summary_rows,
        DECISION_SUMMARY_FIELDNAMES,
    )
    _write_csv(
        artifacts.leak_attributions,
        leak_attribution_rows,
        LEAK_ATTRIBUTION_FIELDNAMES,
    )
    _write_csv(
        artifacts.leak_attribution_summary,
        leak_attribution_summary_rows,
        LEAK_ATTRIBUTION_SUMMARY_FIELDNAMES,
    )
    _write_csv(artifacts.summary, summary_rows, SUMMARY_FIELDNAMES)
    _write_csv(
        artifacts.paired_differences,
        paired_difference_rows,
        PAIRED_DIFFERENCE_FIELDNAMES,
    )
    _write_csv(
        artifacts.generalization_matrix,
        generalization_rows,
        GENERALIZATION_FIELDNAMES,
    )
    _write_csv(artifacts.learning_curves, curve_rows, CURVE_FIELDNAMES)
    _write_csv(
        artifacts.learning_curve_summary,
        curve_summary_rows,
        CURVE_SUMMARY_FIELDNAMES,
    )
    _write_csv(
        artifacts.training_dynamics,
        training_dynamics_rows,
        TRAINING_DYNAMICS_FIELDNAMES,
    )
    _write_csv(
        artifacts.probe_dynamics,
        probe_dynamics_rows,
        PROBE_DYNAMICS_FIELDNAMES,
    )
    parameter_records = _unique_model_parameter_records(run_rows)
    _write_json(
        artifacts.model_parameter_counts,
        {
            "schema_version": 1,
            "models": parameter_records,
        },
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
    if protocol.create_plot and generalization_rows:
        figure_paths += tuple(
            plot_generalization_matrix(
                generalization_rows,
                artifacts.generalization_figure_base,
                methods=methods,
                train_scenarios=tuple(train_scenario_configs),
                eval_scenarios=tuple(eval_scenario_configs),
            )
        )

    completed_at = datetime.now(timezone.utc)
    config_record = _build_config_record(
        status="completed",
        started_at=started_at,
        completed_at=completed_at,
        protocol=protocol,
        training=training,
        train_scenarios=train_scenario_configs,
        eval_scenarios=eval_scenario_configs,
        methods=methods,
        artifacts=artifacts,
    )
    config_record["figure_contract"] = {
        "core_conclusion": (
            "Compare selected learning methods and rule baselines "
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
        "episode_rows": len(episode_rows),
        "decision_rows": len(decision_rows),
        "decision_summary_rows": len(decision_summary_rows),
        "leak_attribution_rows": len(leak_attribution_rows),
        "leak_attribution_summary_rows": len(
            leak_attribution_summary_rows
        ),
        "paired_difference_rows": len(paired_difference_rows),
        "generalization_rows": len(generalization_rows),
        "curve_rows": len(curve_rows),
        "training_dynamics_rows": len(training_dynamics_rows),
        "probe_dynamics_rows": len(probe_dynamics_rows),
        "model_parameter_records": len(parameter_records),
    }
    _write_json(artifacts.config, config_record)

    return BenchmarkResult(
        artifacts=artifacts,
        run_rows=tuple(run_rows),
        episode_rows=tuple(episode_rows),
        decision_rows=tuple(decision_rows),
        decision_summary_rows=tuple(decision_summary_rows),
        leak_attribution_rows=tuple(leak_attribution_rows),
        leak_attribution_summary_rows=tuple(leak_attribution_summary_rows),
        summary_rows=tuple(summary_rows),
        paired_difference_rows=tuple(paired_difference_rows),
        generalization_rows=tuple(generalization_rows),
        curve_rows=tuple(curve_rows),
        curve_summary_rows=tuple(curve_summary_rows),
        training_dynamics_rows=tuple(training_dynamics_rows),
        probe_dynamics_rows=tuple(probe_dynamics_rows),
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


def summarize_paired_differences(
    run_rows: Sequence[MetricRow],
    *,
    metrics: Sequence[str],
    confidence_level: float,
    method_order: Sequence[str],
) -> list[MetricRow]:
    indexed_rows = {
        (
            str(row["train_scenario"]),
            str(row["eval_scenario"]),
            str(row["method"]),
            int(row["run_index"]),
        ): row
        for row in run_rows
    }
    scenario_pairs = sorted(
        {
            (str(row["train_scenario"]), str(row["eval_scenario"]))
            for row in run_rows
        }
    )
    output: list[MetricRow] = []
    for train_scenario, eval_scenario in scenario_pairs:
        for method_a, method_b in combinations(method_order, 2):
            paired_run_indices = sorted(
                {
                    int(row["run_index"])
                    for row in run_rows
                    if row["train_scenario"] == train_scenario
                    and row["eval_scenario"] == eval_scenario
                    and row["method"] == method_a
                }
                & {
                    int(row["run_index"])
                    for row in run_rows
                    if row["train_scenario"] == train_scenario
                    and row["eval_scenario"] == eval_scenario
                    and row["method"] == method_b
                }
            )
            if not paired_run_indices:
                continue
            for run_index in paired_run_indices:
                row_a = indexed_rows[
                    (train_scenario, eval_scenario, method_a, run_index)
                ]
                row_b = indexed_rows[
                    (train_scenario, eval_scenario, method_b, run_index)
                ]
                if row_a["evaluation_seed"] != row_b["evaluation_seed"]:
                    raise ValueError(
                        "Paired comparison requires identical evaluation seeds for "
                        f"{method_a} and {method_b}"
                    )

            for metric in metrics:
                differences = np.asarray(
                    [
                        float(
                            indexed_rows[
                                (train_scenario, eval_scenario, method_a, run_index)
                            ][metric]
                        )
                        - float(
                            indexed_rows[
                                (train_scenario, eval_scenario, method_b, run_index)
                            ][metric]
                        )
                        for run_index in paired_run_indices
                    ],
                    dtype=np.float64,
                )
                mean, std, sem, ci_low, ci_high = _mean_confidence_interval(
                    differences,
                    confidence_level,
                )
                output.append(
                    {
                        "train_scenario": train_scenario,
                        "eval_scenario": eval_scenario,
                        "method_a": method_a,
                        "method_b": method_b,
                        "metric": metric,
                        "n_pairs": len(differences),
                        "mean_difference": mean,
                        "std_difference": std,
                        "sem_difference": sem,
                        "confidence_level": confidence_level,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
    return output


def build_generalization_matrix_rows(
    summary_rows: Sequence[MetricRow],
    *,
    methods: Sequence[str],
    train_scenarios: Sequence[str],
    eval_scenarios: Sequence[str],
) -> list[MetricRow]:
    method_order = {method: index for index, method in enumerate(methods)}
    train_order = {
        scenario: index for index, scenario in enumerate(train_scenarios)
    }
    eval_order = {
        scenario: index for index, scenario in enumerate(eval_scenarios)
    }
    ordered_rows = sorted(
        summary_rows,
        key=lambda row: (
            method_order[str(row["method"])],
            str(row["metric"]),
            train_order[str(row["train_scenario"])],
            eval_order[str(row["eval_scenario"])],
        ),
    )
    return [
        {
            "method": row["method"],
            "method_type": row["method_type"],
            "train_scenario": row["train_scenario"],
            "eval_scenario": row["eval_scenario"],
            "metric": row["metric"],
            "n_runs": row["n_runs"],
            "mean": row["mean"],
            "std": row["std"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
        }
        for row in ordered_rows
    ]


def plot_generalization_matrix(
    generalization_rows: Sequence[MetricRow],
    output_base: str | Path,
    *,
    methods: Sequence[str],
    train_scenarios: Sequence[str],
    eval_scenarios: Sequence[str],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reward_rows = [
        row for row in generalization_rows if row["metric"] == "avg_reward"
    ]
    if not reward_rows:
        return []

    ncols = min(3, len(methods))
    nrows = int(np.ceil(len(methods) / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.0 * ncols, 2.4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    values = np.asarray([float(row["mean"]) for row in reward_rows])
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if np.isclose(value_min, value_max):
        value_min -= 1.0
        value_max += 1.0

    images = []
    active_axes = []
    for method_index, method in enumerate(methods):
        axis = axes.flat[method_index]
        matrix = np.full(
            (len(train_scenarios), len(eval_scenarios)),
            np.nan,
            dtype=np.float64,
        )
        row_index = {name: index for index, name in enumerate(train_scenarios)}
        column_index = {name: index for index, name in enumerate(eval_scenarios)}
        for row in reward_rows:
            if row["method"] != method:
                continue
            matrix[row_index[str(row["train_scenario"])], column_index[str(row["eval_scenario"])]] = float(
                row["mean"]
            )
        image = axis.imshow(
            matrix,
            cmap="RdYlGn",
            vmin=value_min,
            vmax=value_max,
            aspect="auto",
        )
        images.append(image)
        active_axes.append(axis)
        axis.set_title(method.replace("_", " "))
        axis.set_xticks(range(len(eval_scenarios)), eval_scenarios, rotation=35, ha="right")
        axis.set_yticks(range(len(train_scenarios)), train_scenarios)
        axis.set_xlabel("Evaluation scenario")
        axis.set_ylabel("Training scenario")
        for train_index in range(len(train_scenarios)):
            for eval_index in range(len(eval_scenarios)):
                value = matrix[train_index, eval_index]
                if np.isfinite(value):
                    axis.text(
                        eval_index,
                        train_index,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

    for unused_axis in axes.flat[len(methods):]:
        unused_axis.set_visible(False)
    figure.colorbar(
        images[0],
        ax=active_axes,
        label="Average episode reward",
        shrink=0.85,
    )

    output_path = Path(output_base)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for extension, dpi in (("svg", 300), ("pdf", 300), ("png", 300)):
        path = output_path.with_suffix(f".{extension}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(path)
    plt.close(figure)
    return saved_paths


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
        "conflict_free_maskable_ppo": "#B33A3A",
        "autoregressive_maskable_ppo": "#7A5195",
        "autoregressive_ppo_order_012": "#7A5195",
        "autoregressive_ppo_order_120": "#2A9D8F",
        "autoregressive_ppo_order_201": "#E76F51",
        "role_conditioned_ar_ppo_order_012": "#3A6B35",
        "role_conditioned_ar_ppo_order_120": "#C78A1D",
        "role_conditioned_ar_ppo_order_201": "#A23B72",
        "factorized_engagement_ar_ppo_order_012": "#007C91",
    }
    method_labels = {
        "ppo": "PPO",
        "maskable_ppo": "Maskable PPO",
        "conflict_free_maskable_ppo": "Conflict-free Maskable PPO",
        "autoregressive_maskable_ppo": "Autoregressive Maskable PPO",
        "autoregressive_ppo_order_012": "Autoregressive order 012",
        "autoregressive_ppo_order_120": "Autoregressive order 120",
        "autoregressive_ppo_order_201": "Autoregressive order 201",
        "role_conditioned_ar_ppo_order_012": "Role-conditioned order 012",
        "role_conditioned_ar_ppo_order_120": "Role-conditioned order 120",
        "role_conditioned_ar_ppo_order_201": "Role-conditioned order 201",
        "factorized_engagement_ar_ppo_order_012": "Factorized engagement order 012",
    }

    for metric, axis in axes.items():
        metric_rows = [row for row in summary_rows if row["metric"] == metric]
        series_keys = sorted(
            {
                (str(row["method"]), str(row["train_scenario"]))
                for row in metric_rows
            }
        )
        multiple_train_scenarios = len(
            {train_scenario for _, train_scenario in series_keys}
        ) > 1
        line_styles = ("-", "--", ":", "-.")
        train_line_styles = {
            scenario: line_styles[index % len(line_styles)]
            for index, scenario in enumerate(
                sorted({scenario for _, scenario in series_keys})
            )
        }
        for method, train_scenario in series_keys:
            method_rows = sorted(
                (
                    row
                    for row in metric_rows
                    if row["method"] == method
                    and row["train_scenario"] == train_scenario
                ),
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
                linestyle=train_line_styles[train_scenario],
                label=(
                    f"{method_labels.get(method, method)} @ {train_scenario}"
                    if multiple_train_scenarios
                    else method_labels.get(method, method)
                ),
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
    train_scenario: str,
    methods: Sequence[str],
    evaluation_blocks: Sequence[EvaluationBlock],
    run_index: int,
    episodes: int,
    high_threat_threshold: float,
    record_decisions: bool,
) -> tuple[
    list[MetricRow],
    list[MetricRow],
    list[MetricRow],
    list[MetricRow],
    list[MetricRow],
    list[MetricRow],
]:
    rows: list[MetricRow] = []
    episode_rows: list[MetricRow] = []
    decision_rows: list[MetricRow] = []
    leak_attribution_rows: list[MetricRow] = []
    for method in methods:
        policy_factory = RULE_POLICY_FACTORIES[method]
        for eval_scenario, environment, evaluation_seed in evaluation_blocks:
            method_episode_rows: list[MetricRow] = []

            def record_episode(raw_metrics: dict[str, float | int | bool]) -> None:
                episode_index = len(method_episode_rows)
                method_episode_rows.append(
                    {
                        "method": method,
                        "method_type": "rule",
                        "train_scenario": train_scenario,
                        "eval_scenario": eval_scenario,
                        "run_index": run_index,
                        "train_seed": "",
                        "evaluation_seed": evaluation_seed,
                        "episode_index": episode_index,
                        "episode_seed": evaluation_seed + episode_index,
                        **raw_metrics,
                    }
                )

            def record_decision(
                episode_index: int,
                raw_decision: dict[str, Any],
            ) -> None:
                decision_rows.append(
                    {
                        "method": method,
                        "method_type": "rule",
                        "train_scenario": train_scenario,
                        "eval_scenario": eval_scenario,
                        "run_index": run_index,
                        "train_seed": "",
                        "evaluation_seed": evaluation_seed,
                        "episode_index": episode_index,
                        "episode_seed": evaluation_seed + episode_index,
                        **raw_decision,
                    }
                )

            def record_leak(
                episode_index: int,
                raw_attribution: dict[str, Any],
            ) -> None:
                leak_attribution_rows.append(
                    {
                        "method": method,
                        "method_type": "rule",
                        "train_scenario": train_scenario,
                        "eval_scenario": eval_scenario,
                        "run_index": run_index,
                        "train_seed": "",
                        "evaluation_seed": evaluation_seed,
                        "episode_index": episode_index,
                        "episode_seed": evaluation_seed + episode_index,
                        **raw_attribution,
                    }
                )

            metrics = evaluate_air_defense_v1_policy(
                env_factory=lambda environment=environment: (
                    AirDefenseResourceAssignmentEnvV1(config=environment)
                ),
                policy_factory=policy_factory,
                episodes=episodes,
                seed=evaluation_seed,
                high_threat_threshold=high_threat_threshold,
                episode_metrics_callback=record_episode,
                decision_trace_callback=(record_decision if record_decisions else None),
                leak_attribution_callback=(record_leak if record_decisions else None),
            )
            episode_rows.extend(method_episode_rows)
            rows.append(
                {
                    "method": method,
                    "method_type": "rule",
                    "train_scenario": train_scenario,
                    "eval_scenario": eval_scenario,
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
    return rows, episode_rows, decision_rows, leak_attribution_rows


def _train_learning_methods(
    *,
    train_scenario: str,
    train_environment: AirDefenseV1EnvConfig,
    evaluation_blocks: Sequence[EvaluationBlock],
    methods: Sequence[str],
    base_training: AirDefenseV1PPOConfig,
    artifacts: BenchmarkArtifacts,
    protocol: AirDefenseV1BenchmarkConfig,
    run_index: int,
    train_seed: int,
    curve_seed: int,
    progress_callback: ProgressCallback | None,
) -> tuple[
    list[MetricRow],
    list[MetricRow],
    list[MetricRow],
    list[MetricRow],
    list[MetricRow],
    list[MetricRow],
    list[MetricRow],
]:
    available_method_specs = {
        "ppo": (train_ppo, False, None),
        "maskable_ppo": (train_maskable_ppo, True, None),
        "conflict_free_maskable_ppo": (
            train_conflict_free_maskable_ppo,
            True,
            None,
        ),
        "autoregressive_maskable_ppo": (
            train_autoregressive_maskable_ppo,
            True,
            None,
        ),
    }
    available_method_specs.update(
        {
            method: (train_autoregressive_maskable_ppo, True, unit_order)
            for method, unit_order in AUTOREGRESSIVE_ORDER_METHODS.items()
        }
    )
    available_method_specs.update(
        {
            method: (train_bpce_ppo, True, unit_order)
            for method, unit_order in (
                BPCE_PPO_METHODS | BPCE_RANDOM_PROBE_PPO_METHODS
            ).items()
        }
    )
    available_method_specs.update(
        {
            method: (train_rg_mch_ppo, True, unit_order)
            for method, unit_order in RG_MCH_PPO_METHODS.items()
        }
    )
    available_method_specs.update(
        {
            method: (train_sa_rg_mch_ppo, True, unit_order)
            for method, unit_order in SA_RG_MCH_PPO_METHODS.items()
        }
    )
    available_method_specs.update(
        {
            method: (train_mch_ppo, True, unit_order)
            for method, unit_order in MCH_PPO_METHODS.items()
        }
    )
    available_method_specs.update(
        {
            method: (
                train_factorized_engagement_autoregressive_ppo,
                True,
                unit_order,
            )
            for method, unit_order in FACTORIZED_ENGAGEMENT_METHODS.items()
        }
    )
    available_method_specs.update(
        {
            method: (
                train_role_conditioned_autoregressive_ppo,
                True,
                unit_order,
            )
            for method, unit_order in ROLE_CONDITIONED_ORDER_METHODS.items()
        }
    )
    rows: list[MetricRow] = []
    curves: list[MetricRow] = []
    episode_rows: list[MetricRow] = []
    decision_rows: list[MetricRow] = []
    leak_attribution_rows: list[MetricRow] = []
    training_dynamics_rows: list[MetricRow] = []
    probe_dynamics_rows: list[MetricRow] = []
    probe_corpus = (
        PolicyProbeCorpus.load(protocol.probe_corpus_path)
        if protocol.probe_corpus_path is not None
        else None
    )
    training = replace(
        base_training,
        seed=train_seed,
        tensorboard_log=str(artifacts.tensorboard_dir),
    )
    for method in methods:
        train_fn, use_action_masks, unit_order = available_method_specs[method]
        _report_progress(
            progress_callback,
            (
                f"train_scenario={train_scenario}, "
                f"run {run_index + 1}/{len(protocol.train_seeds)}: "
                f"train {method}, seed={train_seed}"
            ),
        )
        curve_callback = EvaluationCurveCallback(
            method=method,
            train_scenario=train_scenario,
            train_seed=train_seed,
            eval_freq=protocol.curve_eval_freq,
            eval_episodes=protocol.curve_eval_episodes,
            eval_seed=curve_seed,
            env_config=train_environment,
            use_action_masks=use_action_masks,
            env_factory=lambda method=method, environment=train_environment: (
                _create_learning_environment(method, environment)
            ),
            high_threat_threshold=protocol.high_threat_threshold,
        )
        diagnostics_callback: PPOTrainingDiagnosticsCallback | None = None
        callback: Any = curve_callback
        if protocol.record_training_dynamics:
            diagnostics_callback = PPOTrainingDiagnosticsCallback(
                method=method,
                train_scenario=train_scenario,
                train_seed=train_seed,
                record_freq=protocol.diagnostics_freq,
                probe_corpus=probe_corpus,
            )
            callback = CallbackList([curve_callback, diagnostics_callback])
        save_path = None
        if protocol.save_models:
            save_path = (
                artifacts.models_dir
                / train_scenario
                / f"{method}_seed{train_seed}.zip"
            )
        started = perf_counter()
        method_training = training
        if method in BPCE_RANDOM_PROBE_PPO_METHODS:
            method_training = replace(
                training,
                bpce_probe_selection_mode="random",
            )
        train_kwargs: dict[str, Any] = {
            "env_config": train_environment,
            "train_config": method_training,
            "save_path": save_path,
            "callback": callback,
            "tb_log_name": f"{train_scenario}_{method}_seed{train_seed}",
        }
        if unit_order is not None:
            train_kwargs["unit_order"] = unit_order
        model = train_fn(
            **train_kwargs,
        )
        training_seconds = perf_counter() - started
        parameter_counts = policy_parameter_counts(model.policy)
        mch_training_diagnostics = getattr(
            model, "last_mch_training_diagnostics", {}
        )
        bpce_diagnostics = {
            **{
                f"bpce_probe_{key}": value
                for key, value in getattr(
                    model, "last_bpce_probe_diagnostics", {}
                ).items()
            },
            **{
                f"bpce_train_{key}": value
                for key, value in getattr(
                    model, "last_bpce_training_diagnostics", {}
                ).items()
            },
        }
        for eval_scenario, environment, evaluation_seed in evaluation_blocks:
            method_episode_rows: list[MetricRow] = []

            def record_episode(raw_metrics: dict[str, float | int | bool]) -> None:
                episode_index = len(method_episode_rows)
                method_episode_rows.append(
                    {
                        "method": method,
                        "method_type": "learning",
                        "train_scenario": train_scenario,
                        "eval_scenario": eval_scenario,
                        "run_index": run_index,
                        "train_seed": train_seed,
                        "evaluation_seed": evaluation_seed,
                        "episode_index": episode_index,
                        "episode_seed": evaluation_seed + episode_index,
                        **raw_metrics,
                    }
                )

            def record_decision(
                episode_index: int,
                raw_decision: dict[str, object],
            ) -> None:
                decision_rows.append(
                    {
                        "method": method,
                        "method_type": "learning",
                        "train_scenario": train_scenario,
                        "eval_scenario": eval_scenario,
                        "run_index": run_index,
                        "train_seed": train_seed,
                        "evaluation_seed": evaluation_seed,
                        "episode_index": episode_index,
                        "episode_seed": evaluation_seed + episode_index,
                        **raw_decision,
                    }
                )

            def record_leak(
                episode_index: int,
                raw_attribution: dict[str, object],
            ) -> None:
                leak_attribution_rows.append(
                    {
                        "method": method,
                        "method_type": "learning",
                        "train_scenario": train_scenario,
                        "eval_scenario": eval_scenario,
                        "run_index": run_index,
                        "train_seed": train_seed,
                        "evaluation_seed": evaluation_seed,
                        "episode_index": episode_index,
                        "episode_seed": evaluation_seed + episode_index,
                        **raw_attribution,
                    }
                )

            metrics = evaluate_air_defense_v1_model(
                model,
                env_factory=lambda method=method, environment=environment: (
                    _create_learning_environment(method, environment)
                ),
                episodes=protocol.eval_episodes,
                seed=evaluation_seed,
                use_action_masks=use_action_masks,
                high_threat_threshold=protocol.high_threat_threshold,
                episode_metrics_callback=record_episode,
                decision_trace_callback=(
                    record_decision if protocol.record_decisions else None
                ),
                leak_attribution_callback=(
                    record_leak if protocol.record_decisions else None
                ),
            )
            episode_rows.extend(method_episode_rows)
            rows.append(
                {
                    "method": method,
                    "method_type": "learning",
                    "train_scenario": train_scenario,
                    "eval_scenario": eval_scenario,
                    "run_index": run_index,
                    "train_seed": train_seed,
                    "evaluation_seed": evaluation_seed,
                    "requested_timesteps": method_training.total_timesteps,
                    "training_timesteps": int(model.num_timesteps),
                    "training_seconds": training_seconds,
                    "model_path": str(save_path) if save_path is not None else "",
                    **parameter_counts,
                    **mch_training_diagnostics,
                    **bpce_diagnostics,
                    **metrics,
                }
            )
        for curve_row in curve_callback.rows:
            curves.append({"run_index": run_index, **curve_row})
        if diagnostics_callback is not None:
            training_dynamics_rows.extend(
                {"run_index": run_index, **row}
                for row in diagnostics_callback.training_rows
            )
            probe_dynamics_rows.extend(
                {"run_index": run_index, **row}
                for row in diagnostics_callback.probe_rows
            )
    return (
        rows,
        curves,
        episode_rows,
        decision_rows,
        leak_attribution_rows,
        training_dynamics_rows,
        probe_dynamics_rows,
    )


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


def _unique_model_parameter_records(
    run_rows: Sequence[MetricRow],
) -> list[MetricRow]:
    records: dict[tuple[object, ...], MetricRow] = {}
    for row in run_rows:
        if row.get("method_type") != "learning":
            continue
        key = (
            row["method"],
            row["train_scenario"],
            row["run_index"],
            row["train_seed"],
        )
        records[key] = {
            "method": row["method"],
            "train_scenario": row["train_scenario"],
            "run_index": row["run_index"],
            "train_seed": row["train_seed"],
            "model_path": row.get("model_path", ""),
            "actor_parameters": int(row["actor_parameters"]),
            "critic_parameters": int(row["critic_parameters"]),
            "shared_parameters": int(row["shared_parameters"]),
            "total_parameters": int(row["total_parameters"]),
        }
    return [records[key] for key in sorted(records, key=str)]


def _build_config_record(
    *,
    status: str,
    started_at: datetime,
    completed_at: datetime | None,
    protocol: AirDefenseV1BenchmarkConfig,
    training: AirDefenseV1PPOConfig,
    train_scenarios: dict[str, AirDefenseV1EnvConfig],
    eval_scenarios: dict[str, AirDefenseV1EnvConfig],
    methods: Sequence[str],
    artifacts: BenchmarkArtifacts,
) -> dict[str, Any]:
    first_train_environment = next(iter(train_scenarios.values()))
    return {
        "schema_version": 8,
        "status": status,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat() if completed_at else None,
        "benchmark": asdict(protocol),
        "training": asdict(training),
        "environment": asdict(first_train_environment),
        "scenarios": {
            "training": {
                name: asdict(config) for name, config in train_scenarios.items()
            },
            "evaluation": {
                name: asdict(config) for name, config in eval_scenarios.items()
            },
            "space_signatures": {
                name: _scenario_space_signature(config)
                for name, config in {**train_scenarios, **eval_scenarios}.items()
            },
        },
        "methods": {
            "selected": list(methods),
            "rule": [method for method in methods if method in RULE_POLICY_FACTORIES],
            "learning": [method for method in methods if method in LEARNING_METHODS],
            "space_signatures": {
                method: _method_space_signature(method, first_train_environment)
                for method in methods
            },
        },
        "evaluation_protocol": {
            "paired_scenario_blocks": True,
            "final_evaluation_seed_formula": (
                "eval_seed + run_index * len(eval_scenarios) * eval_episodes "
                "+ eval_scenario_index * eval_episodes"
            ),
            "curve_evaluation_seed_formula": (
                "curve_eval_seed + run_index * curve_eval_episodes"
            ),
            "confidence_interval": "two-sided Student-t interval across runs",
            "high_threat_definition": (
                "target.threat >= benchmark.high_threat_threshold"
            ),
            "diagnostic_aggregation": (
                "rates are recomputed from pooled raw episode counts; "
                "damage and cost are episode means"
            ),
            "decision_time_scope": (
                "action selection only; environment.step state transition is excluded"
            ),
            "decision_trace_scope": (
                "final evaluation only when benchmark.record_decisions is true; "
                "training rollouts and curve evaluations are excluded"
            ),
            "training_seed_source": (
                "benchmark.train_seeds; the training template seed is overwritten "
                "for each run"
            ),
            "timestep_accounting": (
                "requested_timesteps is the configured budget; training_timesteps "
                "is the actual SB3 rollout count"
            ),
            "learning_curve_scenario": (
                "each model is evaluated on its own training scenario"
            ),
            "paired_difference_definition": "method_a - method_b across matched runs",
            "parameter_count_contract": (
                "actor includes policy_net and action_net; critic includes "
                "value branch and value_net; shared parameters are reported separately"
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
    "train_scenario",
    "eval_scenario",
    "run_index",
    "train_seed",
    "evaluation_seed",
    "requested_timesteps",
    "training_timesteps",
    "training_seconds",
    "model_path",
    "actor_parameters",
    "critic_parameters",
    "shared_parameters",
    "total_parameters",
    "mch_engagement_reliability",
    "mch_target_reliability",
    "mch_engagement_residual_abs",
    "mch_target_residual_abs",
    "mch_engagement_gate_active_rate",
    "mch_target_gate_active_rate",
    "mch_engagement_support",
    "mch_target_support",
    "mch_anchor_kl",
    "mch_anchor_penalty",
    "mch_anchor_excess_rate",
    "bpce_probe_selected_count",
    "bpce_probe_accepted_count",
    "bpce_probe_acceptance_rate",
    "bpce_probe_positive_count",
    "bpce_probe_negative_count",
    "bpce_probe_extra_transitions",
    "bpce_probe_selected_mean_abs_delta",
    "bpce_probe_selected_mean_sign_agreement",
    "bpce_probe_effect_pass_rate",
    "bpce_probe_agreement_pass_rate",
    "bpce_probe_selected_mean_informative_repeats",
    "bpce_probe_selected_mean_opposite_repeats",
    "bpce_probe_cumulative_extra_transitions",
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
    "bpce_train_auxiliary_loss",
    "bpce_train_active_label_rate",
    "bpce_train_joint_gradient_norm",
    "bpce_train_auxiliary_gradient_norm",
    "bpce_train_gradient_cosine",
    "bpce_train_cumulative_auxiliary_train_calls",
    "bpce_train_cumulative_mean_auxiliary_loss",
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
    "avg_decision_time_ms",
    "high_threat_leak_rate",
    "avg_zone_weighted_damage",
    "assignment_conflict_rate",
    "overkill_rate",
    "damage_reduction_per_ammo",
    "avg_resource_cost",
    "engagement_rate",
    "actionable_engagement_rate",
    "all_noop_episode_rate",
)

EPISODE_FIELDNAMES = (
    "method",
    "method_type",
    "train_scenario",
    "eval_scenario",
    "run_index",
    "train_seed",
    "evaluation_seed",
    "episode_index",
    "episode_seed",
    "total_reward",
    "steps",
    "num_targets",
    "num_intercepted",
    "num_leaked",
    "total_damage",
    "ammo_used",
    "shots",
    "hits",
    "invalid_actions",
    "unit_decisions",
    "actionable_decisions",
    "engagements",
    "actionable_engagements",
    "all_noop_episode",
    "success",
    "decision_time_seconds",
    "decision_time_ms",
    "high_threat_threshold",
    "num_high_threat_targets",
    "num_high_threat_leaked",
    "high_threat_leak_rate",
    "zone_weighted_damage",
    "engaged_target_events",
    "conflict_target_events",
    "assignment_conflict_rate",
    "overkill_assignments",
    "overkill_rate",
    "intercepted_damage_potential",
    "damage_reduction_per_ammo",
    "resource_cost",
)

DECISION_FIELDNAMES = (
    "method",
    "method_type",
    "train_scenario",
    "eval_scenario",
    "run_index",
    "train_seed",
    "evaluation_seed",
    "episode_index",
    "episode_seed",
    "step_index",
    "unit_index",
    "resource_type",
    "unit_order",
    "unit_order_position",
    "selected_action",
    "selected_target",
    "selected_noop",
    "base_action_legal",
    "conditional_action_legal",
    "ammo_before",
    "cooldown_before",
    "energy_before",
    "num_base_legal_targets",
    "num_conditional_legal_targets",
    "num_conditional_high_threat_targets",
    "prefix_denied_target_count",
    "avoidable_noop",
    "selected_high_threat",
    "target_alive",
    "target_threat",
    "target_payload",
    "target_time_to_impact",
    "target_distance",
    "hit_probability",
    "damage_potential",
    "expected_damage_reduction",
    "best_expected_damage_reduction",
    "matching_efficiency",
    "target_already_selected_by_prefix",
    "shot_fired",
    "hit",
    "target_intercepted",
    "target_status_after",
)

DECISION_SUMMARY_FIELDNAMES = (
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
    "decision_opportunities",
    "assignments",
    "assignment_rate",
    "actionable_decisions",
    "avoidable_noops",
    "avoidable_noop_rate",
    "high_threat_assignments",
    "high_threat_legal_target_opportunities",
    "high_threat_assignment_rate",
    "mean_assigned_threat",
    "mean_expected_damage_reduction",
    "mean_matching_efficiency",
    "prefix_denied_target_opportunities",
    "base_legal_target_opportunities",
    "prefix_denial_rate",
    "collapsed_unit",
)

LEAK_ATTRIBUTION_FIELDNAMES = (
    "method",
    "method_type",
    "train_scenario",
    "eval_scenario",
    "run_index",
    "train_seed",
    "evaluation_seed",
    "episode_index",
    "episode_seed",
    "target_index",
    "target_threat",
    "target_payload",
    "target_class",
    "attribution",
)

LEAK_ATTRIBUTION_SUMMARY_FIELDNAMES = (
    "method",
    "method_type",
    "train_scenario",
    "eval_scenario",
    "run_index",
    "train_seed",
    "attribution",
    "count",
    "total_high_threat_leaks",
    "rate",
)

SUMMARY_FIELDNAMES = (
    "method",
    "method_type",
    "train_scenario",
    "eval_scenario",
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
    "train_scenario",
    "eval_scenario",
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
    "avg_decision_time_ms",
    "high_threat_leak_rate",
    "avg_zone_weighted_damage",
    "assignment_conflict_rate",
    "overkill_rate",
    "damage_reduction_per_ammo",
    "avg_resource_cost",
)

CURVE_SUMMARY_FIELDNAMES = (
    "method",
    "train_scenario",
    "eval_scenario",
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

TRAINING_DYNAMICS_FIELDNAMES = (
    "method",
    "train_scenario",
    "run_index",
    "train_seed",
    "timesteps",
    "policy_loss",
    "value_loss",
    "entropy_loss",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "advantage_mean",
    "advantage_std",
    "positive_advantage_rate",
    "actor_gradient_norm",
    "critic_gradient_norm",
)

PROBE_DYNAMICS_FIELDNAMES = (
    "method",
    "train_scenario",
    "run_index",
    "train_seed",
    "timesteps",
    "probe_scenario",
    "probe_states",
    "actionable_decisions",
    "engage_probability_mean",
    "noop_probability_mean",
    "noop_margin_mean",
    "engagement_entropy_mean",
    "conditional_target_entropy_mean",
    "deterministic_engagement_rate",
    "probe_value_mean",
)

PAIRED_DIFFERENCE_FIELDNAMES = (
    "train_scenario",
    "eval_scenario",
    "method_a",
    "method_b",
    "metric",
    "n_pairs",
    "mean_difference",
    "std_difference",
    "sem_difference",
    "confidence_level",
    "ci_low",
    "ci_high",
)

GENERALIZATION_FIELDNAMES = (
    "method",
    "method_type",
    "train_scenario",
    "eval_scenario",
    "metric",
    "n_runs",
    "mean",
    "std",
    "ci_low",
    "ci_high",
)
