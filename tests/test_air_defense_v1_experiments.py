import json
from dataclasses import replace

import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")

from rein_learning.envs import (
    AirDefenseV1EnvConfig,
    DefenseUnitV1Config,
    ProtectedZoneConfig,
    TargetV1Config,
)
from rein_learning.common import aggregate_air_defense_v1_episode_metrics
from rein_learning.experiments import (
    AirDefenseV1BenchmarkConfig,
    DEFAULT_LEARNING_METHODS,
    LEARNING_METHODS,
    RULE_POLICY_FACTORIES,
    run_air_defense_v1_benchmark,
    summarize_rows,
)
from rein_learning.trainers.air_defense_v1_ppo import AirDefenseV1PPOConfig
import rein_learning.experiments.air_defense_v1_benchmark as benchmark_module


def make_tiny_experiment_env_config() -> AirDefenseV1EnvConfig:
    return AirDefenseV1EnvConfig(
        protected_zones=(
            ProtectedZoneConfig(
                position=(0.0, 0.0),
                radius=2.0,
                value=1.0,
            ),
        ),
        defense_units=(
            DefenseUnitV1Config(
                resource_type="missile",
                position=(10.0, 0.0),
                ammo=2,
                max_range=100.0,
                base_hit_probability=1.0,
                cost=0.0,
            ),
        ),
        targets=(
            TargetV1Config(
                position=(10.0, 0.0),
                speed=0.0,
                threat=1.0,
                target_zone=0,
                payload=1.0,
            ),
        ),
        max_steps=3,
    )


def make_tiny_experiment_train_config() -> AirDefenseV1PPOConfig:
    return AirDefenseV1PPOConfig(
        total_timesteps=16,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        net_arch=(16,),
        verbose=0,
        device="cpu",
    )


def test_summarize_rows_uses_student_t_confidence_interval() -> None:
    rows = [
        {"method": "ppo", "method_type": "learning", "avg_reward": 1.0},
        {"method": "ppo", "method_type": "learning", "avg_reward": 3.0},
    ]

    summary = summarize_rows(
        rows,
        group_keys=("method", "method_type"),
        metrics=("avg_reward",),
        confidence_level=0.95,
    )

    assert len(summary) == 1
    assert summary[0]["n_runs"] == 2
    assert summary[0]["mean"] == pytest.approx(2.0)
    assert summary[0]["std"] == pytest.approx(2.0**0.5)
    assert summary[0]["ci_low"] < 1.0
    assert summary[0]["ci_high"] > 3.0


def test_multi_seed_benchmark_writes_complete_reproducibility_bundle(
    tmp_path,
) -> None:
    protocol = AirDefenseV1BenchmarkConfig(
        train_seeds=(0, 1),
        eval_episodes=1,
        eval_seed=20,
        curve_eval_freq=8,
        curve_eval_episodes=1,
        curve_eval_seed=100,
        save_models=False,
        create_plot=True,
    )

    result = run_air_defense_v1_benchmark(
        output_dir=tmp_path / "benchmark",
        benchmark_config=protocol,
        train_config=make_tiny_experiment_train_config(),
        env_config=make_tiny_experiment_env_config(),
    )

    expected_methods = set(RULE_POLICY_FACTORIES) | set(DEFAULT_LEARNING_METHODS)
    assert len(result.run_rows) == len(expected_methods) * 2
    assert {row["method"] for row in result.run_rows} == expected_methods
    assert {row["evaluation_seed"] for row in result.run_rows} == {20, 21}
    assert len(result.episode_rows) == len(expected_methods) * 2
    assert {row["method"] for row in result.episode_rows} == expected_methods
    assert all("assignment_conflict_rate" in row for row in result.episode_rows)
    assert all("resource_cost" in row for row in result.episode_rows)
    for run_index in (0, 1):
        paired_rows = [row for row in result.run_rows if row["run_index"] == run_index]
        assert len({row["evaluation_seed"] for row in paired_rows}) == 1

    reward_summary = [
        row
        for row in result.summary_rows
        if row["metric"] == "avg_reward"
    ]
    assert len(reward_summary) == len(expected_methods)
    assert all(row["n_runs"] == 2 for row in reward_summary)
    assert {row["method"] for row in result.curve_rows} == set(
        DEFAULT_LEARNING_METHODS
    )
    assert {row["train_seed"] for row in result.curve_rows} == {0, 1}
    learning_rows = [
        row for row in result.run_rows if row["method_type"] == "learning"
    ]
    assert all(row["requested_timesteps"] == 16 for row in learning_rows)
    assert all(row["training_timesteps"] == 16 for row in learning_rows)

    for artifact in (
        result.artifacts.config,
        result.artifacts.runs,
        result.artifacts.episodes,
        result.artifacts.decisions,
        result.artifacts.decision_summary,
        result.artifacts.leak_attributions,
        result.artifacts.leak_attribution_summary,
        result.artifacts.summary,
        result.artifacts.paired_differences,
        result.artifacts.generalization_matrix,
        result.artifacts.learning_curves,
            result.artifacts.learning_curve_summary,
            result.artifacts.model_parameter_counts,
    ):
        assert artifact.exists()
        assert artifact.stat().st_size > 0

    assert {path.suffix for path in result.figure_paths} == {".svg", ".pdf", ".png"}
    assert all(path.exists() and path.stat().st_size > 0 for path in result.figure_paths)

    config_record = json.loads(result.artifacts.config.read_text(encoding="utf-8"))
    assert config_record["status"] == "completed"
    assert config_record["schema_version"] == 8
    assert config_record["benchmark"]["train_seeds"] == [0, 1]
    assert config_record["methods"]["rule"] == list(RULE_POLICY_FACTORIES)
    assert config_record["evaluation_protocol"]["paired_scenario_blocks"] is True
    assert result.paired_difference_rows
    assert result.generalization_rows

    random_run = next(
        row
        for row in result.run_rows
        if row["method"] == "random_joint" and row["run_index"] == 0
    )
    random_episodes = [
        row
        for row in result.episode_rows
        if row["method"] == "random_joint" and row["run_index"] == 0
    ]
    reaggregated = aggregate_air_defense_v1_episode_metrics(random_episodes)
    for metric_name, value in reaggregated.items():
        assert random_run[metric_name] == pytest.approx(value)


def test_conflict_free_benchmark_records_discrete_space_and_zero_conflicts(
    tmp_path,
) -> None:
    protocol = AirDefenseV1BenchmarkConfig(
        train_seeds=(0,),
        eval_episodes=2,
        eval_seed=40,
        curve_eval_freq=8,
        curve_eval_episodes=1,
        curve_eval_seed=200,
        methods=("conflict_free_maskable_ppo",),
        save_models=False,
        create_plot=False,
    )

    result = run_air_defense_v1_benchmark(
        output_dir=tmp_path / "conflict_free",
        benchmark_config=protocol,
        train_config=make_tiny_experiment_train_config(),
        env_config=make_tiny_experiment_env_config(),
    )

    assert len(result.run_rows) == 1
    assert result.run_rows[0]["avg_invalid_actions"] == 0.0
    assert result.run_rows[0]["assignment_conflict_rate"] == 0.0
    assert result.run_rows[0]["overkill_rate"] == 0.0
    assert all(row["conflict_target_events"] == 0 for row in result.episode_rows)
    assert all(row["overkill_assignments"] == 0 for row in result.episode_rows)

    config_record = json.loads(result.artifacts.config.read_text(encoding="utf-8"))
    signature = config_record["methods"]["space_signatures"][
        "conflict_free_maskable_ppo"
    ]
    assert signature["action_space"] == {"type": "Discrete", "n": 2}
    assert signature["action_generator"]["type"] == "enumerated_conflict_free"


def test_autoregressive_benchmark_records_generator_and_zero_conflicts(
    tmp_path,
) -> None:
    protocol = AirDefenseV1BenchmarkConfig(
        train_seeds=(0,),
        eval_episodes=2,
        eval_seed=50,
        curve_eval_freq=8,
        curve_eval_episodes=1,
        curve_eval_seed=300,
        methods=("autoregressive_maskable_ppo",),
        save_models=False,
        create_plot=False,
    )

    result = run_air_defense_v1_benchmark(
        output_dir=tmp_path / "autoregressive",
        benchmark_config=protocol,
        train_config=make_tiny_experiment_train_config(),
        env_config=make_tiny_experiment_env_config(),
    )

    assert len(result.run_rows) == 1
    assert result.run_rows[0]["avg_invalid_actions"] == 0.0
    assert result.run_rows[0]["assignment_conflict_rate"] == 0.0
    assert result.run_rows[0]["overkill_rate"] == 0.0

    config_record = json.loads(result.artifacts.config.read_text(encoding="utf-8"))
    signature = config_record["methods"]["space_signatures"][
        "autoregressive_maskable_ppo"
    ]
    assert signature["action_space"] == {"type": "MultiDiscrete", "nvec": [2]}
    assert signature["action_generator"] == {
        "type": "autoregressive_conflict_free",
        "unit_order": [0],
        "conditional_target_mask": True,
        "joint_log_prob": "sum_of_conditional_log_probs",
        "environment_steps_per_joint_action": 1,
    }


def test_order_ablation_records_decisions_with_environment_unit_indices(
    tmp_path,
) -> None:
    base_config = make_tiny_experiment_env_config()
    three_unit_config = replace(
        base_config,
        defense_units=(
            base_config.defense_units[0],
            base_config.defense_units[0],
            base_config.defense_units[0],
        ),
    )
    protocol = AirDefenseV1BenchmarkConfig(
        train_seeds=(0,),
        eval_episodes=2,
        eval_seed=60,
        curve_eval_freq=8,
        curve_eval_episodes=1,
        curve_eval_seed=400,
        methods=("autoregressive_ppo_order_120",),
        save_models=False,
        create_plot=False,
        record_decisions=True,
    )

    result = run_air_defense_v1_benchmark(
        output_dir=tmp_path / "order_120",
        benchmark_config=protocol,
        train_config=make_tiny_experiment_train_config(),
        env_config=three_unit_config,
    )

    expected_decisions = sum(
        int(row["steps"]) * 3 for row in result.episode_rows
    )
    assert len(result.decision_rows) == expected_decisions
    assert result.decision_summary_rows
    assert {row["unit_order"] for row in result.decision_rows} == {"1-2-0"}
    position_by_unit = {
        int(row["unit_index"]): int(row["unit_order_position"])
        for row in result.decision_rows
    }
    assert position_by_unit == {0: 2, 1: 0, 2: 1}
    assert all(row["episode_index"] in {0, 1} for row in result.decision_rows)
    assert result.artifacts.decisions.exists()
    assert result.artifacts.decision_summary.exists()

    config_record = json.loads(result.artifacts.config.read_text(encoding="utf-8"))
    signature = config_record["methods"]["space_signatures"][
        "autoregressive_ppo_order_120"
    ]
    assert signature["action_generator"]["unit_order"] == [1, 2, 0]
    assert config_record["result_counts"]["decision_rows"] == expected_decisions


def test_role_conditioned_benchmark_records_layout_parameters_and_order(
    tmp_path,
) -> None:
    base_config = make_tiny_experiment_env_config()
    three_unit_config = replace(
        base_config,
        defense_units=(
            base_config.defense_units[0],
            base_config.defense_units[0],
            base_config.defense_units[0],
        ),
    )
    protocol = AirDefenseV1BenchmarkConfig(
        train_seeds=(0,),
        eval_episodes=1,
        eval_seed=70,
        curve_eval_freq=8,
        curve_eval_episodes=1,
        curve_eval_seed=500,
        methods=("role_conditioned_ar_ppo_order_201",),
        save_models=True,
        create_plot=False,
        record_decisions=True,
    )

    result = run_air_defense_v1_benchmark(
        output_dir=tmp_path / "role_order_201",
        benchmark_config=protocol,
        train_config=make_tiny_experiment_train_config(),
        env_config=three_unit_config,
    )

    run = result.run_rows[0]
    assert run["actor_parameters"] > 0
    assert run["critic_parameters"] > 0
    assert run["shared_parameters"] == 0
    assert run["total_parameters"] == (
        run["actor_parameters"] + run["critic_parameters"]
    )
    assert run["avg_invalid_actions"] == 0.0
    assert run["assignment_conflict_rate"] == 0.0
    assert run["overkill_rate"] == 0.0
    assert {row["unit_order"] for row in result.decision_rows} == {"2-0-1"}
    assert {
        int(row["unit_index"]): int(row["unit_order_position"])
        for row in result.decision_rows
    } == {0: 1, 1: 2, 2: 0}

    config_record = json.loads(result.artifacts.config.read_text(encoding="utf-8"))
    signature = config_record["methods"]["space_signatures"][
        "role_conditioned_ar_ppo_order_201"
    ]["action_generator"]
    assert signature["type"] == "role_conditioned_autoregressive_conflict_free"
    assert signature["unit_order"] == [2, 0, 1]
    assert signature["actor_head"]["unit_index_embedding"] is False
    assert signature["observation_layout"]["num_units"] == 3

    parameter_record = json.loads(
        result.artifacts.model_parameter_counts.read_text(encoding="utf-8")
    )
    assert parameter_record["schema_version"] == 1
    assert len(parameter_record["models"]) == 1
    assert parameter_record["models"][0]["actor_parameters"] == run[
        "actor_parameters"
    ]

def test_rules_cross_scenario_matrix_uses_paired_evaluation_blocks(tmp_path) -> None:
    protocol = AirDefenseV1BenchmarkConfig(
        train_seeds=(0, 1),
        eval_episodes=2,
        eval_seed=30,
        train_scenarios=("easy", "medium"),
        eval_scenarios=("easy", "hard"),
        methods=("greedy_damage", "hungarian_damage"),
        include_learning=False,
        save_models=False,
        create_plot=False,
    )

    result = run_air_defense_v1_benchmark(
        output_dir=tmp_path / "cross_scenario",
        benchmark_config=protocol,
    )

    assert len(result.run_rows) == 2 * 2 * 2 * 2
    assert {row["train_scenario"] for row in result.run_rows} == {
        "easy",
        "medium",
    }
    assert {row["eval_scenario"] for row in result.run_rows} == {"easy", "hard"}
    assert {row["method"] for row in result.run_rows} == {
        "greedy_damage",
        "hungarian_damage",
    }
    for run_index in (0, 1):
        for eval_scenario in ("easy", "hard"):
            paired_rows = [
                row
                for row in result.run_rows
                if row["run_index"] == run_index
                and row["eval_scenario"] == eval_scenario
            ]
            assert len({row["evaluation_seed"] for row in paired_rows}) == 1

    assert result.paired_difference_rows
    assert result.generalization_rows
    assert result.artifacts.paired_differences.exists()
    assert result.artifacts.generalization_matrix.exists()


def test_benchmark_rejects_unknown_methods_and_alias_duplicates() -> None:
    with pytest.raises(ValueError, match="Unknown benchmark methods"):
        AirDefenseV1BenchmarkConfig(methods=("unknown",))

    with pytest.raises(ValueError, match="unique canonical scenarios"):
        AirDefenseV1BenchmarkConfig(train_scenarios=("medium", "default"))


def test_scenario_dimension_mismatch_fails_before_training() -> None:
    medium = benchmark_module.get_air_defense_v1_scenario_profile("medium").config
    incompatible = replace(medium, num_random_targets=6)

    with pytest.raises(ValueError, match="Incompatible AirDefense v1 scenario spaces"):
        benchmark_module._validate_scenario_space_compatibility(
            {"medium": medium},
            {"six_targets": incompatible},
        )
