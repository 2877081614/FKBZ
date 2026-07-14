import json

import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")

from rein_learning.envs import (
    AirDefenseV1EnvConfig,
    DefenseUnitV1Config,
    ProtectedZoneConfig,
    TargetV1Config,
)
from rein_learning.experiments import (
    AirDefenseV1BenchmarkConfig,
    RULE_POLICY_FACTORIES,
    run_air_defense_v1_benchmark,
    summarize_rows,
)
from rein_learning.trainers.air_defense_v1_ppo import AirDefenseV1PPOConfig


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

    expected_methods = set(RULE_POLICY_FACTORIES) | {"ppo", "maskable_ppo"}
    assert len(result.run_rows) == len(expected_methods) * 2
    assert {row["method"] for row in result.run_rows} == expected_methods
    assert {row["evaluation_seed"] for row in result.run_rows} == {20, 21}
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
    assert {row["method"] for row in result.curve_rows} == {
        "ppo",
        "maskable_ppo",
    }
    assert {row["train_seed"] for row in result.curve_rows} == {0, 1}
    learning_rows = [
        row for row in result.run_rows if row["method_type"] == "learning"
    ]
    assert all(row["requested_timesteps"] == 16 for row in learning_rows)
    assert all(row["training_timesteps"] == 16 for row in learning_rows)

    for artifact in (
        result.artifacts.config,
        result.artifacts.runs,
        result.artifacts.summary,
        result.artifacts.learning_curves,
        result.artifacts.learning_curve_summary,
    ):
        assert artifact.exists()
        assert artifact.stat().st_size > 0

    assert {path.suffix for path in result.figure_paths} == {".svg", ".pdf", ".png"}
    assert all(path.exists() and path.stat().st_size > 0 for path in result.figure_paths)

    config_record = json.loads(result.artifacts.config.read_text(encoding="utf-8"))
    assert config_record["status"] == "completed"
    assert config_record["benchmark"]["train_seeds"] == [0, 1]
    assert config_record["methods"]["rule"] == list(RULE_POLICY_FACTORIES)
    assert config_record["evaluation_protocol"]["paired_scenario_blocks"] is True
