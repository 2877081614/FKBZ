from rein_learning.agents import DQNConfig, VectorDQNAgent
from rein_learning.trainers.air_defense_dqn import evaluate, train


def test_air_defense_dqn_trainer_runs_short_training() -> None:
    agent = train(
        num_episodes=2,
        agent_config=DQNConfig(
            gamma=0.95,
            learning_rate=1e-3,
            batch_size=2,
            buffer_capacity=100,
            min_replay_size=2,
            target_update_interval=2,
            epsilon=0.5,
            epsilon_min=0.1,
            epsilon_decay=0.9,
            hidden_sizes=(16,),
            device="cpu",
        ),
        seed=0,
        log_interval=0,
    )

    assert isinstance(agent, VectorDQNAgent)


def test_air_defense_dqn_evaluate_returns_metrics() -> None:
    agent = train(
        num_episodes=1,
        agent_config=DQNConfig(
            batch_size=2,
            buffer_capacity=100,
            min_replay_size=2,
            hidden_sizes=(16,),
            device="cpu",
        ),
        seed=1,
        log_interval=0,
    )

    metrics = evaluate(agent, episodes=1, seed=10)

    assert metrics["episodes"] == 1.0
    assert 0.0 <= metrics["intercept_rate"] <= 1.0
    assert 0.0 <= metrics["leak_rate"] <= 1.0
    assert metrics["avg_invalid_actions"] == 0.0
