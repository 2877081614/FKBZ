from __future__ import annotations

from typing import Any

import numpy as np

from ..agents import DQNConfig, VectorDQNAgent
from ..envs import AirDefenseEnvConfig, AirDefenseResourceAssignmentEnv


def default_air_defense_dqn_config() -> DQNConfig:
    return DQNConfig(
        gamma=0.98,
        learning_rate=3e-4,
        batch_size=64,
        buffer_capacity=20_000,
        min_replay_size=256,
        target_update_interval=100,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        hidden_sizes=(128, 128),
        device="auto",
    )


def train(
    num_episodes: int = 400,
    *,
    env_config: AirDefenseEnvConfig | None = None,
    agent_config: DQNConfig | None = None,
    seed: int = 42,
    log_interval: int = 25,
) -> VectorDQNAgent:
    env = AirDefenseResourceAssignmentEnv(config=env_config)
    config = agent_config or default_air_defense_dqn_config()
    observation_dim = int(np.prod(env.observation_space.shape))
    agent = VectorDQNAgent(
        observation_dim=observation_dim,
        num_actions=env.action_space.n,
        config=config,
        seed=seed,
    )

    episode_rewards: list[float] = []
    losses: list[float] = []
    invalid_action_counts: list[int] = []
    intercept_rates: list[float] = []
    leak_rates: list[float] = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        total_reward = 0.0
        invalid_actions = 0
        info: dict[str, Any] = {}

        while not (terminated or truncated):
            action_mask = env.action_mask()
            action = agent.select_action(state, action_mask)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_action_mask = env.action_mask()

            agent.store_transition(
                state,
                action,
                reward,
                next_state,
                done,
                next_action_mask,
            )
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            if info["invalid_action"]:
                invalid_actions += 1
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        invalid_action_counts.append(invalid_actions)
        intercept_rates.append(info["num_intercepted"] / env.num_targets)
        leak_rates.append(info["num_leaked"] / env.num_targets)

        if log_interval > 0 and (episode == 1 or episode % log_interval == 0):
            recent_rewards = episode_rewards[-log_interval:]
            recent_losses = losses[-log_interval:]
            recent_invalid = invalid_action_counts[-log_interval:]
            recent_intercepts = intercept_rates[-log_interval:]
            recent_leaks = leak_rates[-log_interval:]
            average_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            print(
                f"episode={episode:03d}, "
                f"avg_reward={np.mean(recent_rewards):7.2f}, "
                f"avg_loss={average_loss:.4f}, "
                f"epsilon={agent.config.epsilon:.3f}, "
                f"intercept={np.mean(recent_intercepts):.2f}, "
                f"leak={np.mean(recent_leaks):.2f}, "
                f"invalid={np.mean(recent_invalid):.2f}, "
                f"device={agent.device}"
            )

    env.close()
    return agent


def evaluate(
    agent: VectorDQNAgent,
    *,
    episodes: int = 20,
    env_config: AirDefenseEnvConfig | None = None,
    seed: int = 1_000,
) -> dict[str, float]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    rewards: list[float] = []
    steps: list[int] = []
    successes: list[float] = []
    intercepted: list[int] = []
    leaked: list[int] = []
    targets: list[int] = []
    shots: list[int] = []
    hits: list[int] = []
    invalid_actions: list[int] = []

    for episode_index in range(episodes):
        env = AirDefenseResourceAssignmentEnv(config=env_config)
        state, _ = env.reset(seed=seed + episode_index)
        terminated = False
        truncated = False
        total_reward = 0.0
        episode_steps = 0
        episode_shots = 0
        episode_hits = 0
        episode_invalid = 0
        info: dict[str, Any] = {}

        while not (terminated or truncated):
            action = agent.select_action(state, env.action_mask(), greedy=True)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_steps += 1

            if info["invalid_action"]:
                episode_invalid += 1
            if info["action_type"] == "assign" and not info["invalid_action"]:
                episode_shots += 1
            if info["hit"]:
                episode_hits += 1

        rewards.append(total_reward)
        steps.append(episode_steps)
        successes.append(float(info["num_leaked"] == 0 and info["num_alive"] == 0))
        intercepted.append(int(info["num_intercepted"]))
        leaked.append(int(info["num_leaked"]))
        targets.append(env.num_targets)
        shots.append(episode_shots)
        hits.append(episode_hits)
        invalid_actions.append(episode_invalid)
        env.close()

    return {
        "episodes": float(episodes),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean(steps)),
        "success_rate": float(np.mean(successes)),
        "intercept_rate": float(np.sum(intercepted) / np.sum(targets)),
        "leak_rate": float(np.sum(leaked) / np.sum(targets)),
        "avg_shots": float(np.mean(shots)),
        "hit_rate_per_shot": float(np.sum(hits) / max(1, np.sum(shots))),
        "avg_invalid_actions": float(np.mean(invalid_actions)),
    }


def main() -> None:
    trained_agent = train()
    metrics = evaluate(trained_agent)
    print("Greedy AirDefense DQN evaluation:")
    print(
        f"avg_reward={metrics['avg_reward']:.2f}, "
        f"success={metrics['success_rate']:.2f}, "
        f"intercept={metrics['intercept_rate']:.2f}, "
        f"leak={metrics['leak_rate']:.2f}, "
        f"shots={metrics['avg_shots']:.2f}, "
        f"hit_per_shot={metrics['hit_rate_per_shot']:.2f}, "
        f"invalid={metrics['avg_invalid_actions']:.2f}"
    )


if __name__ == "__main__":
    main()
