from ..agents import DQNAgent, DQNConfig
from ..envs import SmallGridWorldEnv


def train(num_episodes: int = 600) -> DQNAgent:
    env = SmallGridWorldEnv()
    agent = DQNAgent(
        num_states=env.observation_space.n,
        num_actions=env.action_space.n,
        config=DQNConfig(
            gamma=0.95,
            learning_rate=1e-3,
            batch_size=32,
            buffer_capacity=5_000,
            min_replay_size=64,
            target_update_interval=50,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.99,
            hidden_sizes=(64, 64),
            device="auto",
        ),
        seed=42,
    )

    episode_rewards: list[float] = []
    losses: list[float] = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=episode)
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if episode == 1 or episode % 50 == 0:
            average_reward = sum(episode_rewards[-50:]) / len(episode_rewards[-50:])
            average_loss = sum(losses[-50:]) / len(losses[-50:]) if losses else 0.0
            print(
                f"episode={episode:03d}, avg_reward={average_reward:6.2f}, "
                f"avg_loss={average_loss:.4f}, epsilon={agent.config.epsilon:.3f}, "
                f"device={agent.device}"
            )

    env.close()
    return agent


def evaluate(agent: DQNAgent) -> None:
    env = SmallGridWorldEnv(render_mode="ansi")
    state, _ = env.reset(seed=999)
    done = False
    total_reward = 0.0
    steps = 0

    print("Greedy DQN evaluation:")
    print(env.render())

    while not done:
        action = agent.select_action(state, greedy=True)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        print(f"step={steps}, action={info['action_name']}, reward={reward}")
        print(env.render())

    print(f"evaluation_total_reward={total_reward}, steps={steps}")
    env.close()


def main() -> None:
    trained_agent = train()
    evaluate(trained_agent)


if __name__ == "__main__":
    main()
