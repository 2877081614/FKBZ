from __future__ import annotations

from ..agents import REINFORCEAgent, REINFORCEConfig, REINFORCETrajectoryStep
from ..envs import SmallGridWorldEnv


def train(num_episodes: int = 500) -> REINFORCEAgent:
    env = SmallGridWorldEnv()
    agent = REINFORCEAgent(
        num_states=env.observation_space.n,
        num_actions=env.action_space.n,
        config=REINFORCEConfig(
            gamma=0.95,
            learning_rate=0.025,
            hidden_sizes=(32,),
            normalize_returns=True,
            device="auto",
        ),
        seed=42,
    )

    episode_rewards: list[float] = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=episode)
        done = False
        total_reward = 0.0
        trajectory: list[REINFORCETrajectoryStep] = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory.append(
                REINFORCETrajectoryStep(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
            )
            state = next_state
            total_reward += reward

        loss = agent.update_episode(trajectory)
        episode_rewards.append(total_reward)

        if episode == 1 or episode % 50 == 0:
            average_reward = sum(episode_rewards[-50:]) / len(episode_rewards[-50:])
            print(
                f"episode={episode:03d}, avg_reward={average_reward:6.2f}, "
                f"loss={loss:.4f}, device={agent.device}"
            )

    env.close()
    return agent


def evaluate(agent: REINFORCEAgent) -> None:
    env = SmallGridWorldEnv(render_mode="ansi")
    state, _ = env.reset(seed=999)
    done = False
    total_reward = 0.0
    steps = 0

    print("Greedy REINFORCE evaluation:")
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
