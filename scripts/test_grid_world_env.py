from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.envs import SmallGridWorldEnv


def run_fixed_episode() -> None:
    env = SmallGridWorldEnv(render_mode="ansi")
    obs, info = env.reset(seed=42)
    print(f"reset obs={obs}, info={info}")
    print(env.render())

    actions = [1, 1, 2, 2, 1, 1, 2, 2]
    total_reward = 0.0

    for step_id, action in enumerate(actions, start=1):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"step={step_id}, action={action}, obs={obs}, reward={reward}, "
            f"terminated={terminated}, truncated={truncated}, info={info}"
        )
        print(env.render())
        if terminated or truncated:
            break

    print(f"total_reward={total_reward}")
    env.close()


if __name__ == "__main__":
    run_fixed_episode()
