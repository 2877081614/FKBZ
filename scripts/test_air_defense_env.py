from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.envs import AirDefenseResourceAssignmentEnv


def run_random_legal_episode(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    env = AirDefenseResourceAssignmentEnv(render_mode="ansi")
    obs, info = env.reset(seed=seed)
    print(f"reset obs_shape={obs.shape}, info={info}")
    print(env.render())

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        mask = env.action_mask()
        legal_actions = np.flatnonzero(mask)
        action = int(rng.choice(legal_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"step={info['current_step']}, action={action}, "
            f"reward={reward:.2f}, terminated={terminated}, truncated={truncated}"
        )
        print(f"info={info}")

    print(f"total_reward={total_reward:.2f}")
    print(env.render())
    env.close()


if __name__ == "__main__":
    run_random_legal_episode()
