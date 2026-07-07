"""
评估脚本: 加载训练好的模型, 在指定场景上评估毁伤率
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch

from config.scenario_config import get_scenario, ALL_SCENARIOS
from env.drta_env import HELS_UAV_DRTA_Env
from algorithm.maddpg_ia import MADDPG_IA


def evaluate(scenario_name, ckpt_path, n_episodes=100, device='cuda', seed_offset=0):
    scenario = get_scenario(scenario_name)
    env = HELS_UAV_DRTA_Env(scenario)
    algo = MADDPG_IA(
        n_agents=env.n_hels, n_actions=env.n_actions, n_uavs=env.n_uavs,
        device=device
    )
    algo.load(ckpt_path)
    algo.gumbel_temp = algo.gumbel_min  # evaluation mode

    damage_rates = []
    all_kills = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        terminated, truncated = False, False

        while not (terminated or truncated):
            actions = {}
            for i in range(env.n_hels):
                a, _ = algo.select_action(obs[f'agent_{i}'], evaluate=True)
                actions[f'agent_{i}'] = a
            obs, rewards, terminated, truncated, info = env.step(actions)

        damage_rates.append(info['damage_rate'])
        all_kills.append(sum(info['total_kills']))

    dr_array = np.array(damage_rates)
    mean_dr = np.mean(dr_array)
    std_dr = np.std(dr_array, ddof=1)

    # Bootstrap 95% CI
    bs = [np.mean(np.random.choice(dr_array, size=n_episodes, replace=True))
          for _ in range(10000)]
    ci_low, ci_high = np.percentile(bs, [2.5, 97.5])

    print(f"\n{'='*60}")
    print(f"Evaluation: {scenario_name} ({n_episodes} episodes)")
    print(f"Damage Rate: {mean_dr*100:.2f}% +/- {std_dr*100:.2f}%")
    print(f"95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
    print(f"Mean Kills: {np.mean(all_kills):.1f} / {env.n_uavs}")
    print(f"{'='*60}")

    env.close()
    return mean_dr, std_dr, ci_low, ci_high


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    evaluate(args.scenario, args.ckpt, args.n_episodes, args.device)
