"""
训练主脚本 (Section 6, Algorithm 1)
Usage: python train.py --scenario small_rural --run_id 0
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
from tqdm import tqdm

from config.scenario_config import get_scenario, list_scenarios
from config.hyperparams import MAX_EPISODES
from env.drta_env import HELS_UAV_DRTA_Env
from algorithm.maddpg_ia import MADDPG_IA
from utils.logger import Logger


def train(scenario_name, run_id=0, device='cuda',
          max_episodes=MAX_EPISODES, log_dir='runs',
          save_dir='checkpoints', save_interval=500):
    scenario = get_scenario(scenario_name)
    print(f"[Train] {scenario_name} | {scenario['n_hels']} HELS vs {scenario['n_uavs']} UAVs | {scenario['env_type']}")
    print(f"[Train] Device: {device}, Max Episodes: {max_episodes}")

    env = HELS_UAV_DRTA_Env(scenario)
    algo = MADDPG_IA(
        n_agents=env.n_hels, n_actions=env.n_actions, n_uavs=env.n_uavs,
        device=device
    )
    logger = Logger(log_dir, scenario_name, run_id)

    episode_rewards = []
    damage_rates = []

    pbar = tqdm(range(max_episodes), desc=scenario_name)
    for episode in pbar:
        obs, info = env.reset()
        ep_reward = np.zeros(env.n_hels)
        terminated, truncated = False, False
        step = 0

        while not (terminated or truncated):
            # Select actions
            actions = {}
            for i in range(env.n_hels):
                a, _ = algo.select_action(obs[f'agent_{i}'])
                actions[f'agent_{i}'] = a

            # Step environment
            next_obs, rewards, terminated, truncated, info = env.step(actions)

            # Compute mixed rewards
            mixed_rewards = {}
            for i in range(env.n_hels):
                rh = algo.compute_mixed_reward(next_obs[f'agent_{i}'],
                                               rewards[f'agent_{i}'])
                mixed_rewards[f'agent_{i}'] = rh
                ep_reward[i] += rh

            # Store experience
            algo.replay_buffer.push(obs, actions, mixed_rewards, next_obs,
                                    terminated or truncated)

            # Update
            update_info = algo.update()

            obs = next_obs
            step += 1

        # Episode summary
        avg_reward = np.mean(ep_reward)
        damage_rate = info.get('damage_rate', 0.0)
        episode_rewards.append(avg_reward)
        damage_rates.append(damage_rate)

        metrics = {
            'avg_reward': avg_reward,
            'damage_rate': damage_rate,
            'total_kills': sum(info['total_kills']),
            'battery_mean': np.mean(info['battery_remaining']),
            'steps': step,
            'time': info['time'],
        }
        if update_info:
            metrics.update(update_info)

        logger.log_episode(episode, metrics)

        pbar.set_postfix({
            'rew': f'{avg_reward:.2f}',
            'dmg': f'{damage_rate:.2%}',
            'kills': f'{metrics["total_kills"]}',
        })

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            ckpt_path = os.path.join(save_dir, scenario_name, f'run{run_id}',
                                     f'ep{episode+1}.pt')
            algo.save(ckpt_path)

    # Save final model and log
    final_path = os.path.join(save_dir, scenario_name, f'run{run_id}', 'final.pt')
    algo.save(final_path)
    log_path = os.path.join('logs', f'{scenario_name}_run{run_id}.json')
    logger.save(log_path)
    logger.close()
    env.close()

    print(f"[Train] Complete. Final damage rate: {damage_rates[-1]:.4f}")
    return episode_rewards, damage_rates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, required=True,
                        choices=list_scenarios())
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--episodes', type=int, default=MAX_EPISODES)
    args = parser.parse_args()
    train(args.scenario, args.run_id, args.device, args.episodes)
