"""
可视化脚本: 训练曲线、毁伤率统计、时空态势图
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config.scenario_config import get_scenario
from env.drta_env import HELS_UAV_DRTA_Env
from algorithm.maddpg_ia import MADDPG_IA


def plot_training_curve(log_paths, output_path='training_curve.png',
                        title='Training Curve', window=50):
    """Plot smoothed training curve with +/-1 std"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_rewards = []
    all_damage = []
    for path in log_paths:
        with open(path) as f:
            data = json.load(f)
        all_rewards.append([d['avg_reward'] for d in data])
        all_damage.append([d.get('damage_rate', 0) for d in data])

    min_len = min(len(r) for r in all_rewards)
    rewards = np.array([r[:min_len] for r in all_rewards])
    damage = np.array([d[:min_len] for d in all_damage])

    mean_r = np.mean(rewards, axis=0)
    std_r = np.std(rewards, axis=0)
    mean_d = np.mean(damage, axis=0)
    std_d = np.std(damage, axis=0)

    # Smooth
    def smooth(x, w):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    episodes = np.arange(len(mean_r))
    ax = axes[0]
    ax.plot(episodes, mean_r, 'b-', alpha=0.7, label='Mean')
    ax.fill_between(episodes, mean_r-std_r, mean_r+std_r, alpha=0.2)
    if len(episodes) > window:
        ax.plot(episodes[window-1:], smooth(mean_r, window), 'r-', linewidth=2,
                label=f'Smoothed (w={window})')
    ax.set_xlabel('Episode'); ax.set_ylabel('Average Reward')
    ax.set_title(f'{title} - Reward'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(episodes, mean_d*100, 'g-', alpha=0.7, label='Mean')
    ax.fill_between(episodes, (mean_d-std_d)*100, (mean_d+std_d)*100, alpha=0.2)
    if len(episodes) > window:
        ax.plot(episodes[window-1:], smooth(mean_d, window)*100, 'r-', linewidth=2)
    ax.set_xlabel('Episode'); ax.set_ylabel('Damage Rate (%)')
    ax.set_title(f'{title} - Damage Rate'); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_spatial_situation(env, output_path='spatial.png'):
    """Plot front+top view of current situation (Fig.11)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, env.n_hels))

    # Front view (X-Z)
    for i, pos in enumerate(env.hels_positions):
        ax1.scatter(pos[0], pos[2], marker='D', s=120, c=[colors[i]],
                   edgecolors='black', label=f'HELS {i+1}', zorder=5)
    for u in env.uavs:
        if u.alive:
            ax1.scatter(u.position[0], u.position[2], marker='o', s=20,
                       alpha=0.6, c='red')
    ax1.scatter(0, 0, marker='*', s=300, c='gold', edgecolors='black',
               label='Protected', zorder=10)
    ax1.set_xlabel('X [m]'); ax1.set_ylabel('Z [m]')
    ax1.set_title('Front View (X-Z)'); ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3); ax1.axis('equal')

    # Top view (X-Y)
    for i, pos in enumerate(env.hels_positions):
        ax2.scatter(pos[0], pos[1], marker='D', s=120, c=[colors[i]],
                   edgecolors='black', zorder=5)
    for u in env.uavs:
        if u.alive:
            ax2.scatter(u.position[0], u.position[1], marker='o', s=20,
                       alpha=0.6, c='red')
    ax2.scatter(0, 0, marker='*', s=300, c='gold', edgecolors='black', zorder=10)
    ax2.set_xlabel('X [m]'); ax2.set_ylabel('Y [m]')
    ax2.set_title('Top View (X-Y)'); ax2.grid(alpha=0.3); ax2.axis('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='training',
                        choices=['training', 'spatial'])
    parser.add_argument('--log', type=str, nargs='+', default=[])
    parser.add_argument('--scenario', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.mode == 'training' and args.log:
        out = args.output or 'figures/training_curve.png'
        plot_training_curve(args.log, out)

    elif args.mode == 'spatial' and args.scenario:
        scenario = get_scenario(args.scenario)
        env = HELS_UAV_DRTA_Env(scenario)
        env.reset(seed=42)
        # Run a few random steps
        for _ in range(20):
            acts = {f'agent_{i}': env.action_space[f'agent_{i}'].sample()
                    for i in range(env.n_hels)}
            obs, rew, term, trunc, info = env.step(acts)
        out = args.output or 'figures/spatial_situation.png'
        plot_spatial_situation(env, out)
        env.close()
