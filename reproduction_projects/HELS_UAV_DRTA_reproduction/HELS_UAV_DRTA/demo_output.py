"""
实验输出示例生成器: 展示完整实验流程的最终输出格式
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)

# ==============================================================================
# 1. 模拟完整2000-episode训练后的毁伤率统计 (Table 2 格式)
# ==============================================================================
print("=" * 70)
print("输出1: 毁伤率统计表 (论文 Table 2)")
print("=" * 70)
print(f"{'Environment':<25s} {'MADDPG-IA Mean±Std':>25s} {'95% CI':>20s}")
print("-" * 70)

results = {
    'Rural (sunshine)':    (0.9965, 0.0032, 0.9954, 0.9976),
    'Desert (light haze)': (0.7937, 0.0215, 0.7882, 0.7992),
    'Coastal (sunshine)':  (0.9125, 0.0178, 0.9082, 0.9168),
}
for env, (mean, std, ci_low, ci_high) in results.items():
    print(f"{env:<25s} {mean*100:6.2f}% ± {std*100:5.2f}%          "
          f"[{ci_low*100:5.2f}%, {ci_high*100:5.2f}%]")

# ==============================================================================
# 2. 训练曲线 (模拟数据, 论文 Figure 14 格式)
# ==============================================================================
print("\n生成 Figure 14: 训练曲线...")
episodes = np.arange(2000)
np.random.seed(42)

# 模拟: 500 episodes后收敛, 曲线带噪声
base = 10 * (1 - np.exp(-episodes / 200)) - 2
noise = np.random.randn(2000) * 0.5
reward = base + noise
reward_smooth = np.convolve(reward, np.ones(100)/100, mode='same')
reward_std = 2.0 * np.exp(-episodes / 300) + 0.3

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 左图: 5个HELS各自奖励 + 总和 (论文Fig.14样式)
colors = plt.cm.tab10(np.linspace(0, 1, 6))
for i in range(5):
    r_i = reward * (0.5 + 0.5 * np.random.random()) + np.random.randn(2000) * 0.2
    ax1.plot(episodes[::10], r_i[::10], color=colors[i], alpha=0.4, linewidth=0.5)
ax1.plot(episodes[::10], reward_smooth[::10], 'r-', linewidth=2, label='Sum Reward')
ax1.fill_between(episodes, reward_smooth - reward_std, reward_smooth + reward_std,
                 alpha=0.15, color='red')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('Rewards for Each HELS Agent and Sum Rewards (Fig.14)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 右图: 毁伤率曲线
damage_base = 1 - 0.95 * np.exp(-episodes / 300) - 0.05 * np.exp(-episodes / 50)
damage_noise = np.random.randn(2000) * 0.02
damage_rate = np.clip(damage_base + damage_noise, 0, 1)
damage_smooth = np.convolve(damage_rate, np.ones(100)/100, mode='same')

ax2.plot(episodes, damage_rate * 100, 'b-', alpha=0.3, linewidth=0.5)
ax2.plot(episodes, damage_smooth * 100, 'r-', linewidth=2, label='Smoothed')
ax2.axhline(y=99.65, color='g', linestyle='--', label='Final: 99.65%')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Damage Rate (%)')
ax2.set_title('Damage Rate Convergence (Rural, Large Scale)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('figures/fig14_training_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: figures/fig14_training_curve.png")

# ==============================================================================
# 3. 时空态势图 (模拟数据, 论文 Figure 11 格式)
# ==============================================================================
print("生成 Figure 11: 时空态势图...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# HELS positions (5 HELS)
hels_pos = np.array([
    [3000, 0, 100], [1500, 2598, 100],
    [-1500, 2598, 100], [-3000, 0, 100], [0, 0, 100]
])

# 模拟50条UAV轨迹
np.random.seed(123)
for _ in range(50):
    angle = np.random.uniform(0, 2*np.pi)
    start_r = 10000
    end_r = np.random.uniform(0, 3000)
    t_vals = np.linspace(0, 1, 100)
    r_vals = start_r + (end_r - start_r) * t_vals
    x = r_vals * np.cos(angle)
    y = r_vals * np.sin(angle)
    z = np.random.uniform(200, 800)
    # 只画还活着的
    if end_r > 500:
        ax1.plot(x, np.full_like(x, z), 'r-', alpha=0.3, linewidth=0.5)
        ax2.plot(x, y, 'r-', alpha=0.3, linewidth=0.5)

# 标记毁伤决策点 (彩色圆)
kill_angles = np.random.uniform(0, 2*np.pi, 30)
kill_dists = np.random.uniform(2000, 8000, 30)
kill_colors = plt.cm.tab10(np.random.rand(30))
for i, (ka, kd) in enumerate(zip(kill_angles, kill_dists)):
    kx, ky = kd * np.cos(ka), kd * np.sin(ka)
    kz = np.random.uniform(200, 800)
    ax1.scatter(kx, kz, c=[kill_colors[i]], s=30, marker='o', zorder=5)
    ax2.scatter(kx, ky, c=[kill_colors[i]], s=30, marker='o', zorder=5)

# HELS位置
for i, (hx, hy, hz) in enumerate(hels_pos):
    ax1.scatter(hx, hz, marker='D', s=120, c='blue', edgecolors='black',
               linewidth=1.5, zorder=10)
    ax2.scatter(hx, hy, marker='D', s=120, c='blue', edgecolors='black',
               linewidth=1.5, zorder=10)
    label_offset = 300
    ax1.annotate(f'H{i+1}', (hx + label_offset, hz + 50), fontsize=9)
    ax2.annotate(f'H{i+1}', (hx + label_offset, hy + 50), fontsize=9)

# Protected asset
ax1.scatter(0, 0, marker='*', s=300, c='gold', edgecolors='black',
           linewidth=2, zorder=15, label='Protected Asset')
ax2.scatter(0, 0, marker='*', s=300, c='gold', edgecolors='black',
           linewidth=2, zorder=15)

ax1.set_xlabel('X [m]'); ax1.set_ylabel('Z [m]')
ax1.set_title('(a) Front View (X-Z)'); ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3); ax1.set_xlim(-11000, 11000)

ax2.set_xlabel('X [m]'); ax2.set_ylabel('Y [m]')
ax2.set_title('(b) Top View (X-Y)'); ax2.grid(True, alpha=0.3)
ax2.set_xlim(-11000, 11000); ax2.set_ylim(-11000, 11000)

plt.suptitle('Figure 11: Spatial Situation After Decision-Making (5 HELS vs 50 UAVs, Rural)',
             fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig11_spatial.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: figures/fig11_spatial.png")

# ==============================================================================
# 4. 照射时序图 (模拟数据, 论文 Figure 12 格式)
# ==============================================================================
print("生成 Figure 12: 照射时序甘特图...")
fig, ax = plt.subplots(figsize=(14, 5))
colors = plt.cm.tab20(np.linspace(0, 1, 50))

# 模拟5个HELS的照射任务序列
np.random.seed(99)
for h in range(5):
    t_start = np.random.uniform(0, 30)
    n_tasks = np.random.randint(5, 15)
    for _ in range(n_tasks):
        duration = np.random.uniform(1, 8)
        target_id = np.random.randint(0, 50)
        ax.barh(h, duration, left=t_start, height=0.7,
               color=colors[target_id], edgecolor='black', linewidth=0.3,
               alpha=0.85)
        t_start += duration + np.random.uniform(0.5, 2)

ax.set_yticks(range(5))
ax.set_yticklabels([f'HELS {i+1}' for i in range(5)])
ax.set_xlabel('Time [s]')
ax.set_title('Figure 12: Irradiation Timeline of Each HELS Agent')
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(0, 120)
plt.tight_layout()
plt.savefig('figures/fig12_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: figures/fig12_timeline.png")

# ==============================================================================
# 5. 算法对比曲线 (模拟数据, 论文 Figure 16 格式)
# ==============================================================================
print("生成 Figure 16: 算法对比...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

algorithms = ['MADDPG-IA', 'MAPPO', 'QMIX', 'DQN']
alg_rewards = {
    'MADDPG-IA': 10 * (1 - np.exp(-episodes / 200)) + np.random.randn(2000) * 0.3 - 1,
    'MAPPO':      8 * (1 - np.exp(-episodes / 250)) + np.random.randn(2000) * 0.5 - 1,
    'QMIX':       6 * (1 - np.exp(-episodes / 350)) + np.random.randn(2000) * 0.8 - 1,
    'DQN':        3 * (1 - np.exp(-episodes / 500)) + np.random.randn(2000) * 1.0 - 1,
}
linestyles = ['-', '--', '-.', ':']

for (name, rew), ls in zip(alg_rewards.items(), linestyles):
    smooth = np.convolve(rew, np.ones(100)/100, mode='same')
    ax1.plot(episodes[::20], smooth[::20], linewidth=2, linestyle=ls, label=name)

ax1.set_xlabel('Episode'); ax1.set_ylabel('Global Average Reward')
ax1.set_title('(a) Small Scale (2 HELS vs 10 UAVs)')
ax1.legend(); ax1.grid(True, alpha=0.3)

# Large scale: DQN and QMIX fail
ax2.plot(episodes[::20],
         np.convolve(alg_rewards['MADDPG-IA']*0.8, np.ones(100)/100, mode='same')[::20],
         'b-', linewidth=2, label='MADDPG-IA')
ax2.plot(episodes[::20],
         np.convolve(alg_rewards['MAPPO']*0.7, np.ones(100)/100, mode='same')[::20],
         'r--', linewidth=2, label='MAPPO')
ax2.plot(episodes[::20], np.zeros(2000)[::20] + np.random.randn(100)*0.1 - 3,
         'g-.', linewidth=2, label='QMIX (fails)')
ax2.plot(episodes[::20], np.zeros(2000)[::20] + np.random.randn(100)*0.2 - 5,
         'k:', linewidth=2, label='DQN (fails)')

ax2.set_xlabel('Episode'); ax2.set_ylabel('Global Average Reward')
ax2.set_title('(b) Large Scale (5 HELS vs 50 UAVs)')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle('Figure 16: Algorithm Comparison', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig16_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: figures/fig16_comparison.png")

# ==============================================================================
# 6. 输出汇总
# ==============================================================================
print("\n" + "=" * 70)
print("输出汇总: 所有文件列表")
print("=" * 70)
print("""
实验完整运行后, 将产生以下输出:

📁 checkpoints/                     # 模型权重
   └── {scenario}/
       └── run{id}/
           ├── ep500.pt             # 训练500轮时的checkpoint
           ├── ep1000.pt
           ├── ep1500.pt
           ├── ep2000.pt
           └── final.pt             # 最终模型

📁 logs/                             # 训练日志 (JSON)
   └── {scenario}_run{id}.json      # 每轮: reward, damage_rate, losses

📁 runs/                             # TensorBoard日志
   └── {scenario}_{run_id}/
       └── events.out.tfevents.*    # TensorBoard events文件

📁 figures/                          # 可视化图表
   ├── fig11_spatial.png            # 时空态势图 (Fig.11 格式)
   ├── fig12_timeline.png           # 照射时序图 (Fig.12 格式)
   ├── fig14_training_curve.png     # 训练曲线 (Fig.14 格式)
   ├── fig15_ablation.png           # 消融实验 (Fig.15 格式)
   └── fig16_comparison.png         # 算法对比 (Fig.16 格式)

📊 控制台输出 (Table 2 + Table 3 格式):
   毁伤率统计表 (Mean ± Std, 95% CI)
   参数变化实验表
""")

# 文件大小汇总
import glob as _g
fig_files = _g.glob('figures/*.png')
total_size = sum(os.path.getsize(f) for f in fig_files)
print(f"已生成 {len(fig_files)} 个示例图表, 总大小: {total_size/1024:.1f} KB")
for f in sorted(fig_files):
    size_kb = os.path.getsize(f) / 1024
    print(f"  {f} ({size_kb:.1f} KB)")
