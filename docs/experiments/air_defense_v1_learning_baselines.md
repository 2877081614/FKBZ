# AirDefenseResourceAssignmentEnv v1.0 学习型 Baseline

更新时间：2026-07-14

## 1. 目标

本阶段目标是把 `AirDefenseResourceAssignmentEnvV1` 从“规则 baseline 可评估”推进到“学习型算法可训练、可对比、可记录”的实验状态。

当前新增两类学习型 baseline：

- `PPO`：普通 Stable-Baselines3 PPO，不使用动作掩码，非法资源-目标动作由环境惩罚。
- `Maskable PPO`：SB3-Contrib Maskable PPO，训练和评估时使用环境提供的 `action_masks()`。

## 2. 运行命令

单独训练 Maskable PPO：

```powershell
conda run -n rein-learning python scripts\train_air_defense_v1_ppo.py --algorithm maskable-ppo --timesteps 20000 --eval-episodes 20
```

同时训练 PPO 和 Maskable PPO：

```powershell
conda run -n rein-learning python scripts\train_air_defense_v1_ppo.py --algorithm both --timesteps 20000 --eval-episodes 20
```

统一对比全部规则 baseline 与学习型 baseline：

```powershell
conda run -n rein-learning python scripts\compare_air_defense_v1_methods.py --timesteps 100000 --eval-episodes 100 --seeds 0 1 2 3 4 --curve-eval-freq 5000
```

统一实验脚本包含以下方法：

- `random_joint`
- `nearest_joint`
- `highest_threat`
- `time_to_impact`
- `greedy_damage`
- `ppo`
- `maskable_ppo`

每个 `run_index` 使用一个训练种子和一块独立的最终评估场景。该场景块在所有方法之间共享，从而形成配对比较；训练曲线使用另一组评估种子，避免反复查看最终测试场景。

说明：当前 PPO 使用 MLP 策略网络，默认 `--device cpu`，这通常比 GPU 更适合小型 SB3 PPO baseline。后续如果接入更大的网络或图神经网络，可以显式传入 `--device cuda`。

快速 smoke run：

```powershell
conda run -n rein-learning python scripts\compare_air_defense_v1_methods.py --timesteps 128 --n-steps 64 --batch-size 32 --eval-episodes 3 --seeds 0 1 --curve-eval-freq 64 --curve-eval-episodes 2 --no-save-models
```

## 3. 输出

每次运行默认创建带时间戳的独立实验目录：

```text
results/air_defense_v1/benchmark_YYYYMMDD_HHMMSS/
```

目录内包含：

```text
experiment_config.json
runs.csv
summary.csv
learning_curves.csv
learning_curve_summary.csv
learning_curves.svg
learning_curves.pdf
learning_curves.png
models/
tensorboard/
```

- `experiment_config.json`：保存环境参数、PPO 超参数、种子、评估协议、软件版本、命令行和产物路径。
- `runs.csv`：每个方法、每个实验种子的一行最终结果。
- `summary.csv`：跨种子均值、样本标准差、标准误和 Student-t 置信区间。
- `learning_curves.csv`：每个学习算法、每个训练种子的定期评估结果。
- `learning_curve_summary.csv`：训练曲线的跨种子统计结果。
- `learning_curves.*`：以平均奖励为主图，并展示成功率、总损伤和非法动作数；阴影为置信区间。

`runs.csv` 同时记录 `requested_timesteps` 和 `training_timesteps`。前者是命令行请求的预算，后者是 SB3 实际执行的 rollout 步数；PPO 必须完成整个 rollout，因此实际值可能略高。

统一指标包括：

- `avg_reward`
- `success_rate`
- `intercept_rate`
- `leak_rate`
- `avg_total_damage`
- `avg_ammo_used`
- `avg_shots`
- `hit_rate_per_shot`
- `avg_invalid_actions`

查看 TensorBoard：

```powershell
conda run -n rein-learning tensorboard --logdir results\air_defense_v1
```

仅验证五个规则 baseline 和统计留档，不训练 PPO：

```powershell
conda run -n rein-learning python scripts\compare_air_defense_v1_methods.py --rules-only --eval-episodes 100 --seeds 0 1 2 3 4
```

## 4. 当前科研含义

本阶段不是为了追求一次训练后取得很高指标，而是为了建立可重复实验闭环：

```text
规则策略
→ PPO
→ Maskable PPO
→ 固定评估种子
→ 同一指标表
→ 保存结果
→ 分析算法失败原因
```

如果 Maskable PPO 明显减少非法动作并提升资源分配效率，说明动作约束建模是一个可继续深化的研究点。

如果 PPO / Maskable PPO 仍然无法超过 `greedy_damage`，则后续重点应转向：

- 奖励权重校准；
- 场景难度分层；
- curriculum learning；
- attention/GNN 资源-目标关系编码；
- PettingZoo / MAPPO 多智能体版本。

## 5. 正式 100k × 5 seeds 实验

正式实验已经完成：

```text
5 个训练种子
100,000 请求训练步数/模型
100 回合最终评估/场景块
5 个规则 baseline + PPO + Maskable PPO
```

主要结果：

- Maskable PPO：平均奖励 `-35.93`，拦截率 `0.561`，平均损伤 `1.052`，非法动作 `0`；
- greedy_damage：平均奖励 `-38.75`，拦截率 `0.517`，平均损伤 `1.052`；
- 普通 PPO：平均奖励 `-86.52`，任务成功率 `0`，非法动作 `6.84`。

Maskable PPO 显著优于普通 PPO，并达到强规则基线水平；但相对 `greedy_damage` 的配对差异置信区间仍跨 0，尚不能声称稳定超过该规则策略。

完整协议、数据表、学习曲线和结论见：

[AirDefenseResourceAssignmentEnv v1.0 正式基准实验（100k × 5 seeds）](air_defense_v1_formal_benchmark_100k.md)
