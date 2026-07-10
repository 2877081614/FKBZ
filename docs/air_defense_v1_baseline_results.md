# AirDefenseResourceAssignmentEnv v1.0 Baseline Results

更新时间：2026-07-10

## 1. 实验对象

环境：

```text
AirDefenseResourceAssignmentEnvV1
```

核心变化：

- 多保护区域；
- 目标带 `payload`、`target_zone`、`time_to_impact`；
- 防御单元联合动作 `MultiDiscrete`；
- 区域损伤 `total_damage` 进入奖励和终止条件；
- 保留 `action_mask` 与固定随机种子评估。

运行命令：

```powershell
conda run -n rein-learning python scripts\evaluate_air_defense_v1_baselines.py
```

## 2. Baseline 结果

评估设置：

```text
episodes = 50
seed = 200
```

```text
policy              avg_reward  success  intercept  leak   damage  ammo  shots  hit/shot  invalid
random_joint          -54.15     0.08       0.41   0.41     1.39  15.82  15.82      0.14     0.00
nearest_joint         -49.85     0.10       0.42   0.38     1.35  15.70  15.70      0.13     0.00
highest_threat        -60.47     0.04       0.34   0.45     1.51  15.84  15.84      0.11     0.00
time_to_impact        -58.36     0.04       0.36   0.43     1.47  15.86  15.86      0.11     0.00
greedy_damage         -41.22     0.12       0.48   0.34     1.17  15.52  15.52      0.15     0.00
```

## 3. 初步判断

`greedy_damage` 当前表现最好，说明区域价值、目标载荷、命中概率和资源成本组合起来的期望损伤降低指标，比单纯最近目标、最高威胁或最短突防时间更适合作为强规则基线。

但所有 baseline 的成功率仍然较低，最高只有 `0.12`。这说明 v1.0 已经形成比 v0 更强的任务压力：多资源联合分配、低命中概率、有限弹药和区域损伤目标共同造成了明显决策难度。

## 4. 下一步

建议下一步优先做：

1. 实现 PPO / Maskable PPO 在 v1.0 上的训练入口；
2. 将 `random_joint / greedy_damage / PPO / Maskable PPO` 放到统一对比脚本中；
3. 做场景难度敏感性实验，包括目标数量、目标速度、资源数量、保护区域价值和命中概率；
4. 根据 baseline 暴露的问题，判断是否进入图结构动作掩码方法设计。
