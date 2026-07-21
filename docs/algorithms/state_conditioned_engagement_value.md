# 状态条件资源预算与显式交战双价值

更新时间：2026-07-21  
实现状态：离线正式实验完成，总体门槛通过，逐场景门槛未通过  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`

## 1. 方法动机

固定成本-弹药阈值无法表达目标紧迫度、全局资源余量和替代单元能力。本方法不再回归两个绝对分支效用，而是直接预测成对反事实差值：交战带来的安全收益和增量资源成本。

## 2. 双价值定义

```text
safety_gain
= 30 * (damage_noop - damage_engage)
 + 20 * (high_threat_leak_noop - high_threat_leak_engage)

cost_delta
= resource_cost_engage - resource_cost_noop
```

网络使用完整观测、当前单元、前缀占用、动态合法掩码和冻结 margin logit，输出归一化 `g_hat` 与 `c_hat`。尺度只除不减，保留零点语义。

## 3. 状态条件约束分数

```text
score = g_hat - lambda(s, unit, prefix, mask) * relu(c_hat)
engage iff score > 0
```

`lambda` 经 softplus 保证非负，并限制在 `[0,5]`。同容量消融包括 `lambda=0` 的 safety-only 和单一可学习乘子的 global-budget。

训练联合使用安全收益 SmoothL1、成本差 SmoothL1、类别平衡 BCE、符号 margin 和轻量预算正则。ambiguous 样本只监督价值，不参与分类。

## 4. 交叉拟合

模型选择使用202个历史非 test 上下文的三折 grouped cross-fitting，同一状态不跨折。三个方法共享 folds、训练种子和优化预算。最终方法按可行种子数、平均 OOF BA、安全收益符号和结构简洁度排序；最终训练 epoch 取各折最佳 epoch 的中位数。

OOF 结果：

| seed | safety-only BA | global BA | state BA | state 可行 |
| ---: | ---: | ---: | ---: | --- |
| 20 | 0.741 | 0.745 | 0.778 | 否 |
| 21 | 0.745 | 0.745 | 0.756 | 是 |
| 22 | 0.752 | 0.748 | 0.764 | 是 |

状态条件预算以 `2/3` 可行和最高平均 BA 被选中。

## 5. 正式效果

在72个全新状态上，三种子 BA 为 `0.834/0.776/0.768`，相对零 margin 分别变化 `+0.078/+0.021/-0.013`。no-op recall 提高到 `0.702/0.719/0.737`，wasteful-engage 降至 `0.298/0.281/0.263`；总体资源停止能力已恢复。

预算乘子的均值分别为 `0.955/1.268/2.039`，第三个种子的范围达到 `0.113-4.892`，证明模型确实学习了状态变化的资源价格，而不是退化为固定阈值。

## 6. 剩余边界

逐场景门槛仍未稳定：seed20/21 在 time-pressure 的 no-op recall 分别差 `0.087/0.025`；seed22 在 medium engage 和 heterogeneity no-op 上失败。安全收益相关系数为 `0.49-0.53`，成本相关系数仅 `-0.04-0.13`。当前结构主要依靠安全头、冻结 margin 和分类监督形成边界，cost head 尚未可靠辨识真实增量成本。

因此该模型暂保留为 MCH-PPO 的候选 engagement Critic，但尚未接入 Actor。下一步冻结网络结构，只修订跨场景最坏风险选择和成本差监督；不再增加普通网络容量。

## 7. 实现入口

```text
rein_learning/models/state_conditioned_engagement_value.py
rein_learning/common/state_conditioned_value_training.py
scripts/run_air_defense_v1_task14_state_conditioned_value.py
tests/test_air_defense_v1_task14_state_conditioned_value.py
docs/experiments/air_defense_v1_task14_state_conditioned_value.md
```
