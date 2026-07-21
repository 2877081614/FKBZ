# 类别平衡的交战符号 Critic

更新时间：2026-07-21  
实现状态：离线正式实验完成，未接入 PPO  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`

## 1. 研究动机

上一阶段的风险效用方向有效，但随机 test 只有3个可靠 engage，连续回归 Critic 被多数 no-op 主导。本方法同时修改数据入口和监督目标：使用决策前安全临界度富集必要交战状态，并直接学习 engage/no-op 的可靠符号。

## 2. 安全临界度采样

每个合法单元-目标关系的临界度为：

```text
criticality
= target_damage_potential
  * hit_probability
  * (1 + target_threat)
  * [1 + 5 / (1 + time_to_impact)]
```

状态取最大关系分数。每层24个状态中，80%选择高临界且同回合至少间隔3步的状态，20%保留时间和风险多样性。该过程不读取未来回报或 oracle。

正式 targeted test 的可靠 engage 比例从历史随机采样的 `3/57=5.3%` 提高到 `28/74=37.8%`，并覆盖三个核心场景，说明定向采样成功解决正类功效问题。

## 3. 类别平衡监督

模型仍为77,186参数的 `RiskAwareEngagementCritic`，输出：

```text
[z_noop, z_engage]
```

对可靠 oracle 标签定义：

```text
logit = z_engage - z_noop
```

正负类别的样本总权重分别固定为0.5，ambiguous 不进入损失。

### 3.1 Balanced BCE

```text
L_bce = weighted_BCE(logit, oracle)
```

### 3.2 Balanced BCE + Margin

```text
L = L_bce + 0.5 * weighted_max(0, 1 - y_sign * logit)
```

margin 直接扩大两类符号间隔。三个训练种子的 validation 平均 balanced accuracy 为：

```text
balanced_bce        = 0.695
balanced_bce_margin = 0.721
```

因此正式候选按 validation 冻结为 `balanced_bce_margin`。

## 4. 正式效果

| seed | 回归 BA | Margin BA | 回归 false-noop | Margin false-noop |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 0.612 | 0.758 | 0.429 | 0.071 |
| 21 | 0.583 | 0.711 | 0.464 | 0.143 |
| 22 | 0.583 | 0.708 | 0.464 | 0.214 |

三种子均达到总体 `BA>=0.70`，相对回归提高 `0.125-0.146`，并大幅降低 false-noop。类别平衡符号学习因此确实解决了回归模型遗漏必要交战的问题。

## 5. 失败边界

候选同时增加了浪费性交战：

```text
seed20: 0.348 -> 0.413
seed21: 0.370 -> 0.435
seed22: 0.370 -> 0.370
```

最明显的失败位于 `time_pressure`：三种子的 engage recall 为 `1.000 / 1.000 / 0.909`，但 no-op recall 只有 `0.455 / 0.182 / 0.273`。模型从“看不见必要交战”转为“能看见交战，但缺少资源约束下的停止边界”。

因此当前结论是：

> 安全临界采样和类别平衡 margin 已恢复交战正类与总体判别能力，但统一零阈值没有同时控制高成本交战。

下一步应在 validation 上研究带 wasteful-engage 约束的阈值或对偶校准，并在全新 test 上验证；这与任务十三对原 PPO 概率做普通统一阈值扫描不同，校准对象是 oracle 监督后的 engagement sign logit。

## 6. 实现入口

```text
rein_learning/common/critical_engagement_sampling.py
rein_learning/common/balanced_engagement_training.py
scripts/run_air_defense_v1_task14_balanced_engagement.py
tests/test_air_defense_v1_task14_balanced_engagement.py
```
