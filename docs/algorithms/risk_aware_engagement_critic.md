# 风险与约束感知的交战效用 Critic

更新时间：2026-07-20  
实现状态：离线诊断完成，未接入 PPO  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`

## 1. 研究动机

均值回报 engagement head 将拦截收益、资源成本、区域毁伤和终止风险压缩为一个期望值。任务十四分层实验已经表明，这种标签不能稳定区分 engage/no-op。本方法先拆分反事实结果，再构造风险和约束感知效用，目标是同时识别：

- 应当交战却选择 no-op；
- 没有安全收益却继续消耗资源。

## 2. 分量化反事实结果

对同一状态、前缀和单元，使用共同随机数运行 no-op 与 engage 两个分支。每次 rollout 保存：

```text
operational_return
resource_cost
damage
high_threat_leaks
total_return
shots
```

其中：

```text
total_return
= operational_return - resource_cost - 30 * damage
```

正式数据的最大重构误差为 `7.63e-06`。

## 3. 风险效用

单次 rollout 的效用为：

```text
U = operational_return
    - lambda_cost * resource_cost
    - lambda_damage * damage
    - lambda_high * high_threat_leaks
```

分支标签采用下尾 CVaR 惩罚：

```text
Q_risk = mean(U) - beta * (mean(U) - lower_CVaR_alpha(U))
```

validation 冻结的配置为：

```text
lambda_cost = 2.0
lambda_damage = 30.0
lambda_high = 0.0
beta = 0.5
alpha = 0.25
```

它表示在保留原毁伤权重的同时，提高资源成本约束，并惩罚最差25%回报尾部。`lambda_high=0` 说明本轮数据中高威胁泄漏附加项没有提供 validation 增益，不代表该风险在任务上不重要。

## 4. 独立判据

效用公式不使用自身作为真值。安全-资源 oracle 定义：

```text
harm = 30 * damage + 20 * high_threat_leaks
```

- engage 显著降低 harm：`oracle_engage`；
- engage 未显著降低 harm 且显著增加 cost：`oracle_noop`；
- 其余情况：ambiguous。

错误被拆成 `false_noop_rate` 与 `wasteful_engage_rate`，分别对应 all-noop 和高成本交战尾部。

## 5. 模型结构

`RiskAwareEngagementCritic` 输入：

- 环境观测；
- 当前单元 one-hot 与单元特征；
- 前缀目标占用；
- 当前动态合法动作掩码。

输出固定为：

```text
[Q_noop, Q_engage]
```

模型包含77,186个参数。mean-return 与 risk-constraint 对照使用完全相同的网络、数据划分、优化器和训练种子，只改变监督标签。

## 6. 正式结果

效用公式本身在独立 test 上表现出正向信号：

| 指标 | 均值回报 | 风险/约束效用 |
| --- | ---: | ---: |
| balanced accuracy | 0.713 | 0.926 |
| false-noop rate | 0.333 | 0.000 |
| wasteful-engage rate | 0.241 | 0.148 |

但二元回归 Critic 没有复现这一优势。三个 risk-constraint 模型的 balanced accuracy 为 `0.398 / 0.435 / 0.435`，正式通过数为 `0/3`。

## 7. 失败边界

正式 test 的57个可靠组中，只有3个 `oracle_engage`，54个为 `oracle_noop`；train 中也只有4个可靠 engage。绝对值回归仍被多数 no-op 与状态共同价值主导，不能据此否定风险效用本身。

因此当前结论是：

> 风险/约束效用提供了有价值的标签方向，但均值回归式 engagement critic 与当前正类功效不足以支撑 PPO 集成。

下一步应采用安全临界状态定向采样、类别平衡的成对符号损失或分布式估值。不得直接恢复 MCH-PPO，也不得转入 GNN。

## 8. 实现入口

```text
rein_learning/models/risk_aware_engagement_critic.py
rein_learning/common/engagement_utility_diagnostics.py
scripts/run_air_defense_v1_task14_engagement_utility.py
scripts/analyze_air_defense_v1_task14_engagement_power.py
tests/test_air_defense_v1_task14_engagement_utility.py
```
