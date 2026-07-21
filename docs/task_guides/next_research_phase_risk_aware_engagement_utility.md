# 下一研究阶段：风险与约束感知的交战效用诊断

更新时间：2026-07-20  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
阶段编号：任务十四·交战效用修订  
阶段状态：已完成；效用层有正向信号，正类功效与估值层未通过  
阶段主题：安全收益、资源代价、尾部风险与 engage/no-op 稳定判别

## 1. 阶段定位

任务十四分层 Q 诊断表明，给定 engage 后的 conditional-target 排序达到 `0.830-0.870`，但显式均值回报 engagement head 的符号准确率仅为 `0.588-0.706`，相对单标量基线平均下降 `0.255`。因此当前瓶颈不是目标关系表示，而是均值回报同时混合任务收益、毁伤风险和资源成本后，无法稳定回答“是否值得交战”。

本阶段只研究交战效用语义和离线估值，不训练 PPO Actor，不修改环境奖励，不实现 GNN。目标是判断风险敏感或显式约束效用能否同时减少两类错误：

1. 应当交战却预测 no-op，对应 all-noop 风险；
2. 没有安全收益却预测 engage，对应高成本交战风险。

## 2. 核心研究问题

> 将任务收益、资源成本、区域毁伤和高威胁泄漏分开记录，并使用尾部风险或约束效用后，是否能在独立状态上比原始均值回报更稳定地判别 engage/no-op？

该问题必须分成两层验证：

- 效用层：候选效用公式与安全-资源反事实判据是否一致；
- 估值层：非图 engagement critic 是否能在未见状态上学习该效用符号。

只有两层同时通过，才允许冻结 MCH-PPO 的 engagement advantage。

## 3. 冻结范围

继续冻结：

- AirDefense v1.0 环境、奖励参数和终止条件；
- `medium`、`time_pressure`、`heterogeneity_pressure` 三个核心场景；
- factorized policy seeds 8/10 与自回归顺序 `012`；
- conditional-target critic 结构和任务十四目标排序结论；
- PPO Actor、Critic、clip、KL、学习率与训练步数；
- GNN、GAT、Transformer 和变规模环境。

本阶段不得根据测试集结果修改效用权重、置信阈值或门控标准。

## 4. 新反事实数据协议

对每个状态 `s`、前缀 `h_i` 和可行动单元 `i`，使用相同环境随机数和策略随机数分别执行：

```text
branch 0: 当前单元固定 no-op，后续单元和后续时刻按冻结策略采样
branch 1: 当前单元按冻结 conditional-target 分布交战，后续过程同上
```

每个分支、每次 rollout 分别记录：

- `operational_return`：排除资源 cost 和 damage 项后的折扣任务回报；
- `resource_cost`：从快照到回合结束的实际资源成本；
- `damage`：从快照到回合结束的新增区域毁伤；
- `high_threat_leaks`：新增高威胁目标泄漏数量；
- `total_return`：环境原始折扣回报，用作冻结基线；
- `shots`：未来合法射击数量。

正式数据使用新的 `eval_seed`，按 `state_id` 在每个“source seed × scenario”层内划分 train/validation/test，比例冻结为 `40%/20%/40%`，为两类罕见反事实保留足够独立测试功效。同一状态不得跨 split；任务十四既有测试状态不得进入本轮训练或测试。

## 5. 效用定义

对分支 `e in {noop, engage}` 的单次 rollout：

```text
U(e) = operational_return(e)
       - lambda_cost * resource_cost(e)
       - lambda_damage * damage(e)
       - lambda_high * high_threat_leaks(e)
```

风险敏感标签为：

```text
Q_risk(e) = mean(U(e))
            - beta * [mean(U(e)) - lower_CVaR_alpha(U(e))]
```

其中 `beta=0` 退化为均值效用。冻结基线对应环境原始语义：

```text
lambda_cost=1, lambda_damage=30, lambda_high=0, beta=0
```

候选网格只能在 validation 上选择。优先级依次为：安全-资源判据 balanced accuracy、最差类别召回率、与冻结基线的参数距离。

## 6. 独立安全-资源判据

判据不使用候选效用值，而直接使用成对结果：

```text
harm = 30 * damage + 20 * high_threat_leaks
```

- `oracle_engage`：engage 的 harm 在置信区间上显著低于 no-op；
- `oracle_noop`：engage 未显著降低 harm，且资源成本显著高于 no-op；
- 其他组：记为 ambiguous，不进入符号准确率，但保留覆盖率统计。

该判据分别对应 all-noop 和高成本交战两种失败尾部，避免用同一个平均奖励给自身打分。

## 7. 模型对照

使用结构完全相同的非图二元 engagement critic：

1. `mean_return`：学习冻结原始总回报；
2. `risk_constraint`：学习 validation 冻结后的风险/约束效用。

两者使用相同 train/validation/test、网络宽度、优化器、训练种子和早停规则。只允许监督标签不同。

## 8. 正式门槛

### 8.1 数据与功效

- 新旧测试状态交集为 0；
- 每个分支至少 32 个共同随机数 rollout；
- test 的 oracle 有效组总数不少于 20；
- 每个核心场景 oracle 有效组不少于 5；
- oracle engage 与 oracle no-op 各不少于 8 组。

### 8.2 效用层

- test balanced accuracy 不低于 `0.70`；
- 相对冻结均值回报提高至少 `0.10`；
- false-noop rate 不高于冻结基线；
- wasteful-engage rate 不高于冻结基线。

### 8.3 估值层

- 3 个训练种子至少 2 个通过；
- oracle balanced accuracy 不低于 `0.70`；
- 相对同种子 mean-return critic 提高至少 `0.10`；
- false-noop 与 wasteful-engage 均不劣于同种子基线；
- 推理耗时低于 32-rollout Monte Carlo。

## 9. 决策规则

```text
功效是否满足？
├─ 否：扩大独立状态，不进入 PPO
└─ 是
   └─ 风险/约束效用是否优于均值回报？
      ├─ 否：重新审查交战效用与环境任务语义
      └─ 是
         └─ engagement critic 是否至少 2/3 seeds 通过？
            ├─ 是：冻结 engagement advantage，允许最小 MCH-PPO 30k
            └─ 否：研究分布式/分类式估值，不进入 PPO
```

任一失败分支均不会自动触发 GNN。

## 10. 交付物

```text
docs/task_guides/next_research_phase_risk_aware_engagement_utility.md
rein_learning/models/risk_aware_engagement_critic.py
rein_learning/common/engagement_utility_diagnostics.py
scripts/run_air_defense_v1_task14_engagement_utility.py
tests/test_air_defense_v1_task14_engagement_utility.py
docs/algorithms/risk_aware_engagement_critic.md
docs/experiments/air_defense_v1_task14_engagement_utility.md
results/air_defense_v1/task14_engagement_utility/
```

## 11. 阶段完成定义

满足任一条件即可结束本阶段：

- 效用层与估值层同时通过，解锁最小 MCH-PPO；
- 功效充足但效用层失败，形成“风险/约束重标仍不足”的可靠负结果；
- 效用层通过但估值层失败，定位为估值分布或分类学习问题；
- 功效不足，量化所需独立状态数并停止算法结论。

## 12. 执行结果

正式实验完成108个全新状态、150个 engage/no-op 上下文组和每分支32次共同随机数 rollout。三轮旧测试观测重叠均为0，状态切分无泄漏，总回报分量重构最大误差为 `7.63e-06`。

validation 冻结配置为 `cost=2.0, damage=30.0, high=0.0, CVaR beta=0.5, alpha=0.25`。在独立 test 上，候选效用相对均值回报把 balanced accuracy 从 `0.713` 提高到 `0.926`，false-noop 从 `0.333` 降至 `0`，wasteful-engage 从 `0.241` 降至 `0.148`。

但57个可靠 test 组中只有3个 oracle-engage、54个 oracle-noop，未达到两类各8组的冻结功效。三种子风险 Critic balanced accuracy 仅为 `0.398 / 0.435 / 0.435`，通过数 `0/3`。因此不恢复 MCH-PPO，不进入 GNN。

功效投影显示，按点估计需要约152个有效 test 组、约261个总状态；95% Wilson 下界对应约760个总状态。下一入口为安全临界状态定向采集与类别平衡的成对符号估值，不进行均匀暴力扩样。详见[正式实验报告](../experiments/air_defense_v1_task14_engagement_utility.md)。
