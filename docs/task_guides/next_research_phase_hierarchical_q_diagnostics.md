# 下一研究阶段：显式交战与目标分层 Q 诊断

更新时间：2026-07-20  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
阶段编号：任务十四·分层诊断  
阶段状态：已完成；目标层通过主要排序门槛，交战层未通过  
阶段主题：`Q_engage`、条件 `Q_target` 与 MCH-PPO 最终离线门控

## 1. 阶段定位

任务十四修订在全新测试集上证明：组内中心化和配对监督可使动作排序相对纯回归平均提高 `0.167`，目标排序达到 `0.696-0.826`；但 engage/no-op 符号准确率仍为 `0.545`。这说明单标量 `Q(s,h_i,a_i)` 仍混合了两个不同决策：

1. 当前单元是否交战；
2. 已决定交战后选择哪个目标。

本阶段用非图结构的显式双层 Q-Critic 分离这两个价值问题。仍不训练 PPO Actor，不实现 GNN。

## 2. 核心研究问题

> 把交战价值与条件目标价值拆成两个监督头后，是否能在独立状态上同时恢复 engage/no-op advantage 符号和合法目标排序？

若两个层级同时通过，才允许进入任务十五最小 MCH-PPO；若目标层通过而交战层失败，则第一创新假设必须继续收窄到交战信用或约束优化，不能直接训练完整算法。

## 3. 冻结价值语义

### 3.1 交战层

对状态 `s`、自回归前缀 `h_i`、当前单元 `i` 和二元交战决策 `e_i`：

```text
Q_engage(s,h_i,e_i=0) = Q(s,h_i,no-op)

Q_engage(s,h_i,e_i=1)
= sum_target pi(target | engage,s,h_i)
  * Q(s,h_i,target)
```

每个 rollout 中先按相同条件目标概率对目标分支回报加权，再与同 rollout 的 no-op 回报配对。局部交战 advantage 为：

```text
A_engage = Q_engage(e=1) - Q_engage(e=0)
```

### 3.2 目标层

仅在 `e_i=1` 且至少存在两个合法目标时：

```text
Q_target(s,h_i,target)
= Q(s,h_i,target)
```

目标层只比较条件合法、未被前缀占用的目标，不包含 no-op。

## 4. 模型结构

实现非图 `HierarchicalMaskedQCritic`：

```text
observation + unit + prefix + legal mask
    -> engagement MLP -> [Q_noop, Q_engage]

observation + unit + target relation + prefix + legal mask
    -> target MLP -> Q_target
```

两个头不共享最后层，避免目标数量和目标差异主导交战输出。目标头沿用任务十四完整 `MaskedActionQCritic` 的输入语义。

## 5. 数据隔离协议

### 5.1 训练与验证

- 只使用任务十四原 `train=338` 行；
- 只使用任务十四原 `validation=117` 行做早停；
- 任务十四原 test 116 行永久排除；
- 任务十四修订的 36 状态测试集永久排除；
- 不使用任何旧测试结果选择损失权重或 epoch。

### 5.2 全新正式测试

| 项目 | 冻结值 |
| --- | ---: |
| 来源策略 | factorized seeds 8、10 |
| 场景 | medium、time_pressure、heterogeneity_pressure |
| 每个 seed × 场景状态数 | 18 |
| 总测试状态数 | 108 |
| 每候选共同随机数 rollout | 32 |
| 测试随机种子 | 291000 |
| 折扣因子 | 0.98 |

采用随机 reservoir 状态抽样，不按交战 advantage 或目标差异筛选。

## 6. 对照方法

### 6.1 单标量基线

加载任务十四修订的 `difference_aware` seeds 14/15/16，在全新测试集上直接推理，并从动作 Q 计算：

- 策略加权 engage/no-op 符号；
- 条件目标排序；
- 条件目标 top-1。

### 6.2 显式分层候选

联合训练两个离线头：

```text
L_engage = MSE(Qe) + centered_MSE(Qe)
L_target = MSE(Qt) + centered_MSE(Qt) + 0.5 * pairwise(Qt)
L_total  = L_engage + L_target
```

验证分数由 engagement 与 target 的绝对 MAE、中心化 MAE之和构成。正式测试不参与早停。

## 7. 诊断指标

### 7.1 交战层

- `Q_engage` MAE/RMSE；
- `A_engage` 符号准确率；
- 高置信 engage/no-op 组数量；
- 按场景和来源策略的符号准确率；
- 预测 advantage 的偏差与标准差。

### 7.2 目标层

- `Q_target` MAE/RMSE；
- pairwise target ranking accuracy；
- target top-1 accuracy；
- 高置信目标对和 top-1 组数量；
- 按场景排序准确率。

### 7.3 工程与效率

- 新旧状态 ID 交集；
- 旧 test 进入训练的行数；
- 两头参数量和推理耗时；
- 相对 32-rollout Monte Carlo 的加速比。

## 8. 预注册门槛

### 8.1 数据门槛

- 旧 test 进入训练/验证的行数为 0；
- 新测试状态与两轮旧测试状态交集为 0；
- 新测试正好 108 个状态、每候选 32 rollout；
- 所有组都恰有一个 no-op，目标条件概率和为 1；
- 不存在跨 `state + unit` 的配对。

### 8.2 科学门槛

| 层级 | 门槛 |
| --- | --- |
| engage 总体 | 符号准确率 >= 0.70，有效组 >= 30 |
| engage 场景 | 每场景准确率 >= 0.60，有效组 >= 10 |
| target 总体 | 排序准确率 >= 0.65，有效对 >= 30 |
| target top-1 | 准确率 >= 0.50，有效组 >= 30 |
| target 场景 | 每场景排序 >= 0.60，有效对 >= 10 |
| 数值非劣 | 两层 MAE 均不超过单标量基线的 1.10 倍 |
| 稳定性 | 3 个训练种子至少 2 个整体通过 |
| 效率 | 双头推理快于 32-rollout Monte Carlo |

附加贡献门槛：

- engage 符号相对同种子单标量基线平均提升 >= 0.10；
- target 排序相对单标量基线平均下降不超过 0.05。

## 9. 决策规则

```text
数据功效是否满足 engage>=30、target>=30、top1>=30？
├─ 否：只输出功效结论，不解锁算法
└─ 是
   └─ Q_engage 与 Q_target 是否至少 2/3 seeds 同时通过？
      ├─ 是：冻结双层价值接口，允许任务十五最小 MCH-PPO 30k 筛选
      └─ 否
         ├─ target 通过、engage 失败：研究交战约束/风险敏感信用
         ├─ engage 通过、target 失败：研究关系表示与目标匹配
         └─ 两者都失败：重新审查反事实标签与 Q 语义
```

任何失败分支都不会自动进入 GNN。

## 10. 交付物

```text
docs/task_guides/next_research_phase_hierarchical_q_diagnostics.md
rein_learning/models/hierarchical_masked_q_critic.py
rein_learning/common/hierarchical_q_diagnostics.py
scripts/run_air_defense_v1_task14_hierarchical_q.py
tests/test_air_defense_v1_task14_hierarchical_q.py
docs/algorithms/hierarchical_masked_q_critic.md
docs/experiments/air_defense_v1_task14_hierarchical_q.md
results/air_defense_v1/task14_hierarchical_q/
```

## 11. 阶段完成定义

满足任一条件即可完成：

- 双层 Critic 通过全部门槛，允许进入任务十五；
- 数据功效达标但仅一个层级通过，定位第一创新剩余瓶颈；
- 两个层级均未通过，形成反事实 Q 语义或模型结构的可靠负结果；
- 正式功效不足，量化下一次独立状态需求并停止算法结论。

## 12. 执行结果

正式实验完成 108 个全新状态、684 个动作候选和每候选 32 次共同随机数 rollout；旧 test 进入训练为 0，上一轮正式观测重叠为 0。

双头 target 排序为 `0.83-0.87`，相对单标量基线平均提高 `0.057`，但 target MAE 恶化约 `17%-21%`，top-1 有效组只有 24。双头 engage 符号为 `0.588-0.706`，有效组17，相对基线平均下降 `0.255`。整体通过种子数为 `0/3`。

本阶段结论为“目标层可学，显式均值型交战头不成立”。不恢复 MCH-PPO，不进入 GNN。下一入口应转向风险敏感或约束感知的交战效用诊断，详见[分层 Q 正式实验报告](../experiments/air_defense_v1_task14_hierarchical_q.md)。
