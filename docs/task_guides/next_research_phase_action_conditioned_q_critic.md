# 下一研究阶段：非图结构掩码条件动作价值 Critic

更新时间：2026-07-19  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
阶段编号：任务十四  
阶段状态：已完成；正式门控未通过  
阶段主题：动作条件价值、动态合法集、反事实排序与 MCH-PPO 恢复门槛

## 1. 阶段定位

任务十三否决了“统一交战阈值即可修复”的解释，同时确认现有 PPO Critic 只输出 `V(s)`，不能区分同一状态下的 `no-op` 和不同目标动作。共同随机数反事实实验出现联合 value-advantage 与局部动作 advantage 方向不一致的样本，但分支方差仍高，不足以直接冻结完整 MCH-PPO。

任务十四先建立一个不使用 GNN 的动作条件 Q-Critic，回答：

> 在动态合法动作和自回归前缀约束下，普通 MLP 是否已经能够稳定预测候选动作价值、排序合法目标，并恢复 engage/no-op 的局部信用方向？

只有 Q-Critic 在独立状态上通过估值、排序和符号门槛，才允许恢复 MCH-PPO 训练候选。

## 2. 核心研究问题

1. `Q(s,h_i,a_i)` 相对 `V(s)` 是否显著降低合法候选价值误差？
2. Q-Critic 是否能正确判断 engage 相对 no-op 的收益符号？
3. Q-Critic 是否能在多个合法目标之间恢复正确排序？
4. 动态合法掩码、前缀占用和候选动作是否都对预测有必要？
5. 估值结果能否跨 `medium / time_pressure / heterogeneity_pressure` 保持稳定？
6. 普通 MLP 的失败是否足以构成后续图反事实 Critic 的进入依据？

## 3. 冻结的动作价值语义

单元按固定顺序 `012` 决策。对状态 `s`、当前单元 `i` 和前序动作 `h_i`：

```text
Q^pi(s,h_i,a_i)
= E[ G_t |
     state=s,
     earlier unit actions=h_i,
     current unit action=a_i,
     later unit actions~pi(.|s,h_i,a_i),
     future actions~pi ]
```

反事实生成必须满足：

- 早于当前单元的动作保持固定；
- 当前单元依次替换为 `no-op` 和每个条件合法目标；
- 晚于当前单元的动作根据修改后的前缀重新采样；
- 后续环境步骤继续使用同一个冻结策略；
- 不同候选使用相同环境随机种子和策略采样种子；
- 非法目标、已被前缀占用的目标不进入候选集合。

任务十三“保持其他所有单元动作不变”的分支只用于诊断，任务十四改为上述自回归 Q 语义。

## 4. 阶段边界

### 4.1 允许内容

- 普通 MLP/关系 MLP 动作条件 Q-Critic；
- 状态、单元、候选目标、no-op、前缀占用和合法掩码特征；
- 冻结策略生成的共同随机数 Monte Carlo 标签；
- 按状态分组的数据切分；
- Q 误差、排序、符号、校准、方差和复杂度诊断；
- `V(s)`、一步回报和简单均值基线；
- 输入消融的离线训练比较。

### 4.2 必须冻结

- AirDefense v1.0 环境、奖励和终止条件；
- 三个核心场景；
- factorized policy 的模型参数与顺序 `012`；
- 环境和策略共同随机数协议；
- 数据集切分与测试集；
- GNN、GAT、Transformer；
- PPO Actor 更新、独立 ratio/clip 和 30k/100k 训练；
- 变规模环境。

## 5. 数据协议

### 5.1 数据来源

使用任务十二 factorized 冻结模型：

- seed 8：低交战/塌缩代表；
- seed 10：高交战代表。

场景：

- `medium`；
- `time_pressure`；
- `heterogeneity_pressure`。

### 5.2 每条样本

```text
state_id, source_model_seed, scenario,
observation,
unit_index,
prefix_occupancy,
candidate_action,
conditional_legal_mask,
policy_probability,
mc_q_mean, mc_q_std, mc_q_se,
one_step_reward_mean,
frozen_v_value
```

同一 `state_id + unit_index` 下至少包含 no-op；存在合法目标时应包含全部条件合法目标。

### 5.3 切分规则

数据必须按 `state_id` 分组切分，不能按候选动作行随机切分：

```text
train / validation / test = 60% / 20% / 20%
```

同一状态的所有单元和候选动作只能出现在一个 split。切分种子在训练前冻结。

## 6. Q-Critic 输入与模型

主模型为非图结构关系 MLP：

```text
observation
+ selected unit features
+ selected target features or no-op zeros
+ unit one-hot
+ candidate-action one-hot
+ prefix occupancy
+ conditional legal mask
+ unit-target relative geometry
-> MLP -> scalar Q
```

该模型允许使用现有 observation layout，但不得加入消息传递、attention 或图卷积。

## 7. 评价指标

### 7.1 数值误差

- MAE；
- RMSE；
- 相对 `V(s)` baseline 的 MAE 降幅；
- 按场景、来源种子、单元和动作类型分组误差。

### 7.2 排序与信用

- 同一状态-单元候选的 pairwise ranking accuracy；
- 最优候选动作 top-1 accuracy；
- engage 相对 no-op 的 advantage 符号准确率；
- 合法目标之间的 pairwise target ranking accuracy；
- 预测反事实 baseline 与 Monte Carlo baseline 的误差。

标签差异小于成对标准误时记为不确定，不进入符号/排序分母。

### 7.3 效率

- 参数量；
- 单样本推理耗时；
- 一次状态全部合法候选估值耗时；
- 相对 Monte Carlo 分支的加速比。

## 8. 预注册门槛

主模型只有同时满足以下条件才通过：

| 类别 | 门槛 |
| --- | --- |
| 数据 | train/validation/test 无 state_id 泄漏 |
| 数值 | 测试 MAE 相对 `V(s)` 至少降低 10% |
| 排序 | 总体 pairwise ranking accuracy >= 0.70 |
| 交战 | engage/no-op 符号准确率 >= 0.70 |
| 目标 | target ranking accuracy >= 0.65 |
| 最优动作 | top-1 accuracy >= 0.50 |
| 场景 | 三核心场景 ranking accuracy 均 >= 0.60 |
| 稳定性 | 3 个训练种子中至少 2 个通过主要门槛 |
| 效率 | Q-Critic 全候选估值快于 16-rollout Monte Carlo |

若有效比较对数量不足 30，则对应门槛记为“证据不足”，不得按通过处理。

## 9. 消融

若主模型通过基础数据和实现检查，至少比较：

- `full`：完整输入；
- `no_prefix`：移除前缀占用；
- `no_mask`：移除条件合法掩码；
- `observation_action_only`：只保留 observation、unit/action one-hot。

消融用于判断动态掩码信息是否真正贡献估值，不用于事后选择主模型门槛。

## 10. 决策规则

```text
Q-Critic 是否优于 V(s) 且通过排序/符号门槛？
├─ 否
│  ├─ 标签方差过高：增加共同随机数或改进监督目标
│  ├─ 数据覆盖不足：扩展独立状态集
│  └─ MLP 表达不足且数据可靠：才允许规划图反事实 Critic
└─ 是
   └─ 冻结 Q 接口和估值器
      └─ 下一阶段实现最小 MCH-PPO 并运行 30k 独立种子筛选
```

不能因为普通 MLP 未通过就自动进入 GNN。必须先排除标签噪声、数据覆盖和训练问题。

## 11. 交付物

```text
rein_learning/models/masked_action_q_critic.py
rein_learning/common/q_critic_diagnostics.py
scripts/run_air_defense_v1_task14_q_critic.py
tests/test_air_defense_v1_task14_q_critic.py
docs/algorithms/masked_action_q_critic.md
docs/experiments/air_defense_v1_task14_q_critic.md
results/air_defense_v1/task14_q_critic/
```

## 12. 阶段完成定义

满足以下任一结果即可完成任务十四：

- Q-Critic 通过全部门槛，允许恢复 MCH-PPO 候选；
- Q 数值误差改善但排序/符号未通过，定位具体失败来源；
- Monte Carlo 标签方差或有效比较数不足，证明当前数据协议不能支撑算法训练；
- 普通 MLP 在可靠数据上稳定失败，形成进入更强关系估值模型的必要证据。

## 13. 执行结果

任务十四已完成代码、测试、正式数据生成、三训练种子门控和四结构消融。正式数据包含 90 个独立状态、571 个合法候选动作样本，每个候选使用 8 次共同随机数 rollout；`state_id` 分组划分不存在泄漏。

完整 Q-Critic 的测试 MAE 为 `10.626-11.295`，相对冻结 `V(s)` 的 `17.747` 改善 `36.4%-40.1%`。但高置信总体排序只有 8 对，准确率为 `0.250-0.375`；目标排序、top-1、engage/no-op 符号和跨场景门槛均未通过。三个训练种子只有 MAE 与效率通过，整体通过数为 `0/3`。

本阶段结论为：Q 数值回归可学，但动作差异尚不可可靠判别。不得恢复 MCH-PPO，不得进入 GNN。下一入口先修订标签覆盖和组内排序监督，详见[任务十四实验报告](../experiments/air_defense_v1_task14_q_critic.md)。
