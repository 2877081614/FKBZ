# 下一研究阶段：状态条件资源预算与显式约束价值

更新时间：2026-07-21  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
阶段状态：已完成；总体门槛通过，逐场景鲁棒性门槛未通过  
阶段定位：进入 MCH-PPO 前的第一项前置任务

## 1. 研究问题

上一阶段证明固定成本-弹药压力虽然方向合理，却无法形成稳定停止边界。原因是“是否值得消耗资源”不仅取决于当前单元成本和弹药，还取决于目标紧迫度、全局剩余资源、未来任务负荷、其他单元替代能力和当前分配前缀。

本阶段检验：将交战的安全收益与增量资源成本分别估值，并让资源乘子随状态变化后，能否在全新状态上同时保持必要交战召回和资源停止能力。

## 2. 冻结内容

- AirDefense v1.0 环境、奖励和三个核心场景；
- factorized source policies seeds `8/10`；
- 安全临界状态采样协议；
- 每分支32次共同随机数 rollout；
- 安全-资源 oracle、置信规则和 ambiguous 定义；
- BCE+margin critics seeds `20/21/22`，只作为冻结输入和基线；
- conditional-target 层及所有旧正式 test；
- MCH-PPO、30k/100k 和 GNN 继续冻结。

本轮正式 test 不得参与模型选择、epoch 选择、特征修改或门槛调整。

## 3. 显式双价值语义

对同一状态、前缀和单元的成对分支定义：

```text
safety_gain
= 30 * (damage_noop - damage_engage)
 + 20 * (high_threat_leak_noop - high_threat_leak_engage)

cost_delta
= resource_cost_engage - resource_cost_noop
```

模型分别输出归一化的：

```text
g_hat = predicted safety_gain
c_hat = predicted incremental cost
```

归一化只除以训练集尺度，不减均值，保证0仍表示“无安全收益”和“无增量成本”。

## 4. 状态条件预算

比较三个同容量候选：

```text
safety_only:
    score = g_hat

global_budget:
    score = g_hat - softplus(lambda_global) * relu(c_hat)

state_budget:
    score = g_hat - softplus(lambda(state, unit, prefix, mask)) * relu(c_hat)
```

最终 `score>0` 判为 engage。状态条件乘子读取完整观测、当前单元、前缀占用、动态合法掩码和冻结 margin logit，不使用未来结果。

训练目标为：

```text
L = SmoothL1(g_hat, safety_gain)
  + SmoothL1(c_hat, cost_delta)
  + balanced_BCE(score, oracle_label)
  + 0.25 * balanced_margin(score, oracle_label)
  + 0.001 * mean(lambda)
```

ambiguous 组参与双价值回归，但不参与分类损失。

## 5. 交叉拟合协议

训练语料只使用上一轮 `analysis_dataset.npz` 的非 test 组。按 `state_id` 和 source/scenario 分层执行三折 grouped cross-fitting：

- 同一状态不得跨 fold；
- 三个方法使用完全相同的 folds、训练种子和优化预算；
- 每折最优 epoch 只由该折验证数据确定；
- 方法族先比较满足逐场景约束的模型种子数，再比较平均 OOF balanced accuracy；
- 最终模型在全部非 test 语料上训练到各折最佳 epoch 的中位数；
- 所有尺度只由对应训练数据计算。

该协议不再重复使用单一 validation 做多轮人工调参。

## 6. 独立测试

正式新增：

```text
2 source policies * 3 scenarios * 12 states = 72 states
```

使用新的评估种子，每分支32次 rollout。新 test 必须与任务十四以来全部正式数据观测重叠为0。

数据功效门槛：可靠组不少于40，engage/no-op 各不少于10，每场景可靠组不少于8，至少两个场景含 engage，总回报分量重构误差不超过 `1e-4`。

## 7. 正式模型门槛

每个模型种子必须同时满足：

- balanced accuracy 不低于 `0.70`；
- engage recall 不低于 `0.60`；
- no-op recall 不低于 `0.65`；
- false-noop 不高于同种子风险回归基线；
- wasteful-engage 不高于同种子风险回归基线；
- 每个含 engage 场景 engage recall 不低于 `0.60`；
- 每个含 no-op 场景 no-op recall 不低于 `0.65`；
- safety gain 的可靠符号准确率不低于 `0.70`；
- 推理耗时低于32-rollout Monte Carlo。

至少2/3模型种子整体通过，才进入 MCH-PPO 最小实现。无论结果如何，本阶段不直接进入 GNN。

## 8. 交付物

```text
docs/task_guides/next_research_phase_state_conditioned_constrained_value.md
rein_learning/models/state_conditioned_engagement_value.py
rein_learning/common/state_conditioned_value_training.py
scripts/run_air_defense_v1_task14_state_conditioned_value.py
tests/test_air_defense_v1_task14_state_conditioned_value.py
docs/algorithms/state_conditioned_engagement_value.md
docs/experiments/air_defense_v1_task14_state_conditioned_value.md
results/air_defense_v1/task14_state_conditioned_value/
```

## 9. 完成定义

代码、测试、交叉拟合、72状态独立实验和结果文档全部完成后结束。通过则下一任务为 MCH-PPO 公式和接口冻结；未通过则依据双价值误差、预算乘子分布和逐场景错误，判断应修订价值估计、预算状态还是 oracle，禁止继续进行无约束阈值搜索。

## 10. 执行结果

三折交叉拟合从 `safety_only/global_budget/state_budget` 中选择了 `state_budget`。其 OOF BA 为 `0.778 / 0.756 / 0.764`，后两个种子满足全部逐场景约束；固定全局预算三个种子均不可行，说明状态条件乘子相对固定资源价格具有明确必要性。

正式实验新增72个状态、97个上下文组和每分支32次 rollout。87个可靠组包含30个 engage 和57个 no-op，三个场景分别有30、29、28个可靠组。旧观测重叠全部为0，总回报重构误差为 `1.53e-05`，数据与功效门槛全部通过。

独立 test 的总体 BA 为 `0.834 / 0.776 / 0.768`，engage recall 为 `0.967 / 0.833 / 0.800`，no-op recall 为 `0.702 / 0.719 / 0.737`，wasteful-engage 降至 `0.298 / 0.281 / 0.263`。三种子均通过总体双类召回、对风险回归双非劣、安全收益符号和推理成本门槛。

完整门槛仍为 `0/3`：seed20/21 的 `time_pressure` no-op recall 为 `0.563/0.625`；seed22 的 `medium` engage recall 为 `0.556`，`heterogeneity_pressure` no-op recall 为 `0.550`。安全收益相关系数约 `0.49-0.53`，但成本预测相关系数仅 `-0.04-0.13`，说明剩余问题集中于跨场景预算泛化和成本价值辨识。

因此不立即恢复 MCH-PPO，也不进入 GNN。下一阶段只需进行一次跨场景鲁棒预算修订：冻结双价值结构，引入逐场景最坏召回或分布鲁棒选择，并改善 cost-delta 监督。若独立测试达到至少2/3种子完整通过，即进入 MCH-PPO 公式与接口冻结。
