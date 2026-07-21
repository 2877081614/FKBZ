# 下一研究阶段：跨场景鲁棒预算与可靠成本差监督

更新时间：2026-07-21  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
阶段状态：已完成；鲁棒候选未被交叉拟合选中，正式门控未通过  
阶段定位：进入 MCH-PPO 前的最后一项离线门控任务

## 1. 阶段问题

状态条件双价值已经使三种子总体 BA 达到 `0.768-0.834`，总体 engage/no-op recall 和双错误非劣全部通过。剩余失败只位于逐场景局部边界：两个种子在 `time_pressure` 过度交战，一个种子在 `medium` 偏保守并在 `heterogeneity_pressure` 偏激进。

本阶段检验：不增加模型容量，只通过场景-类别鲁棒训练、最坏场景选择和可靠成本差监督，能否消除这种局部漂移。

## 2. 冻结内容

- `StateConditionedEngagementValue` 网络结构、输入和三种训练种子；
- safety gain、cost delta 和状态条件乘子的数学语义；
- BCE+margin 冻结输入、环境、奖励、oracle 和临界采样；
- source policies seeds `8/10` 与三个核心场景；
- 每分支32次共同随机数 rollout；
- 三折 grouped cross-fitting 和全部旧正式 test；
- MCH-PPO、30k/100k、GNN 和额外网络容量继续冻结。

禁止根据本轮 test 修改场景权重、可靠性范围或通过门槛。

## 3. 候选训练目标

比较三个完全同结构的 state-budget 候选：

```text
standard:
    上一阶段的全局类别平衡 BCE + margin

scenario_robust:
    每个非空 (scenario, oracle_class) 块总权重相等
    + 0.5 * 最大场景-类别块损失

scenario_robust_reliable_cost:
    scenario_robust
    + 成对 rollout 可靠性加权 cost-delta SmoothL1
```

最大块损失使优化器不能只依靠样本较多或更容易的场景提高平均指标。

## 4. 成本差可靠性

对每个上下文的成对成本差样本：

```text
cost_delta_samples = cost_engage - cost_noop
SE = std(cost_delta_samples) / sqrt(num_rollouts)
reliability = abs(mean(cost_delta_samples)) / (1.645 * SE)
```

可靠性裁剪到 `[0.25, 4.0]`，再归一化为均值1。零方差且非零差值取上界，零方差且零差值取下界。该权重只影响成本回归，不改变安全收益和 oracle 标签。

## 5. 模型选择

继续使用202个历史非 test 上下文和相同三折 grouped cross-fitting。排序键冻结为：

1. 满足全部逐场景门槛的模型种子数；
2. 三种子的平均最坏场景-类别召回；
3. 平均 OOF balanced accuracy；
4. 平均成本相关系数；
5. 平局时选择更简单的 standard。

最终 epoch 仍取各折最佳 epoch 的中位数，不读取正式 test。

## 6. 独立测试

正式新增：

```text
2 source policies * 3 scenarios * 12 states = 72 states
```

使用新评估种子、每分支32次 rollout，并与所有旧正式观测检查零重叠。功效门槛沿用上一阶段：可靠组不少于40，engage/no-op 各不少于10，每场景不少于8个可靠组，至少两个场景含 engage。

## 7. 正式验收

每个候选种子必须同时满足：

- overall BA `>=0.70`；
- overall engage recall `>=0.60`；
- overall no-op recall `>=0.65`；
- 每个含 engage 场景 engage recall `>=0.60`；
- 每个含 no-op 场景 no-op recall `>=0.65`；
- false-noop 与 wasteful-engage 均不高于风险回归；
- safety sign accuracy `>=0.70`；
- 推理快于32-rollout Monte Carlo。

阶段级附加门槛：

- 至少2/3种子完整通过；
- 候选三种子的平均 worst-scenario recall 不低于冻结 state-budget 基线；
- 候选平均 BA 相对冻结 state-budget 的下降不超过 `0.02`；
- 候选平均 cost correlation 不低于冻结基线。

## 8. 决策

若全部阶段门槛通过，立即冻结 engagement Critic、预算接口和损失，下一任务进入 MCH-PPO 公式/接口实现，不再增加离线估值阶段。若仅成本相关性失败但逐场景策略门槛通过，可冻结策略接口并将 cost-value 标记为辅助估值限制；若逐场景仍失败，则不得进入 MCH-PPO，需审查场景条件表示或 oracle 跨场景一致性。

无论结果如何，本阶段不进入 GNN。

## 9. 交付物

```text
docs/task_guides/next_research_phase_cross_scenario_robust_budget.md
rein_learning/common/state_conditioned_value_training.py
scripts/run_air_defense_v1_task14_cross_scenario_robust_value.py
tests/test_air_defense_v1_task14_cross_scenario_robust_value.py
docs/algorithms/cross_scenario_robust_engagement_value.md
docs/experiments/air_defense_v1_task14_cross_scenario_robust_value.md
results/air_defense_v1/task14_cross_scenario_robust_value/
```

## 10. 执行结果

三折 OOF 中，standard、scenario-robust、reliable-cost 的可行种子数分别为 `2/3、2/3、1/3`。三种子的平均最差场景召回以 standard 更高，因此按冻结规则保留 standard；鲁棒损失没有形成跨种子一致改进，可靠成本加权也未稳定提高 cost correlation。

正式实验新增72个状态、88个上下文组和每分支32次 rollout。81个可靠组包含38个 engage 和43个 no-op，三个场景分别有26、26、29个可靠组。旧观测重叠全部为0，总回报重构误差为 `7.63e-06`，数据与功效全部通过。

由于 OOF 选择 standard，新候选与冻结 state-budget 在 test 上完全一致。三种子 BA 为 `0.758 / 0.662 / 0.634`，完整通过数 `0/3`。seed20 仅 heterogeneity no-op recall `0.611` 未达标；seed21/22 的 heterogeneity engage recall 降至 `0.273 / 0.182`，且安全收益符号准确率降至 `0.675 / 0.675`。

新 test 的 engage 比例为 `46.9%`，高于上一批的 `34.5%`。同一场景内部的临界状态组成发生变化后，模型由“过度交战”转为两个种子的“异质场景漏交战”。因此剩余问题不是单一场景平均权重，而是跨临界状态批次的分布漂移与安全收益表示不稳定。

本阶段不恢复 MCH-PPO，也不进入 GNN。下一阶段必须建立多独立批次训练/校准和 leave-one-batch-out 门控，验证模型不是只适配某一批安全临界状态；不得继续增加场景权重或在本轮 test 上调参。
