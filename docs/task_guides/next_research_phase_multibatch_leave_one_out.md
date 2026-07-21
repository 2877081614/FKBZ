# 下一研究阶段：多批次临界状态语料与留一批次泛化

更新时间：2026-07-21  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
阶段状态：已完成；批次功效通过，OOB与最终门控未通过  
阶段定位：MCH-PPO 前的批次外泛化门控

## 1. 研究问题

同一 state-budget 模型在两批独立临界状态上的异质场景 engage recall 从 `1.0` 下降到 `0.273/0.182`。普通按状态随机交叉拟合虽然隔离了 state_id，但训练折和验证折仍共享同一次批次生成机制，不能检验临界状态子分布变化。

本阶段检验：使用多个独立采样批次训练，并整批留出验证后，安全收益符号和状态预算能否在未见批次上稳定泛化。

## 2. 冻结内容

- AirDefense v1.0、奖励、oracle、临界度采样和共同随机数协议；
- source policies seeds `8/10` 与三个核心场景；
- `StateConditionedEngagementValue` 网络结构和输入；
- safety gain、cost delta 与状态条件预算语义；
- 模型 seeds `20/21/22`、优化器范围和32 rollout；
- 所有旧正式 test 继续冻结，只用于观测重叠审计；
- MCH-PPO、GNN 和网络扩容继续冻结。

旧 test 不得转为训练数据。

## 3. 专用多批次语料

独立生成三个训练批次：

```text
每批次：2 source policies * 3 scenarios * 8 states = 48 states
三个批次：144 states
每个分支：32 paired rollouts
```

每批使用不同环境/策略采样种子，状态 ID 显式包含 `batch_id`。每批必须满足：

- 恰好48个状态；
- 可靠组不少于30；
- engage 与 no-op 各不少于6；
- 每场景可靠组不少于5；
- 与全部旧正式 test 和其他训练批次观测重叠为0。

## 4. Leave-One-Batch-Out

三次训练分别留出一个完整批次：

```text
train batch B/C -> validate A
train batch A/C -> validate B
train batch A/B -> validate C
```

同一留出批次中的所有 source policy、场景和状态都不进入对应训练折。比较：

```text
standard:
    全局类别平衡 BCE+margin

batch_robust:
    每个 (batch, scenario, class) 块等权 + 最差块惩罚

batch_robust_reliable_cost:
    batch_robust + 成本差可靠性加权
```

选择顺序：可行模型种子数、平均最差留出批次/场景召回、平均 BA、平均 cost correlation、结构简洁度。

## 5. 留批次可行门槛

每个模型种子的 pooled OOB 预测必须满足：

- overall BA `>=0.70`；
- engage recall `>=0.60`；
- no-op recall `>=0.65`；
- 每个留出批次 engage recall `>=0.60`、no-op recall `>=0.65`；
- 每个场景 engage recall `>=0.60`、no-op recall `>=0.65`；
- safety sign accuracy `>=0.70`。

至少2/3种子可行，才允许在三个批次全集上训练最终候选。

## 6. 第四批次独立确认

最终 test 使用第四个从未参与训练或选择的批次：

```text
2 source policies * 3 scenarios * 12 states = 72 states
每分支32 paired rollouts
```

数据和功效门槛沿用前序正式实验，并要求与三个训练批次及全部旧 test 观测重叠为0。

每个模型种子的正式门槛：

- overall BA `>=0.70`；
- overall engage/no-op recall 分别 `>=0.60/0.65`；
- 每场景 engage/no-op recall 分别 `>=0.60/0.65`；
- false-noop 与 wasteful-engage 不高于风险回归；
- safety sign accuracy `>=0.70`；
- 推理快于32-rollout Monte Carlo。

阶段通过还要求：至少2/3种子完整通过，平均最差场景召回不低于冻结 state-budget，平均 BA 下降不超过0.02。

## 7. 决策

若 leave-one-batch-out 和第四批次确认均通过，则冻结 engagement Critic、批次协议和预算接口，下一任务直接进入 MCH-PPO 公式与实现。若 OOB 通过而最终批次失败，说明三批次覆盖仍不足；若 OOB 本身失败，则审查安全收益表示和采样子分布，不得继续训练 Actor。

本阶段失败不自动触发 GNN；只有失败与资源-目标关系表示或变规模泛化明确相关时才重新评估图结构。

## 8. 交付物

```text
docs/task_guides/next_research_phase_multibatch_leave_one_out.md
scripts/run_air_defense_v1_task14_multibatch_leave_one_out.py
tests/test_air_defense_v1_task14_multibatch_leave_one_out.py
docs/algorithms/multibatch_engagement_value_generalization.md
docs/experiments/air_defense_v1_task14_multibatch_leave_one_out.md
results/air_defense_v1/task14_multibatch_leave_one_out/
```

## 9. 执行结果

三个专用训练批次全部通过数据与功效门槛，共144个状态、193个上下文组。三个批次可靠组为 `63/61/59`，engage 为 `24/21/23`，no-op 为 `39/40/36`；批次间及与旧正式数据的观测重叠全部为0。

leave-one-batch-out 选择 `scenario_robust_reliable_cost`，但其可行种子数只有 `1/3`。三个种子的 pooled OOB BA 为 `0.815/0.819/0.817`，总体指标较高；最差留出批次召回为 `0.650/0.667/0.475`，最差场景召回为 `0.621/0.690/0.586`。seed22 的批次外局部召回仍明显失稳。

第四批次正式 test 包含72个状态、87个上下文和79个可靠组，其中35 engage、44 no-op。候选三种子 engage recall 为 `0.829/0.943/0.886`，但 no-op recall 只有 `0.500/0.477/0.545`；三个场景 no-op recall 均低于0.65，完整通过数 `0/3`。

多批次语料解决了上一批 seed21/22 的必要交战漏判，却再次出现系统性过度交战。平均 BA `0.697` 高于冻结基线 `0.671`，但平均最差场景召回由 `0.419` 降至 `0.378`。因此多批次覆盖有效改善安全方向，却未形成稳定的安全-资源 Pareto 边界。

本阶段不恢复 MCH-PPO，也不进入 GNN。下一步不应继续扩充随机批次，而应使用已生成的 OOB 预测进行预注册 Pareto 可行性审计：验证是否存在统一的批次鲁棒停止边界，在保持每批次/场景 engage recall 的同时恢复 no-op recall。只有 OOB 存在可行边界，才允许在全新最终批次确认。
