# 下一研究阶段：OOB 安全-停止 Pareto 可行性审计

更新时间：2026-07-22  
执行状态：已完成，主门控通过  
所属阶段：任务十四离线门控修订  
前置任务：多批次临界状态语料与 leave-one-batch-out 泛化

## 1. 阶段背景

上一阶段使用三个独立训练批次和第四个最终隔离批次，证明多批次覆盖能够恢复必要交战识别，但同时把模型推向系统性过度交战。选中目标在 leave-one-batch-out 中仅 `1/3` 种子可行，最终批次为 `0/3`；最终 engage recall 达到 `0.829/0.943/0.886`，no-op recall 仅为 `0.500/0.477/0.545`。

当前需要区分两种机制解释：

1. 连续 score 已经包含可分信息，只是零阈值没有落在跨批次共同可行区间；
2. engage 与 no-op 的 score 排序本身跨批次冲突，不存在任何阈值能够同时满足安全与停止约束。

本阶段只回答上述可行性问题，不新增 rollout，不训练 Actor，不修改环境、奖励、oracle、网络结构或主线研究计划。

## 2. 核心研究问题

对冻结的 OOB 连续 score，是否存在一个批次鲁棒停止边界，使至少2/3模型种子同时满足：

- 总体 balanced accuracy `>=0.70`；
- 总体 engage recall `>=0.60`；
- 总体 no-op recall `>=0.65`；
- 每个留出批次 engage/no-op recall 分别 `>=0.60/0.65`；
- 每个场景 engage/no-op recall 分别 `>=0.60/0.65`；
- safety sign accuracy `>=0.70`。

阈值只允许使用上一阶段 `oob_predictions.csv` 选择。第四批次 `test_group_diagnostics.csv` 及其标签禁止进入扫描、排序和门控。

## 3. 冻结内容

以下内容保持不变：

- AirDefense v1.0 环境、三个核心场景与奖励函数；
- 安全收益/增量成本 oracle 语义；
- 三个训练批次及其 batch_id；
- 模型种子20/21/22；
- 上一阶段冻结选择的 `scenario_robust_reliable_cost` 目标；
- OOB 模型输出与 safety sign 结果；
- 所有召回率、BA和种子通过门槛。

`standard` 和 `scenario_robust` 允许作为机制诊断参照，但不能取代冻结目标形成阶段通过结论。

## 4. 分析方法

### 4.1 完整阈值集合

对每个“目标函数×模型种子”，使用所有唯一 score 的相邻中点以及两端边界构造完整阈值集合。预测规则为：

```text
engage, score > threshold
no-op,  score <= threshold
```

该集合覆盖 score 能产生的全部不同二分类结果，避免依赖任意等距网格。

### 4.2 Pareto 前沿

每个阈值记录：

- overall BA、engage recall、no-op recall；
- 最差批次 engage/no-op recall；
- 最差场景 engage/no-op recall；
- engage 约束余量、no-op 约束余量、BA余量；
- 最小约束余量；
- 是否位于“最差 engage recall－最差 no-op recall”非支配前沿；
- 是否满足全部冻结门槛。

使用最大最小余量选择诊断阈值：先最大化最小约束余量，再比较 BA，最后选择绝对值更接近零的阈值。该阈值用于解释，不得在门槛失败时放宽约束。

### 4.3 三种校准强度

1. `zero threshold`：上一阶段默认边界；
2. `seed-specific robust threshold`：每个种子独立选择跨批次/场景最大最小余量阈值，是主可行性判据；
3. `shared raw threshold`：三个种子共享同一原始 score 阈值，作为更强的尺度一致性诊断，不作为必要通过条件。

## 5. 阶段验收标准

### 5.1 数据与隔离

- 只读取 `oob_predictions.csv`、`experiment_config.json` 和 OOB 门控摘要；
- OOB 行数、批次数、场景数、目标函数和模型种子完整；
- 每个主分析种子在每个批次与场景中均有可靠 engage/no-op 标签；
- 新增 rollout 数为0；
- 最终 test 数据访问标记为 false。

### 5.2 主门控

- 冻结目标至少2/3种子存在满足全部约束的 seed-specific threshold；
- 对每个通过种子，阈值必须同时满足所有批次和场景，而不是只满足 pooled 指标；
- safety sign accuracy 保持 `>=0.70`；
- 阈值和选择规则完整留档，可由 CSV/JSON 复核。

### 5.3 决策规则

若主门控通过：

- 冻结每种子的鲁棒阈值选择规则；
- 允许下一阶段只生成一次全新独立确认批次；
- 独立确认达到至少2/3后才恢复最小 MCH-PPO。

若主门控失败：

- 宣布当前连续 score 不存在足够稳定的跨批次共同边界；
- 停止继续扩大标量阈值、场景权重和随机批次数；
- 下一阶段必须修改价值语义或引入显式安全/资源约束结构；
- MCH-PPO、30k/100k训练和GNN继续冻结。

## 6. 交付物

```text
docs/task_guides/next_research_phase_oob_pareto_feasibility.md
rein_learning/common/pareto_feasibility.py
scripts/run_air_defense_v1_task14_oob_pareto_audit.py
tests/test_air_defense_v1_task14_oob_pareto_audit.py
docs/algorithms/oob_safety_stop_pareto_calibration.md
docs/experiments/air_defense_v1_task14_oob_pareto_audit.md
results/air_defense_v1/task14_oob_pareto_audit/
```

## 7. 本阶段明确不做

- 不生成新仿真状态或 Monte Carlo rollout；
- 不读取上一阶段最终 test 标签进行阈值选择；
- 不训练或修改 PPO Actor；
- 不修改环境、奖励和 oracle；
- 不实现 GNN、GAT 或 Transformer；
- 不因审计结果放宽预注册召回率门槛。

## 8. 执行结果

本阶段于2026-07-22完成。审计复用上一阶段1737行 OOB 预测，其中冻结目标
`scenario_robust_reliable_cost` 在每个模型种子下均有183个可靠标签；没有生成新
rollout，也没有访问最终 test 数据。

零阈值仅 seed21 通过全部约束，可行数为 `1/3`。完整阈值扫描后，seed20、
seed21、seed22 分别存在23、20、2个可行阈值，种子级主门控达到 `3/3`：

| seed | 选定阈值 | BA | 最差批次 engage/no-op | 最差场景 engage/no-op | 最小余量 |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.1052 | 0.798 | 0.750 / 0.675 | 0.679 / 0.724 | 0.0238 |
| 21 | 0.0288 | 0.832 | 0.667 / 0.750 | 0.808 / 0.724 | 0.0238 |
| 22 | 0.3540 | 0.757 | 0.667 / 0.650 | 0.643 / 0.690 | 0.0000 |

结论是：当前连续 score 保留了可用于区分必要交战与合理停止的排序信息，上一阶段
`1/3` 的 OOB 结果主要受零阈值和种子间尺度差异影响，并非价值表示完全失效。
但 seed22 的可行区间仅为 `0.3540-0.3629`，且选定点最小余量为0；共享原始阈值
也只能使2/3种子通过。因此结果属于“有条件可校准”，而不是“边界已经稳定”。

按照预注册决策规则，下一阶段只允许冻结种子级鲁棒阈值选择协议，并生成一次全新
独立确认批次。MCH-PPO、30k/100k训练和GNN仍保持冻结，直至独立确认达到至少
`2/3` 完整通过。
