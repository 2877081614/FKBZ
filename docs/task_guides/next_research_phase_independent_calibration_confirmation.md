# 下一研究阶段：冻结 OOB 校准协议的独立批次确认

更新时间：2026-07-22
执行状态：已完成，主门控未通过
所属阶段：任务十四离线门控最终确认
前置任务：OOB 安全-停止 Pareto 可行性审计

## 1. 阶段目的

上一阶段证明冻结的 `scenario_robust_reliable_cost` 连续 score 在历史 OOB 数据上存在
安全-停止共同可行边界：零阈值仅 `1/3` 种子通过，种子级鲁棒阈值达到 `3/3`。
但 seed22 可行区间很窄，且三个种子的原始 score 尺度明显不同。

本阶段只回答一个问题：**完全冻结 OOB 选择结果后，鲁棒阈值能否在一批全新、独立、
不参与任何选择的数据上保持至少2/3种子完整可行？**

## 2. 冻结协议

在生成确认批次前冻结：

- 环境：AirDefenseResourceAssignmentEnv v1.0；
- 场景：`medium`、`time_pressure`、`heterogeneity_pressure`；
- source seeds：8、10；
- 模型目标：`scenario_robust_reliable_cost`；
- 模型种子：20、21、22；
- 最终模型检查点：多批次阶段在全部三个训练批次上拟合的模型；
- OOB 种子级阈值：`0.105205 / 0.028757 / 0.354024`；
- 分类规则：`score > threshold` 为 engage；
- 全部召回率、BA、安全符号和种子通过门槛。

确认数据生成后，不允许重新扫描阈值、切换目标、重训模型或调整约束。

## 3. 唯一确认批次

```text
eval_seed: 887000
states: 2 source seeds × 3 scenarios × 12 = 72
episodes_per_stratum: 30
rollouts_per_branch: 32
gamma: 0.98
```

使用与前序正式实验相同的 targeted critical-state 生成和共同随机数反事实 rollout
协议。该批次只生成一次；无论结果通过或失败，都不得追加随机种子直到得到有利结果。

## 4. 数据完整性与功效门槛

- 状态数等于72，每分支 rollout 数等于32；
- 与历史正式训练、校准和测试 `test_dataset.npz` 的观测重叠均为0；
- 总回报重构最大误差不超过 `1e-4`；
- 可靠标签总数不少于40；
- engage 与 no-op 各不少于10；
- 每个场景可靠组不少于8；
- 每个场景同时具有 engage 和 no-op，确保逐场景双类召回可判定；
- 模型、阈值文件和输入配置记录 SHA-256。

任一数据或功效门槛失败时，本批次只判为“不可判定”，不得据此放行 MCH-PPO。

## 5. 每种子完整门控

在新批次上重新计算：

```text
balanced accuracy >= 0.70
overall engage recall >= 0.60
overall no-op recall >= 0.65
每场景 engage recall >= 0.60
每场景 no-op recall >= 0.65
safety sign accuracy >= 0.70
推理时间 < rollout 数据生成时间
```

由于只有一个独立确认批次，批次级召回与总体召回相同。零阈值结果只作为诊断，不能
替代冻结 OOB 阈值形成阶段结论。

## 6. 阶段决策

若数据与功效门槛全部通过，且冻结阈值下至少2/3模型种子完整通过：

- 宣布非图 engagement estimator 已通过当前离线安全-停止门控；
- 允许下一阶段冻结 MCH-PPO 公式、接口和理论假设；
- 允许随后运行最小 `30k × 3 seeds` MCH-PPO 筛选；
- 暂不直接运行100k正式实验，也不进入GNN。

若少于2/3种子完整通过：

- 不得使用本确认批次重新选阈值；
- MCH-PPO继续冻结；
- 根据失败方向决定是增加显式约束结构，还是重新定义跨种子可校准的价值尺度；
- GNN仍不得因本阶段失败自动启动。

## 7. 交付物

```text
docs/task_guides/next_research_phase_independent_calibration_confirmation.md
scripts/run_air_defense_v1_task14_independent_confirmation.py
tests/test_air_defense_v1_task14_independent_confirmation.py
docs/experiments/air_defense_v1_task14_independent_confirmation.md
results/air_defense_v1/task14_independent_confirmation/
```

## 8. 本阶段明确不做

- 不重新训练 engagement value 模型；
- 不读取确认标签后修改阈值；
- 不增加第二个确认批次；
- 不修改环境、奖励、oracle 或临界状态采样协议；
- 不训练 PPO/MCH-PPO；
- 不实现 GNN、GAT 或 Transformer。

## 9. 执行结果

本阶段于2026-07-22完成。唯一确认批次使用冻结的 `eval_seed=887000`，生成72个
全新状态、87个上下文组和每分支32次 rollout。81个可靠组包含35个 engage 和46个
no-op；三个场景均同时包含双类别。与19个历史数据集的观测重叠全部为0，总回报重构
最大误差为 `7.63e-06`。数据完整性与功效门槛全部通过。

冻结 OOB 阈值下结果为：

| seed | 阈值 | BA | engage recall | no-op recall | 最差场景 engage/no-op | safety sign | 通过 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 20 | 0.1052 | 0.625 | 0.771 | 0.478 | 0.636 / 0.333 | 0.740 | 否 |
| 21 | 0.0288 | 0.646 | 0.857 | 0.435 | 0.727 / 0.333 | 0.740 | 否 |
| 22 | 0.3540 | 0.625 | 0.686 | 0.565 | 0.364 / 0.467 | 0.753 | 否 |

完整通过数为 `0/3`，低于预注册的 `2/3`。seed20/21 主要表现为跨场景过度
交战；seed22 的高阈值改善部分 no-op，却使异质场景 engage recall 降至 `0.364`，同时
仍未达到停止门槛。零阈值同样为 `0/3`。

因此上一阶段的 OOB 可行性只能解释为历史三批次内部存在可调边界，不能解释为阈值
已经跨批次稳定。确认批次不用于回调阈值，也不追加第二确认批次。MCH-PPO继续冻结，
下一研究入口转为跨批次 score 尺度对齐、不确定性感知或显式安全-资源约束，而不是继续
增加标量阈值和随机数据。GNN仍不自动启动。
