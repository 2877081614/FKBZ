# AirDefense v1 多批次 Leave-One-Batch-Out 实验

更新时间：2026-07-21  
实验状态：正式实验完成  
结论状态：批次功效通过，OOB仅1/3可行，最终0/3通过

## 1. 实验目的

检验状态条件 engagement Critic 在完全未见的临界状态采样批次上是否稳定，并区分单批次偶然性与真实跨批次泛化。

## 2. 数据规模

| 项目 | 数值 |
| --- | ---: |
| 专用训练批次 | 3 |
| 训练状态 | 144 |
| 训练上下文组 | 193 |
| 最终 test 状态 | 72 |
| 最终 test 上下文 | 87 |
| 每分支 rollout | 32 |
| 总数据生成时间 | 1436.53 s |
| 旧数据、批次间和最终-训练重叠 | 全部0 |
| 回报重构最大误差 | `7.63e-06` |

三个训练批次均通过独立功效门槛。最终 test 有79个可靠组，其中35 engage、44 no-op；medium、time-pressure、heterogeneity 分别有25、29、25个可靠组。

## 3. Leave-One-Batch-Out

选中的 `scenario_robust_reliable_cost`：

| seed | OOB BA | engage | no-op | worst batch | worst scenario | safety sign | 可行 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | 0.815 | 0.838 | 0.791 | 0.650 | 0.621 | 0.724 | 否 |
| 21 | 0.819 | 0.838 | 0.800 | 0.667 | 0.690 | 0.724 | 是 |
| 22 | 0.817 | 0.912 | 0.722 | 0.475 | 0.586 | 0.718 | 否 |

selected OOB 通过数为 `1/3`，未达到进入最终模型确认所需的2/3。实验仍继续执行第四批次，只作为预注册失败边界审计，不改变选择。

## 4. 最终批次总体结果

| seed | 方法 | BA | engage recall | no-op recall | false-noop | wasteful-engage |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20 | frozen state-budget | 0.687 | 0.829 | 0.545 | 0.171 | 0.455 |
| 20 | multibatch | 0.664 | 0.829 | 0.500 | 0.171 | 0.500 |
| 21 | frozen state-budget | 0.652 | 0.600 | 0.705 | 0.400 | 0.295 |
| 21 | multibatch | 0.710 | 0.943 | 0.477 | 0.057 | 0.523 |
| 22 | frozen state-budget | 0.675 | 0.600 | 0.750 | 0.400 | 0.250 |
| 22 | multibatch | 0.716 | 0.886 | 0.545 | 0.114 | 0.455 |

多批次候选显著提高seed21/22的必要交战召回，却同步降低 no-op recall。三种子完整通过数为 `0/3`。

## 5. 候选逐场景 no-op recall

| seed | medium | time-pressure | heterogeneity |
| ---: | ---: | ---: | ---: |
| 20 | 0.333 | 0.588 | 0.583 |
| 21 | 0.333 | 0.529 | 0.583 |
| 22 | 0.467 | 0.588 | 0.583 |

所有九个“种子×场景”组合都低于0.65。失败已从某一场景局部漂移变成一致的过度交战边界。

## 6. 阶段门控

| 门控 | 结果 |
| --- | --- |
| 三个训练批次功效 | 全部通过 |
| 最终 test 数据与功效 | 全部通过 |
| selected OOB 至少2/3可行 | 失败：1/3 |
| 最终 test 至少2/3通过 | 失败：0/3 |
| 平均 BA 相对冻结基线非劣 | 通过：0.697 vs 0.671 |
| 平均最差场景召回非劣 | 失败：0.378 vs 0.419 |
| 恢复 MCH-PPO | 否 |
| 进入 GNN | 否 |

## 7. 下一步

现有多批次数据已经足够用于无新增 rollout 的 OOB Pareto 审计。下一步先扫描冻结 OOB score 的安全-资源前沿，并要求阈值只由 OOB 选择：

- 检查每个种子是否存在同时满足逐批次/逐场景双类召回的边界；
- 比较统一阈值、种子内阈值和显式双约束校准；
- 若OOB无可行解，停止阈值路线并修改价值/约束结构；
- 若至少2/3存在可行解，再生成一批最终确认数据；
- 本轮最终 test 不参与调参。

## 8. 结果入口

```text
results/air_defense_v1/task14_multibatch_leave_one_out/training_dataset.npz
results/air_defense_v1/task14_multibatch_leave_one_out/training_batch_audit.json
results/air_defense_v1/task14_multibatch_leave_one_out/oob_predictions.csv
results/air_defense_v1/task14_multibatch_leave_one_out/oob_curves.csv
results/air_defense_v1/task14_multibatch_leave_one_out/test_dataset.npz
results/air_defense_v1/task14_multibatch_leave_one_out/gate_summary.json
```
