# AirDefense v1 资源约束交战边界校准实验

更新时间：2026-07-21  
实验状态：正式实验完成  
结论状态：数据功效通过，标量资源边界门控未通过

## 1. 实验问题

在冻结 BCE+margin Critic、风险 oracle 和环境后，检验全局阈值或成本-弹药对偶压力能否同时保留必要交战并减少过度交战。所有边界参数只使用上一轮 validation，正式 test 为72个全新状态。

## 2. 数据与隔离

| 项目 | 数值 |
| --- | ---: |
| 全新状态 | 72 |
| 上下文组 | 84 |
| 每分支 rollout | 32 |
| 可靠 oracle 组 | 81 |
| engage / no-op | 31 / 50 |
| medium 可靠组 | 24（9 engage） |
| time-pressure 可靠组 | 29（13 engage） |
| heterogeneity 可靠组 | 28（9 engage） |
| 六组旧数据观测重叠 | 全部0 |
| 回报重构最大误差 | `7.63e-06` |
| 数据生成时间 | 341.99 s |

数据完整性、总样本、双类数量、逐场景数量和跨场景 engage 门槛全部通过。

## 3. Validation 校准

| seed | 全局 BA | 全局可行 | 资源对偶 BA | lambda | 资源对偶可行 |
| ---: | ---: | --- | ---: | ---: | --- |
| 20 | 0.741 | 否 | 0.755 | 1.00 | 否 |
| 21 | 0.755 | 否 | 0.769 | 0.25 | 否 |
| 22 | 0.755 | 否 | 0.755 | 0.25 | 否 |
| 平均 | 0.750 | 0/3 | 0.759 | - | 0/3 |

按“可行种子数、平均 BA、模型简洁度”的预注册顺序，两个方法可行数同为0，资源对偶因平均 BA 较高被选中。需要强调：被选中不代表满足约束；validation 已经给出了无可行边界的预警。

## 4. 独立测试

| seed | 方法 | BA | engage recall | no-op recall | false-noop | wasteful-engage |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20 | zero margin | 0.589 | 0.839 | 0.340 | 0.161 | 0.660 |
| 20 | risk regression | 0.595 | 0.710 | 0.480 | 0.290 | 0.520 |
| 20 | resource dual | 0.593 | 0.806 | 0.380 | 0.194 | 0.620 |
| 21 | zero margin | 0.622 | 0.903 | 0.340 | 0.097 | 0.660 |
| 21 | risk regression | 0.595 | 0.710 | 0.480 | 0.290 | 0.520 |
| 21 | resource dual | 0.612 | 0.903 | 0.320 | 0.097 | 0.680 |
| 22 | zero margin | 0.609 | 0.839 | 0.380 | 0.161 | 0.620 |
| 22 | risk regression | 0.579 | 0.677 | 0.480 | 0.323 | 0.520 |
| 22 | resource dual | 0.605 | 0.871 | 0.340 | 0.129 | 0.660 |

资源对偶保持了较高 engage recall，但没有达到 `BA>=0.70`，也没有相对零阈值把 wasteful-engage 降低0.10。三种子均劣于风险回归的 `0.52` wasteful-engage。

## 5. 逐场景失败

校准后 no-op recall：

| seed | medium | time-pressure | heterogeneity |
| ---: | ---: | ---: | ---: |
| 20 | 0.267 | 0.500 | 0.368 |
| 21 | 0.200 | 0.375 | 0.368 |
| 22 | 0.200 | 0.438 | 0.368 |

所有场景均低于 `0.65`，因此问题不再只属于 `time_pressure`。在本轮新状态分布上，零阈值分类器本身的 no-op recall 也从上一轮的 `0.565-0.630` 降至 `0.34-0.38`，显示交战边界存在明显的跨批次状态条件漂移。

## 6. 压力特征诊断

以 seed20 的冻结 logit 为例：

| oracle | 平均资源压力 | 平均 margin logit |
| --- | ---: | ---: |
| no-op | 0.610 | 0.038 |
| engage | 0.467 | 0.096 |

资源压力方向正确，但两类压力范围均覆盖约 `0.09-1.67`。validation 为保留 engage recall 选择了负阈值，抵消了对偶压力的一部分作用。结果说明单元成本与当前弹药无法单独表达“当前是否值得消耗资源”，还需要剩余任务预算、未来风险、目标紧迫度和替代单元能力等状态条件。

## 7. 门控结论

| 门控 | 结果 |
| --- | --- |
| 数据完整性 | 通过 |
| 双类与逐场景功效 | 通过 |
| BA >= 0.70 | 0/3 |
| false-noop 对风险回归非劣 | 3/3 |
| wasteful-engage 对风险回归非劣 | 0/3 |
| 相对零阈值 wasteful 改善 >=0.10 | 0/3 |
| 逐场景 no-op recall >=0.65 | 0/3 |
| 至少2/3种子整体通过 | 未通过：0/3 |
| 恢复 MCH-PPO | 否 |
| 进入 GNN | 否 |

## 8. 下一步

停止继续扩大标量 `tau/lambda` 网格。下一阶段应先建立状态条件的资源预算表示和独立校准协议，比较：

- 显式预算约束或 cost-value head；
- 随状态变化的拉格朗日乘子，而不是固定 `lambda`；
- 交叉拟合或独立 calibration split，减少同一 validation 的重复选择；
- 在保持安全召回下直接约束 wasteful-engage 的 Pareto 前沿。

只有该机制在新的独立 test 上至少2/3种子通过，才恢复最小 MCH-PPO。当前失败不构成进入 GNN 的依据。

## 9. 结果入口

```text
results/air_defense_v1/task14_engagement_calibration/test_dataset.npz
results/air_defense_v1/task14_engagement_calibration/calibration_grid.csv
results/air_defense_v1/task14_engagement_calibration/model_metrics.csv
results/air_defense_v1/task14_engagement_calibration/test_group_diagnostics.csv
results/air_defense_v1/task14_engagement_calibration/gate_summary.json
results/air_defense_v1/task14_engagement_calibration/experiment_config.json
```
