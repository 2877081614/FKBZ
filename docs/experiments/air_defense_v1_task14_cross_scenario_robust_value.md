# AirDefense v1 跨场景鲁棒预算正式实验

更新时间：2026-07-21  
实验状态：正式实验完成  
结论状态：鲁棒目标未被选择，新批次暴露异质场景漏交战

## 1. 实验目的

在不改变 state-budget 网络容量的条件下，比较标准损失、场景-类别鲁棒损失和可靠成本差加权，判断逐场景局部漂移是否可由 Group-DRO 风格监督解决。

## 2. 数据与隔离

| 项目 | 数值 |
| --- | ---: |
| 历史非 test 训练组 | 202 |
| grouped cross-fitting | 3折 |
| 全新 test 状态 | 72 |
| test 上下文组 | 88 |
| 每分支 rollout | 32 |
| 可靠 oracle 组 | 81 |
| engage / no-op | 38 / 43 |
| medium / time / heterogeneity | 26 / 26 / 29 |
| 旧观测重叠 | 全部0 |
| 回报重构最大误差 | `7.63e-06` |
| 数据生成时间 | 367.84 s |

数据完整性和全部功效门槛通过。

## 3. 交叉拟合

| seed | objective | BA | worst recall | cost corr | 可行 |
| ---: | --- | ---: | ---: | ---: | --- |
| 20 | standard | 0.778 | 0.634 | 0.160 | 否 |
| 20 | robust | 0.764 | 0.675 | 0.108 | 是 |
| 20 | robust+cost | 0.760 | 0.675 | 0.145 | 是 |
| 21 | standard | 0.756 | 0.700 | 0.196 | 是 |
| 21 | robust | 0.734 | 0.537 | 0.127 | 否 |
| 21 | robust+cost | 0.763 | 0.634 | 0.187 | 否 |
| 22 | standard | 0.764 | 0.706 | 0.288 | 是 |
| 22 | robust | 0.764 | 0.675 | 0.149 | 是 |
| 22 | robust+cost | 0.770 | 0.575 | 0.188 | 否 |

standard 与 robust 均为2/3可行，但 standard 平均最差召回更高；reliable-cost 只有1/3可行。因此正式候选冻结为 standard，鲁棒目标没有获得 test 选择机会。

## 4. 新 test 结果

| seed | BA | engage recall | no-op recall | false-noop | wasteful-engage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.758 | 0.842 | 0.674 | 0.158 | 0.326 |
| 21 | 0.662 | 0.579 | 0.744 | 0.421 | 0.256 |
| 22 | 0.634 | 0.500 | 0.767 | 0.500 | 0.233 |

seed20 通过总体门槛；seed21/22 由上一批的资源偏激进转为安全偏保守，BA、engage recall、false-noop 和 safety sign 同时失败。

## 5. 逐场景定位

| seed | 场景 | engage recall | no-op recall |
| ---: | --- | ---: | ---: |
| 20 | medium | 0.769 | 0.769 |
| 20 | time-pressure | 1.000 | 0.667 |
| 20 | heterogeneity | 0.727 | 0.611 |
| 21 | medium | 0.615 | 0.846 |
| 21 | time-pressure | 0.786 | 0.750 |
| 21 | heterogeneity | **0.273** | 0.667 |
| 22 | medium | 0.692 | 0.923 |
| 22 | time-pressure | 0.571 | 0.833 |
| 22 | heterogeneity | **0.182** | 0.611 |

主要失败集中于异质场景的必要交战漏判，而上一批正式 test 中三种子的异质 engage recall 为 `1.0`。该方向翻转证明存在显著的批次内分布漂移。

## 6. 阶段门控

| 门控 | 结果 |
| --- | --- |
| 数据与功效 | 全部通过 |
| OOF 鲁棒目标优于 standard | 否 |
| 至少2/3 test 种子完整通过 | 0/3 |
| 平均最差场景召回非劣 | 持平，因为选择了 standard |
| 平均 BA 非劣 | 持平 |
| 平均 cost correlation 非劣 | 持平 |
| 恢复 MCH-PPO | 否 |
| 进入 GNN | 否 |

## 7. 科学解释

场景-类别等权只能处理已观察样本中的组不平衡，不能覆盖同一场景内部未观察的临界状态子分布。训练语料来自有限采样批次，OOF 随机按状态划分时，各折仍共享相同批次生成机制，因此无法预测新的批次漂移。

下一阶段应建立多个独立训练/校准批次，并把 `batch_id` 纳入分组：

- 独立生成至少三个训练批次；
- leave-one-batch-out，而不是普通随机 state folds；
- 比较批次级 Group-DRO、批次重采样和安全收益表示；
- 最终仍使用全新未见批次一次性测试；
- 不使用本轮 test 训练或调参。

只有跨批次至少2/3种子完整通过，才能进入 MCH-PPO。

## 8. 结果入口

```text
results/air_defense_v1/task14_cross_scenario_robust_value/test_dataset.npz
results/air_defense_v1/task14_cross_scenario_robust_value/crossfit_curves.csv
results/air_defense_v1/task14_cross_scenario_robust_value/crossfit_predictions.csv
results/air_defense_v1/task14_cross_scenario_robust_value/model_metrics.csv
results/air_defense_v1/task14_cross_scenario_robust_value/test_group_diagnostics.csv
results/air_defense_v1/task14_cross_scenario_robust_value/gate_summary.json
```
