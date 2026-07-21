# AirDefense v1.0 任务十四分层 Q 正式实验

更新时间：2026-07-20  
实验状态：正式实验完成，门控未通过  
路线判定：不恢复 MCH-PPO，不进入 GNN

## 1. 实验目的

任务十四修订已经证明组内动作差异监督能够改善目标排序，但 engage/no-op 仍不稳定。本轮保持环境、冻结策略、训练数据和非图结构不变，将动作价值显式拆为交战头与条件目标头，并与上一轮 `difference_aware` 单标量模型进行同测试集对照。

## 2. 数据协议

| 项目 | 结果 |
| --- | ---: |
| 原 train / validation | 338 / 117 行 |
| 原 test 排除 | 116 行 |
| 上一轮正式观测重叠 | 0 |
| 全新测试状态 | 108 |
| 动作候选行 | 684 |
| actionable engagement 组 | 144 |
| target 行 | 360 |
| 每候选 rollout | 32 |
| 数据生成耗时 | 1861.50 秒 |
| 测试随机种子 | 291000 |

新测试未参与训练、早停或损失权重选择。

## 3. 交战层结果

| seed | 单标量基线符号 | 双头符号 | 有效组 | 基线 MAE | 双头 MAE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 0.824 | 0.706 | 17 | 12.669 | 12.612 |
| 15 | 0.882 | 0.588 | 17 | 13.331 | 13.064 |
| 16 | 0.941 | 0.588 | 17 | 12.930 | 13.041 |

双头 engagement 符号相对基线分别变化 `-0.118 / -0.294 / -0.353`，平均下降 `0.255`。数值 MAE保持非劣，但局部符号明显变差，说明单独回归 `[Q_noop,Q_engage]` 没有形成更可靠的交战判别。

逐场景有效组为：

| 场景 | 有效组 | 双头准确率范围 |
| --- | ---: | ---: |
| medium | 5 | 0.60 |
| time_pressure | 7 | 0.571-0.857 |
| heterogeneity_pressure | 5 | 0.60 |

总体和逐场景功效均未达到预注册数量门槛。

## 4. 目标层结果

| seed | 基线排序 | 双头排序 | 有效对 | 基线 top-1 | 双头 top-1 | top-1 组 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 0.770 | 0.870 | 100 | 0.667 | 0.875 | 24 |
| 15 | 0.800 | 0.850 | 100 | 0.667 | 0.833 | 24 |
| 16 | 0.810 | 0.830 | 100 | 0.667 | 0.750 | 24 |

目标排序平均提高 `0.057`，三个场景排序均超过 `0.60`：

- medium：`0.900-0.967`，30 对；
- time_pressure：`0.857-0.952`，42 对；
- heterogeneity_pressure：`0.679-0.714`，28 对。

但是目标 MAE 从基线的 `12.22-12.87` 恶化到 `14.61-15.12`，超过 10% 非劣门槛；top-1 虽达到 `0.75-0.875`，有效组只有 24。

## 5. 功效投影

基于当前 108 状态：

| rollout | engage 组 | target 对 | target top-1 组 |
| ---: | ---: | ---: | ---: |
| 32 | 17 | 100 | 24 |
| 64 | 37 | 173 | 44 |
| 128 | 58 | 257 | 70 |
| 256 | 77 | 281 | 81 |

64 rollout 预计可使总体功效达标，并使三场景 engage 有效组分别达到 `12/13/12`；但异质场景 target top-1 预计仍只有 9，128 rollout 才能使所有层级和场景数量门槛同时有机会满足。

本轮不追加 rollout。原因是双头 engagement 在已有高置信组上稳定劣于基线，追加采样只能改善置信数量，不能消除当前负向方法对照。

## 6. 门控判定

| 门控 | 结果 |
| --- | --- |
| 数据隔离 | 通过 |
| engage 总体 | 未通过：17 组，准确率 `0.588-0.706` |
| engage 场景 | 未通过：每场景仅 5-7 组 |
| target 总体 | 通过：100 对，准确率 `0.83-0.87` |
| target 场景 | 通过 |
| target top-1 | 未通过：24 组 |
| engage MAE 非劣 | 通过 |
| target MAE 非劣 | 未通过 |
| 效率 | 通过 |
| engage 相对基线提升 | 未通过：平均 `-0.255` |
| target 相对基线非劣 | 通过：平均 `+0.057` |
| 至少 2/3 seeds 整体通过 | 未通过，`0/3` |

正式状态：

- `task14_hierarchical_passed = false`；
- `resume_mch_ppo = false`；
- `enter_gnn = false`。

## 7. 科学结论

> 条件目标价值已经具备较强且跨场景稳定的排序能力；当前核心瓶颈不是目标关系表示，而是均值型 engage/no-op 价值无法稳定表达交战决策。

显式拆头本身不是解决方案。基线单标量模型在已有高置信 engagement 组上反而更准确，说明简单增加层级结构可能破坏状态与动作之间的共享正则化。

下一阶段不应继续堆叠 Q 头或直接训练 MCH-PPO，而应审查交战效用：

- 均值回报是否掩盖低概率高损失突防；
- 资源成本与毁伤收益是否需要显式约束或对偶变量；
- `Q_engage` 是否应预测回报分布、CVaR 或多目标分量；
- seed 8/10 的交战价值是否存在系统性分布偏移。

在风险/约束语义离线通过前，第一创新假设仍不成立，GNN 也没有进入依据。

## 8. 产物

```text
results/air_defense_v1/task14_hierarchical_q/test_dataset.npz
results/air_defense_v1/task14_hierarchical_q/test_dataset_samples.csv
results/air_defense_v1/task14_hierarchical_q/analysis_dataset.npz
results/air_defense_v1/task14_hierarchical_q/models/
results/air_defense_v1/task14_hierarchical_q/training_curves.csv
results/air_defense_v1/task14_hierarchical_q/metrics.csv
results/air_defense_v1/task14_hierarchical_q/engagement_predictions.csv
results/air_defense_v1/task14_hierarchical_q/gate_summary.json
results/air_defense_v1/task14_hierarchical_q/power_analysis/
```
