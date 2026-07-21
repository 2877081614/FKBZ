# AirDefense v1.0 任务十四：动作条件 Q-Critic 门控实验

更新时间：2026-07-19  
实验状态：正式实验完成，门控未通过  
路线判定：不恢复 MCH-PPO，不进入 GNN

## 1. 实验目的

任务十三确认现有 PPO Critic 只输出 `V(s)`，无法比较同一状态下的具体动作。任务十四在冻结环境、奖励、场景和任务十二策略的前提下，实现非图结构的掩码条件 Q-Critic，检验它是否能：

- 降低候选动作回报误差；
- 恢复候选动作和目标排序；
- 判断 engage 相对 no-op 的局部 advantage 符号；
- 在三个核心场景保持稳定；
- 以远低于 Monte Carlo 的开销完成全候选估值。

## 2. 正式协议

| 项目 | 配置 |
| --- | --- |
| 来源模型 | factorized seed 8、10 |
| 场景 | medium、time_pressure、heterogeneity_pressure |
| 状态数 | 90，每个 seed × 场景 15 个 |
| 候选动作样本 | 571 |
| 反事实 rollout | 每个候选 8 次共同随机数 rollout |
| 数据划分 | 338 / 117 / 116 行；按 state_id 分组的 60/20/20 |
| 训练种子 | 14、15、16 |
| 主模型 | 非图 MLP，256/128，83,457 参数 |
| 消融 | no_prefix、no_mask、observation_action_only |

数据划分不存在 `state_id` 泄漏。反事实数据生成耗时 334.78 秒，平均每个状态 3.72 秒。

## 3. 主模型结果

| 训练种子 | Q MAE | V MAE | 相对 V 改善 | 排序准确率 | 有效排序对 | 目标排序 | engage 符号 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 10.626 | 17.747 | 40.1% | 0.250 | 8 | 0.200 | 0.500 |
| 15 | 10.887 | 17.747 | 38.7% | 0.250 | 8 | 0.200 | 0.500 |
| 16 | 11.295 | 17.747 | 36.4% | 0.375 | 8 | 0.400 | 0.500 |

三个种子都通过“相对 V 的 MAE 至少改善 10%”和效率门槛，但均未通过排序、目标排序、top-1、engage/no-op 符号及跨场景门槛。完整门控通过种子数为 `0/3`，低于要求的 `2/3`。

有效比较覆盖也不足：

- 总体高置信候选比较只有 8 对，低于预注册的 30 对；
- 高置信目标比较只有 5 对；
- 高置信 top-1 状态只有 1 个；
- 高置信 engage/no-op 组只有 2 个。

因此这些判别指标不仅数值偏低，而且统计证据不足，不能通过扩大模型复杂度直接解释。

## 4. 场景表现

主模型三个训练种子的场景排序准确率范围为：

| 场景 | 排序准确率范围 | 有效比较数 |
| --- | ---: | ---: |
| heterogeneity_pressure | 0.25 | 4 |
| medium | 0.333-0.667 | 3 |
| time_pressure | 0.00 | 1 |

异质性压力场景最弱，但各场景有效比较数都太少，当前不能把失败归因于关系表示容量。

## 5. 消融结果

下表为三个训练种子的均值：

| 结构 | Q MAE | 总体排序 | 目标排序 |
| --- | ---: | ---: | ---: |
| full | 10.936 | 0.292 | 0.267 |
| no_prefix | 11.344 | 0.333 | 0.333 |
| no_mask | 10.423 | 0.333 | 0.333 |
| observation_action_only | 10.348 | 0.125 | 0.200 |

完整输入没有形成一致优势；去掉 mask 或实体/前缀信息后，MAE 甚至略低。该结果说明当前纯 Q 回归主要学习了状态层面的回报变化，尚未证明模型利用动态合法集和前缀信息恢复了动作间差值。

## 6. 门控判定

| 门控 | 结果 |
| --- | --- |
| state_id 无泄漏 | 通过 |
| MAE 相对 V 改善 >= 10% | 通过，3/3 seeds |
| 总体排序 >= 0.70 且有效对 >= 30 | 未通过 |
| engage/no-op 符号 >= 0.70 且有效组 >= 30 | 未通过 |
| 目标排序 >= 0.65 且有效对 >= 30 | 未通过 |
| top-1 >= 0.50 且有效状态 >= 30 | 未通过 |
| 三场景排序均 >= 0.60 | 未通过 |
| 推理快于 Monte Carlo | 通过 |
| 至少 2/3 训练种子整体通过 | 未通过，0/3 |

正式结论：

> 非图 Q-Critic 已证明“动作条件回报可做低误差近似”，但没有证明“动作差异可被可靠排序”。任务十四未达到接入反事实 PPO 的证据门槛。

因此：

- `resume_mch_ppo = false`；
- `enter_gnn = false`；
- 不运行 30k/100k MCH-PPO；
- 不把任务十四结果表述为第一创新成立。

## 7. 失败机理与下一入口

当前最可能的两个原因是：

1. 冻结策略访问的大多数状态中，候选动作长期回报差异相对轨迹方差很小，导致高置信监督样本不足；
2. MSE 优化更容易拟合状态共同价值，而不是同一状态内的细微动作差值。

下一入口应是任务十四的标签与目标修订，而不是任务十五：

- 使用自适应状态采样，提高存在明确候选差异的状态覆盖；
- 增加 rollout 或使用方差缩减回报，确保每类有效比较至少 30；
- 在保持测试协议冻结的前提下，引入组内 pairwise ranking 或 advantage-centered 监督；
- 先验证非图模型能利用 mask/prefix，再重新判定是否恢复 MCH-PPO；
- 只有数据可靠而非图模型仍稳定失败，才形成图反事实 Critic 的必要证据。

## 8. 产物

```text
results/air_defense_v1/task14_q_critic/dataset.npz
results/air_defense_v1/task14_q_critic/dataset_samples.csv
results/air_defense_v1/task14_q_critic/models/
results/air_defense_v1/task14_q_critic/training_curves.csv
results/air_defense_v1/task14_q_critic/predictions.csv
results/air_defense_v1/task14_q_critic/metrics.csv
results/air_defense_v1/task14_q_critic/gate_summary.json
results/air_defense_v1/task14_q_critic/experiment_config.json
```
