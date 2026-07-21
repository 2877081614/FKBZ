# AirDefense v1.0 任务十四修订：动作差异监督实验

更新时间：2026-07-19  
实验状态：正式实验完成，附加对照通过，原门控未通过  
路线判定：不恢复 MCH-PPO，不进入 GNN

## 1. 实验目的

任务十四的 Q-Critic 显著降低了 Q 数值误差，但没有可靠恢复同一状态内的候选动作排序。本轮保持环境、冻结策略、Q 语义、网络结构和训练种子不变，只修订：

- 测试标签功效；
- 组内中心化监督；
- 配对动作差值监督。

目标是区分“绝对 Q 回归被状态共同价值主导”和“非图关系模型本身无法学习动作差异”。

## 2. 数据隔离与正式协议

| 项目 | 结果 |
| --- | ---: |
| 任务十四旧 train | 338 行，保留训练 |
| 任务十四旧 validation | 117 行，仅用于早停 |
| 任务十四旧 test | 116 行，全部排除 |
| 新旧状态 ID 交集 | 0 |
| 全新测试状态 | 36 |
| 全新测试候选动作 | 192 |
| 每候选 rollout | 32 |
| 新测试数据生成耗时 | 398.44 秒 |
| 训练种子 | 14、15、16 |

全新测试状态使用 `test_eval_seed=191000`，与 smoke 状态池分离。测试标签未用于损失权重、早停、epoch 或结构选择。

## 3. 方法对照

两种方法使用同一个 83,457 参数的 `MaskedActionQCritic`：

```text
absolute_mse:
    L = L_abs

difference_aware:
    L = L_abs + L_center + 0.5 * L_pair
```

`L_center` 比较同一 `state + unit` 内中心化后的 Q；`L_pair` 对候选 Q 差值使用可靠性加权 SmoothL1。权重仅由训练集 8-rollout 配对标准误计算并截断到 `[0.25, 4.0]`。

## 4. 总体结果

| 方法 | seed | Q MAE | 相对 V 改善 | 总体排序 | 有效对 | 目标排序 | top-1 | engage 符号 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| absolute_mse | 14 | 13.075 | 18.2% | 0.568 | 44 | 0.522 | 0.400 | 0.545 |
| absolute_mse | 15 | 14.094 | 11.8% | 0.500 | 44 | 0.435 | 0.500 | 0.636 |
| absolute_mse | 16 | 13.024 | 18.5% | 0.523 | 44 | 0.478 | 0.500 | 0.545 |
| difference_aware | 14 | 13.126 | 17.8% | 0.659 | 44 | 0.696 | 0.600 | 0.545 |
| difference_aware | 15 | 13.924 | 12.8% | 0.727 | 44 | 0.826 | 0.600 | 0.545 |
| difference_aware | 16 | 12.865 | 19.5% | 0.705 | 44 | 0.783 | 0.600 | 0.545 |

差异感知监督相对同种子绝对回归的总体排序提升分别为 `+0.091 / +0.227 / +0.182`，平均提升 `+0.167`。平均 MAE 比值为 `0.993`，没有数值精度退化。两个预注册附加对照均通过：

- 平均排序提升 >= 0.10：通过；
- MAE 平均恶化不超过 10%：通过。

这证明组内动作差异监督确实改变了模型学习内容，不只是重新缩放 Q 值。

## 5. 场景与层级功效

正式 32-rollout 测试的高置信样本数：

| 指标 | 有效数量 | 门槛 |
| --- | ---: | ---: |
| 总体动作对 | 44 | 30 |
| 目标动作对 | 23 | 30 |
| top-1 状态/单元组 | 10 | 30 |
| engage/no-op 组 | 11 | 30 |
| medium 动作对 | 9 | 30 |
| time_pressure 动作对 | 28 | 30 |
| heterogeneity_pressure 动作对 | 7 | 30 |

`difference_aware` 的逐场景排序准确率为：

| 场景 | seed 14 | seed 15 | seed 16 |
| --- | ---: | ---: | ---: |
| medium | 0.667 | 0.667 | 0.667 |
| time_pressure | 0.607 | 0.714 | 0.679 |
| heterogeneity_pressure | 0.857 | 0.857 | 0.857 |

准确率方向积极，但逐场景数量均未达到冻结门槛，因此不能作稳定跨场景结论。

## 6. 功效投影

基于已保存的 32 条配对轨迹，在保持观测效应和方差不变的近似下：

| rollout | 总体对 | 目标对 | top-1 | engage | medium | time | heterogeneity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 44 | 23 | 10 | 11 | 9 | 28 | 7 |
| 64 | 72 | 40 | 14 | 17 | 24 | 36 | 12 |
| 128 | 98 | 51 | 20 | 26 | 36 | 42 | 20 |
| 256 | 116 | 59 | 23 | 30 | 44 | 50 | 22 |

单纯增加同一批状态的 rollout 无法使 `heterogeneity_pressure` 和 top-1 达到 30。后续若继续验证，必须增加独立异质场景状态，而不是只重复轨迹。

## 7. 门控判定

| 门控 | 判定 |
| --- | --- |
| 数据隔离 | 通过 |
| Q MAE 相对 V 改善 | 通过，3/3 seeds |
| 总体排序 | seed 15/16 准确率通过；seed 14 未通过 |
| 目标排序 | 准确率方向通过，数量 23 不足 |
| top-1 | 准确率 0.60，数量 10 不足 |
| engage/no-op | 准确率 0.545，数量 11，不通过 |
| 三场景排序 | 准确率均 >= 0.60，逐场景数量不足 |
| 效率 | 通过 |
| 至少 2/3 seeds 整体通过 | 未通过，0/3 |

正式结论：

> 组内差异监督显著改善了动作和目标排序，证明任务十四的纯回归目标确实受到状态共同价值主导；但 engage/no-op 信用没有改善，层级有效样本也不足，因此动作 Q 仍不能直接接入 MCH-PPO。

对应状态：

- `task14_refinement_passed = false`；
- `resume_mch_ppo = false`；
- `enter_gnn = false`。

## 8. 下一研究入口

下一阶段应继续保持离线和非图结构，建立显式分层 Q 诊断：

1. 单独估计 `Q_engage(s,h_i,e_i)`，比较 engage/no-op；
2. 在 `e_i=engage` 条件下估计 `Q_target(s,h_i,target)`；
3. 对 engage 和 target 分别中心化、分别建立有效样本门槛；
4. 增加独立异质场景状态，而不是把同一状态提高到 256 rollout；
5. 只有 engage 符号和 target 排序同时通过，才实现最小 MCH-PPO。

这一步将检验“分层信用”是否真正必要，同时避免在信用接口尚未成立时引入 PPO 或 GNN。

## 9. 产物

```text
results/air_defense_v1/task14_q_critic_ranking_refinement/test_dataset.npz
results/air_defense_v1/task14_q_critic_ranking_refinement/test_dataset_samples.csv
results/air_defense_v1/task14_q_critic_ranking_refinement/analysis_dataset.npz
results/air_defense_v1/task14_q_critic_ranking_refinement/models/
results/air_defense_v1/task14_q_critic_ranking_refinement/training_curves.csv
results/air_defense_v1/task14_q_critic_ranking_refinement/predictions.csv
results/air_defense_v1/task14_q_critic_ranking_refinement/metrics.csv
results/air_defense_v1/task14_q_critic_ranking_refinement/gate_summary.json
results/air_defense_v1/task14_q_critic_ranking_refinement/power_analysis/
```
