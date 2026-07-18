# AirDefense v1.0 任务十二：no-op 塌缩与 PPO 稳定性实验

更新时间：2026-07-18  
阶段状态：已完成，30k 筛选未通过，未运行 100k

## 1. 研究问题

本阶段检验任务十一的 all-no-op 是否属于真实概率塌缩、确定性 argmax 放大，及交战-目标因子化能否在不修改奖励的前提下消除该问题。

## 2. 固定探针与冻结模型回放

固定探针包含 768 个状态，三个核心场景各 256 个。任务十一种子 0/1/2 在相同环境种子下分别进行 deterministic 和 stochastic 各 100 回合/场景回放。

种子 1 的结果：

| 场景 | deterministic all-noop | stochastic all-noop | stochastic 交战率差 |
| --- | ---: | ---: | ---: |
| medium | 1.000 | 0.000 | 0.364 |
| time_pressure | 1.000 | 0.000 | 0.408 |
| heterogeneity_pressure | 1.000 | 0.000 | 0.369 |

固定探针上种子 1 的平均总交战概率为 0.4726，no-op 概率为 0.5274。结论是 deterministic argmax 明显放大 no-op，但策略概率并未完全退化为 0 交战。

## 3. 10k × 5 seeds 训练分叉

任务十一对照方法使用种子 3/4/5/6/7 训练 10k，每约 1k 步记录 PPO 和固定探针动态。

| seed | 最终状态 | actionable engagement | all-noop | probe p(engage) | no-op margin |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | 正常 | 0.941 | 0.000 | 0.787 | -0.663 |
| 4 | 塌缩 | 0.000 | 1.000 | 0.589 | 0.430 |
| 5 | 塌缩 | 0.000 | 1.000 | 0.450 | 1.007 |
| 6 | 正常 | 0.805 | 0.000 | 0.722 | -0.287 |
| 7 | 塌缩 | 0.000 | 1.000 | 0.385 | 1.310 |

五个种子初始化时总交战概率均约为 0.67，但目标概率分摊使 deterministic 仍倾向 no-op。训练后成功种子的 no-op margin 转负，失败种子转正，说明概率碎片化和 PPO 种子分叉同时存在。

## 4. 因子化候选

候选方法显式输出二元交战概率和条件目标分布，并采用“先判断交战，再选择目标”的分层确定性规则。环境、奖励、Critic、PPO 超参数和关系 scorer 输入语义保持冻结。

Smoke 使用种子 8/9 和一个完整 rollout，4 个模型训练、保存、加载、探针和评估闭环正常。

## 5. 30k × 3 seeds 正式筛选

协议：

```text
训练场景：medium
方法：任务十一 role-conditioned 对照 / factorized engagement 候选
种子：8 / 9 / 10
训练步数：30,000
测试：medium / time_pressure / heterogeneity_pressure
最终评估：50 个配对回合/场景/种子
```

产物完整性：

| 产物 | 数量 |
| --- | ---: |
| 模型 | 6 |
| 场景运行块 | 18 |
| deterministic 主评估回合 | 900 |
| 决策记录 | 109,041 |
| 高威胁泄漏归因 | 1,047 |
| 训练动态记录 | 96 |
| 探针动态记录 | 384 |

平均结果：

| 场景 | 方法 | 奖励 | 毁伤 | 资源成本 | 高威胁泄漏 | all-noop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| medium | 对照 | -60.16 | 1.831 | 3.907 | 0.503 | 0.300 |
| medium | 因子化 | -76.80 | 2.084 | 6.480 | 0.564 | 0.413 |
| time_pressure | 对照 | -78.36 | 2.251 | 3.277 | 0.658 | 0.387 |
| time_pressure | 因子化 | -91.20 | 2.392 | 6.067 | 0.696 | 0.533 |
| heterogeneity_pressure | 对照 | -60.02 | 1.824 | 4.780 | 0.495 | 0.233 |
| heterogeneity_pressure | 因子化 | -70.21 | 1.931 | 5.879 | 0.542 | 0.387 |

候选种子 8 在三个场景均为 100% all-noop；种子 9 仍为低交战；种子 10 则高交战且资源成本达到约 15.4–17.0。因子化没有消除分叉，而是形成“不开火”和“高成本开火”两极状态。

## 6. 门槛判定

共检查 19 项冻结门槛，通过 6 项、失败 13 项。

通过：

- 非法动作、冲突、过度分配均为 0；
- Actor 参数比为 1.0，Critic 完全相同；
- 异质场景 unassigned 泄漏占比由 0.9363 降至 0.7647，绝对下降 0.1716；
- 相对外部参考的 time-pressure 成本门槛通过。

主要失败：

- 6 个候选“场景×种子”组合存在 collapsed unit；
- 最坏场景平均 all-noop 为 0.5333，门槛为 0.02；
- 异质高威胁泄漏平均改善为 -0.0472，即实际恶化；
- medium 奖励相对同轮对照下降 16.64；
- medium 毁伤增加 0.254；
- time-pressure 资源成本增加 2.79；
- stochastic/deterministic 最大绝对交战率差为 0.404；
- 相对任务十决策时延增加 69.2%，门槛为 25%。

## 7. 阶段结论

任务十二完成了机制定位和反证：

1. 任务十一 all-no-op 包含 deterministic argmax 放大；
2. PPO 训练确实会进一步产生早期种子分叉；
3. 显式因子化能够减少未分配类型泄漏，但不足以稳定交战概率；
4. 候选未通过 30k 门槛，因此按预注册规则不运行 100k；
5. 当前不应把剩余问题归因于 GNN 表示能力。

下一阶段应研究交战概率校准与 Actor-Critic 优化稳定性，重点比较 deterministic 阈值、advantage 分布和 Critic 估计，而不是继续增加关系表示复杂度。

## 8. 结果位置

```text
results/air_defense_v1/task12_probe_corpus/
results/air_defense_v1/task12_task11_frozen_replay/
results/air_defense_v1/task12_role_diagnostic_10k_5seeds/
results/air_defense_v1/task12_factorized_smoke/
results/air_defense_v1/task12_factorized_screening_30k_3seeds/
results/air_defense_v1/task12_analysis/
```
