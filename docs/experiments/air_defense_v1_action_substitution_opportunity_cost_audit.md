# AirDefense v1 动作替代与弹药机会成本可辨识性审计

更新时间：2026-07-23  
实验状态：已完成  
路线决策：保留动作替代解释，停止通用机会成本oracle与在线辅助路线

## 1. 研究问题

BPCE短视窗审计中，`time_pressure/resource`的累计资源成本差接近0或为
负。本实验回答两个相互分离的问题：

1. 当前强制交战是否替代了冻结策略未来原本会执行的射击？
2. 在保持当前交战结果不变时，恢复被消耗的一枚弹药是否具有稳定的未来
   行动与安全价值？

第一问解释累计成本标签为何失真；第二问决定能否定义通用的弹药机会成本
监督。二者不得合并为单一训练分数。

## 2. 冻结三分支

对每个上下文、被测单元、合法目标和共同随机数重复构造：

| 分支 | 当前步 | 后续 |
| --- | --- | --- |
| N | 被测单元强制no-op | 冻结策略随机延续 |
| E | 被测单元强制交战合法目标 | 正常扣弹、成本、冷却、命中后随机延续 |
| E-R | 当前步与E完全相同 | 下一策略观察前只恢复被测单元1枚弹药 |

E-R不退还即时成本，不修改冷却、命中、目标推进或其他单元。E/E-R共享
当前命中随机数、后续环境随机带和策略uniform tape。所有未来指标都排除
当前步。

| 项目 | 冻结配置 |
| --- | --- |
| 场景 | `time_pressure`、`heterogeneity_pressure` |
| 策略 | 原10k factorized joint PPO |
| 策略种子 | 8、9、10 |
| 上下文 | 阶段A/A2原72个，安全/资源各36个 |
| 重复 | 每上下文32次 |
| 目标 | 全部合法目标条件概率精确边缘化 |
| Actor更新 | 禁止 |
| transition上限 | 266,198 |

## 3. 估计量

动作替代：

```text
Sub_shot = future_shots(N) - future_shots(E)
Sub_cost = future_cost(N) - future_cost(E)
```

弹药复用与行动集合：

```text
Reuse_probe = future_probe_shots(E-R) - future_probe_shots(E)
OptionEdge = sum legal_edges(E-R) - sum legal_edges(E)
```

安全收益：

```text
AmmoGain_D = zone_damage(E) - zone_damage(E-R)
AmmoGain_L = high_threat_leaks(E) - high_threat_leaks(E-R)
```

可靠正机会价值要求安全分量95%下界超过冻结最小效应，同时
`Reuse_probe>0`或`OptionEdge>0`。终止步标记为
`opportunity_not_observable`，不解释为零机会价值。

## 4. 完整性

| 检查 | 结果 |
| --- | ---: |
| 上下文身份 | 72/72 |
| 目标概率最大误差 | `4.98e-13` |
| 上下文—重复记录 | 2,304 |
| E/E-R目标—重复干预 | 5,408 |
| 当前步不一致 | 0 |
| 非预期状态差 | 0 |
| 终止步机会不可观测 | 505 |
| 正常恢复1枚弹药 | 4,903 |
| 最大成本分解误差 | `4.00e-15` |
| Actor最大参数差 | `0.0` |
| 实际新增transition | 219,142 |
| 完整软件回归 | 259 passed |

全部完整性门控通过。505个当前步终止的分支被单独记录，没有作为“机会
价值为零”的观测使用。

## 5. P-R1：动作替代

`time_pressure/resource`结果：

| 门控 | 结果 | 要求 |
| --- | ---: | ---: |
| `mean(Sub_shot)>0` | 18/18 | 至少12/18 |
| `lower95(Sub_shot)>0` | 18/18 | 至少6/18 |
| 非正累计成本差 | 11个 | 报告 |
| 可由未来替代解释 | 11/11，100% | 至少80% |
| 最大成本重构误差 | `4.00e-15` | 不超过`1e-6` |

该槽跨上下文平均`Sub_shot=0.990`、`Sub_cost=1.995`。当前强制交战平均
替代约一次未来射击，因此其即时资源支出会被后续少射击抵消。首次可观测
替代平均发生在当前动作后的第2.86步。

**P-R1通过。** 阶段A2中的非正累计成本差主要是动作替代造成的结构性
混叠，不是简单增加重复数可以修复的噪声问题。

## 6. P-R2：机会价值

| 场景 | 资源槽可靠机会 | 安全槽可靠机会 |
| --- | ---: | ---: |
| time pressure | 5/18 | 1/18 |
| heterogeneity pressure | 2/18 | 1/18 |

资源槽块级结果：

| 场景/种子 | 可靠机会 |
| --- | ---: |
| time/8 | 1/6 |
| time/9 | 2/6 |
| time/10 | 2/6 |
| heterogeneity/8 | 0/6 |
| heterogeneity/9 | 2/6 |
| heterogeneity/10 | 0/6 |

虽然time资源槽平均`Reuse_probe=1.000`、`OptionEdge=3.904`，异质资源槽
也有`Reuse_probe=0.333`、`OptionEdge=1.261`，但安全收益置信下界只在
少数上下文可靠。异质场景的全部可靠资源机会由seed9贡献。

**P-R2未通过。** “弹药被再次使用或扩大合法动作集”不等价于“稳定改善
最终安全结果”，不能据此建立跨场景机会成本监督。

## 7. P-R3：资源临界性

资源槽可靠率虽高于安全槽，但仍有两项失败：

- time场景单位额外transition的可靠资源机会功效为`4.78e-5`，低于安全
  槽的`5.49e-5`；
- 7个可靠资源上下文全部属于`missile`，没有覆盖`laser`。

异质场景的单位transition功效满足资源高于安全，但最差资源块仍为0。

**P-R3未通过。** 当前机会价值依赖高成本missile和特定策略种子，不能
声称是通用的资源临界信号。

## 8. 决策

完整性和P-R1通过，P-R2/P-R3失败，按预注册进入唯一决策分支：

> 累计资源成本差失效可以由未来动作替代解释；但在当前AirDefense v1
> 环境和冻结策略下，失去一枚弹药的未来安全价值不能跨场景、跨种子、
> 跨资源类型稳定辨识。

因此：

- 保留动作替代作为已验证机制；
- 停止通用弹药机会成本oracle预测任务；
- 不把E-R结果接入PPO、BPCE或MCH-PPO；
- 不通过增加重复、挑选seed9或只报告missile维持正结论；
- GNN继续冻结，不能用关系表示扩展绕过监督不可辨识问题。

## 9. 研究贡献边界

本阶段形成的是测量与可辨识性结论，而不是新算法胜出：

1. 给出累计资源成本标签受未来动作替代混叠的直接配对证据；
2. 用保持当前交战结果不变的单发资源恢复干预隔离未来选择权；
3. 证明“动作集合扩大”与“可靠安全价值”之间存在跨场景、跨种子断裂；
4. 明确通用机会成本在线辅助在当前任务中的停止边界。

这些结果可用于论文的问题定义、负结果和消融边界，但不能表述为已经完成
的MCH-PPO/BPCE算法创新。

## 10. 产物

```text
rein_learning/common/action_substitution_opportunity_cost.py
scripts/run_air_defense_v1_action_substitution_opportunity_cost_audit.py
tests/test_action_substitution_opportunity_cost.py

results/air_defense_v1/action_substitution_opportunity_cost_audit/
  experiment_config.json
  context_identity_check.csv
  intervention_integrity.csv
  repeat_branch_metrics.csv
  context_opportunity_estimates.csv
  target_opportunity_estimates.csv
  block_summary.csv
  gate_summary.json
```
