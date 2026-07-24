# AirDefense v1 动作替代测量失真独立确认

更新时间：2026-07-23  
实验状态：已完成  
路线决策：P-C1/P-C2通过，P-C3失败；冻结资源类型条件贡献

## 1. 研究问题

R1在旧策略种子8/9/10中发现，当前交战会替代未来射击，使回合累计资源
成本差低估当前动作的直接消耗。本实验使用新策略、新状态和新增medium
场景回答：

> 动作替代导致的资源成本测量失真能否跨策略种子复现，其符号掩盖强度受
> 哪些场景和资源类型条件约束？

本任务只运行N/E共同随机数分支，不运行E-R，不训练机会成本网络或任何
Actor/Critic辅助。

## 2. 独立性协议

| 项目 | 正式配置 |
| --- | --- |
| 来源策略 | factorized joint PPO order 012 |
| 新策略种子 | 17、18、19 |
| 场景 | medium、time pressure、heterogeneity pressure |
| 来源模型 | 9个，10k steps，epochs=2 |
| 候选池 | 每块24个新环境回合 |
| 上下文 | 108个，旧hash重叠为0 |
| safety/resource | 每块6/6 |
| resource配额 | 每块3 missile + 3 laser |
| 重复 | 每上下文32次 |
| 目标 | 全部合法目标精确概率边缘化 |
| 后续策略 | stochastic continuation |
| Actor更新 | 禁止 |
| transition上限 | 266,198 |

种子17/18/19曾在测试或Task14预测器训练中出现，但从未用于动作替代标签
设计或factorized来源策略选择，因此按预注册保留。9个模型无条件进入
审计，没有根据all-noop、奖励或成本表现筛选。

## 3. 成本账本修正

首轮正式执行的future-only恒等式在287/7776条目标账本中出现非零残差，
最大为2.0；扩展恒等式误差仅`8.88e-16`。原因是被测单元强制交战后，
无冲突自回归后缀单元可能在同一步少执行一次交战。

因此按协议只修复账本：

```text
Sub_cost_total
= current_other_cost(N) - current_other_cost(E)
 + future_probe_cost(N) - future_probe_cost(E)
 + future_other_cost(N) - future_other_cost(E)
```

```text
Delta_C_episode = probe_direct_cost - Sub_cost_total
```

原始future-only成本和残差继续保留。首轮无效结果已归档至
`pre_ledger_correction/`，随后使用同一模型、上下文、随机带和门槛完成
唯一一次重跑。

## 4. 数据完整性

| 检查 | 结果 |
| --- | ---: |
| 来源模型 | 9/9 |
| 新上下文 | 108/108 |
| 旧hash重叠 | 0 |
| resource missile/laser | 27/27 |
| 目标概率最大误差 | 0 |
| 上下文—重复记录 | 3,456 |
| 目标成本账本 | 7,776 |
| 总替代恒等式最大误差 | `8.88e-16` |
| probe/other子分解最大误差 | `8.88e-16` |
| Actor最大参数差 | 0 |
| 新增transition | 157,485 |
| 软件回归 | 264 passed |

全部独立性和完整性门控通过。

## 5. P-C1：成本分解

修正后的每条账本满足：

```text
episode cost delta
= probe direct cost
 - same-step other-unit substitution
 - future probe substitution
 - future other-unit substitution
```

所有合法engage的被测单元直接成本均大于0，总账本与子分解误差均远低于
`1e-6`。**P-C1通过。**

time/resource平均总替代成本为0.864，其中同一步其他单元替代为0.147，
未来替代为0.718。约83%的替代成本仍来自当前步之后，账本修正没有推翻
R1的未来动作替代解释，只补全了联合动作内部约17%的同一步替代。

## 6. P-C2：独立确认

`time_pressure/resource`结果：

| 门控 | 结果 | 要求 |
| --- | ---: | ---: |
| `mean(Sub_shot)>0` | 13/18 | 至少12/18 |
| `lower95(Sub_shot)>0` | 13/18 | 至少6/18 |
| 正块级下界种子 | 3/3 | 至少2/3 |
| 符号掩盖率不低于50%的种子 | 2/3 | 至少2/3 |
| 非正累计成本差可解释 | 7/7 | 至少80% |

块级结果：

| 种子 | Sub_shot均值 | 95%下界 | 符号掩盖率 |
| --- | ---: | ---: | ---: |
| 17 | 0.878 | 0.757 | 0.969 |
| 18 | 0.260 | 0.029 | 0.271 |
| 19 | 0.511 | 0.166 | 0.526 |

三个新种子均出现可靠正替代，结果不依赖单一优势种子。**P-C2通过。**

## 7. P-C3：资源类型边界

time/resource分层结果：

| 类型 | 上下文 | Sub_shot | 95%下界 | rho_sub | 掩盖上下文 |
| --- | ---: | ---: | ---: | ---: | ---: |
| missile | 9 | 0.373 | 0.133 | 0.571 | 2 |
| laser | 9 | 0.726 | 0.497 | 1.175 | 5 |

两种类型的替代射击均跨三个种子为正，但missile只有2个上下文发生平均累计
成本符号掩盖，低于3个门槛。**P-C3未通过。**

这不是“missile没有动作替代”，而是其替代成本通常不足以稳定改变累计
成本标签的符号；laser较低的直接成本更容易被后续或同一步替代完全抵消。

## 8. 场景适用边界

resource槽聚合：

| 场景 | Sub_shot | Sub_cost | rho_sub | 符号掩盖率 |
| --- | ---: | ---: | ---: | ---: |
| medium | 0.544 | 0.949 | 0.747 | 0.620 |
| time pressure | 0.550 | 0.864 | 0.873 | 0.589 |
| heterogeneity pressure | 0.876 | 1.435 | 0.972 | 0.865 |

三个场景均存在动作替代，但异质场景最强且接近完全抵消。首次未来替代
平均发生在medium第6.01步、time第4.53步、heterogeneity第3.08步。

## 9. 决策

P-C1和P-C2通过，P-C3失败，按预注册进入资源类型条件确认：

> 在AirDefense v1冻结factorized PPO的动态掩码序列分配中，同一步与未来
> 动作替代会系统性偏置回合累计资源成本对当前动作的局部信用读出；该机制
> 可跨全新策略种子复现，但是否改变成本标签符号受场景和资源类型约束。

后续停止增加种子、资源恢复、机会成本网络和BPCE/MCH-PPO实验。该结论
进入论文claim–evidence冻结与相关工作检索，不写成跨资源类型通用规律，
也不写成已经优于PPO的新算法。

## 10. 产物

```text
rein_learning/common/action_substitution_confirmation.py
scripts/run_air_defense_v1_action_substitution_confirmation.py
tests/test_action_substitution_confirmation.py

results/air_defense_v1/action_substitution_confirmation/
  seed_usage_audit.json
  source_model_manifest.json
  source_model_training_log.csv
  context_identity_check.csv
  context_selection.csv
  repeat_cost_ledger.csv
  repeat_marginal_metrics.csv
  context_substitution_estimates.csv
  block_summary.csv
  resource_type_summary.csv
  scenario_boundary_summary.csv
  gate_summary.json
  pre_ledger_correction/
```
