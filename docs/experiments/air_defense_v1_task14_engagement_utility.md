# AirDefense v1 任务十四交战效用正式实验

更新时间：2026-07-20  
实验状态：正式实验与功效审计完成，门控未通过  
结论状态：效用层有正向信号，正类功效与回归估值层未通过

## 1. 实验目的

本实验检验风险/约束感知效用能否比原始均值回报更准确地同时识别“必要交战”和“浪费性交战”，并验证同结构非图 Critic 能否学习该效用。

## 2. 数据与隔离

| 项目 | 数值 |
| --- | ---: |
| 全新状态 | 108 |
| engage/no-op 上下文组 | 150 |
| train/validation/test 组 | 58 / 29 / 63 |
| 每分支 rollout | 32 |
| 数据生成时间 | 954.49 s |
| 三轮旧测试观测重叠 | 0 / 0 / 0 |
| state split 泄漏 | 0 |
| 总回报重构最大误差 | `7.63e-06` |

数据来自冻结 factorized policy seeds 8/10 与 `medium`、`time_pressure`、`heterogeneity_pressure`。test 占40%，用于提高罕见交战反事实的独立功效。

## 3. Validation 冻结配置

108组候选效用网格只在 validation 上比较。最终配置：

```text
cost_weight = 2.0
damage_weight = 30.0
high_threat_leak_weight = 0.0
cvar_weight = 0.5
cvar_alpha = 0.25
```

validation balanced accuracy 为 `0.870`。与其他最高分配置并列时，按预注册规则选择距离冻结基线更近的配置。

## 4. 效用层独立测试

| 指标 | mean return | risk/constraint | 变化 |
| --- | ---: | ---: | ---: |
| accuracy | 0.754 | 0.860 | +0.105 |
| balanced accuracy | 0.713 | 0.926 | +0.213 |
| engage recall | 0.667 | 1.000 | +0.333 |
| no-op recall | 0.759 | 0.852 | +0.093 |
| false-noop rate | 0.333 | 0.000 | -0.333 |
| wasteful-engage rate | 0.241 | 0.148 | -0.093 |

候选效用通过全部四项性能门槛，但因 oracle-engage 数量不足，效用层不能正式判定通过。

## 5. Oracle 功效

test 共63组，其中57组具有可靠 oracle 标签：

| 类别 | 数量 |
| --- | ---: |
| oracle engage | 3 |
| oracle no-op | 54 |
| ambiguous | 6 |

三个 engage 全部来自 `medium`；`time_pressure` 和 `heterogeneity_pressure` 的 test 中没有可靠 engage。全数据150组中共有12个 engage、120个 no-op 和18个 ambiguous，说明类别失衡不是单一 split 偶然造成，但 test 的场景分布进一步放大了问题。

按 test 点估计 `p_engage=0.0526`，获得8个 engage 预计需要152个有效 test 组，对应约261个总状态。按95% Wilson 下界估计则需要443个有效 test 组、约760个总状态。因此不建议直接均匀扩样，应定向采集安全临界状态。

## 6. Critic 结果

| seed | mean BA | risk BA | mean false-noop | risk false-noop | mean wasteful | risk wasteful |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.444 | 0.398 | 0.667 | 1.000 | 0.444 | 0.204 |
| 18 | 0.398 | 0.435 | 0.667 | 0.667 | 0.537 | 0.463 |
| 19 | 0.611 | 0.435 | 0.333 | 0.667 | 0.444 | 0.463 |

risk-constraint Critic 虽然常能减少浪费性交战，但倾向牺牲稀有 engage 类，不能稳定降低 false-noop。三个种子均未达到 `BA>=0.70` 或相对基线提高 `0.10`，通过数为 `0/3`。

## 7. 门控结论

| 门控 | 结果 |
| --- | --- |
| 数据隔离与回报重构 | 通过 |
| test 总有效组与逐场景有效组 | 通过 |
| oracle engage/no-op 各不少于8 | 未通过：3 / 54 |
| 候选效用性能 | 指标通过，但功效不足 |
| Critic 至少2/3 seeds通过 | 未通过：0/3 |
| 恢复 MCH-PPO | 否 |
| 进入 GNN | 否 |

## 8. 科学解释

这不是“风险效用失败”的充分证据。候选公式在独立 test 上同时减少了两类错误，说明提高资源约束并惩罚低回报尾部具有正确方向。真正失败发生在两个位置：

1. 随机状态池中的必要交战反事实过少，类别功效不足；
2. 连续绝对效用回归没有针对稀有符号决策进行类别平衡，网络再次被多数 no-op 主导。

下一阶段应冻结本轮 test 和效用配置，定向采集安全临界状态，并比较类别平衡 BCE、成对 margin/ranking 和分位数估值。只有新的独立 test 与至少2/3训练种子通过后，才允许进入最小 MCH-PPO。

## 9. 结果入口

```text
results/air_defense_v1/task14_engagement_utility/dataset.npz
results/air_defense_v1/task14_engagement_utility/gate_summary.json
results/air_defense_v1/task14_engagement_utility/utility_grid.csv
results/air_defense_v1/task14_engagement_utility/model_metrics.csv
results/air_defense_v1/task14_engagement_utility/test_group_diagnostics.csv
results/air_defense_v1/task14_engagement_utility/oracle_power_counts.csv
results/air_defense_v1/task14_engagement_utility/engage_power_projection.csv
results/air_defense_v1/task14_engagement_utility/power_summary.json
```
