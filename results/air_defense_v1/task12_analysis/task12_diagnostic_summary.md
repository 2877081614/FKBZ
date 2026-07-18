# Task 12 no-op 稳定性诊断摘要

## 冻结模型回放

- 机制判定：`deterministic_argmax_amplification`
- 种子 1 固定探针平均交战概率：0.4726
- 种子 1 固定探针平均 no-op 概率：0.5274

| 场景 | deterministic all-noop | stochastic all-noop | 交战率差 |
| --- | ---: | ---: | ---: |
| medium | 1.000 | 0.000 | 0.364 |
| time_pressure | 1.000 | 0.000 | 0.408 |
| heterogeneity_pressure | 1.000 | 0.000 | 0.369 |

## 10k 训练分叉

- 成功种子：2
- 塌缩种子：3

| seed | collapsed | reward | actionable engage | all-noop | probe p(engage) | no-op margin | stable step |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
## 30k 配对筛选

- 通过门槛：6
- 失败门槛：13
- 是否运行 100k：false

| 类别 | 门槛 | 数值 | 要求 | 通过 |
| --- | --- | ---: | --- | :---: |
| structure | structural_zero_violations | 0.000000 | == 0 | true |
| structure | actor_parameter_ratio | 1.000000 | 0.90 <= ratio <= 1.10 | true |
| structure | critic_parameters_equal | True | true | true |
| stability | candidate_collapsed_scenario_seeds | 6 | == 0 | false |
| stability | max_scenario_mean_all_noop_episode_rate | 0.533333 | <= 0.02 | false |
| mission | heterogeneity_unassigned_leak_reduction | 0.171600 | >= 0.15 | true |
| mission | heterogeneity_high_threat_leak_mean_reduction | -0.047181 | >= 0.02 and >= 2/3 seeds | false |
| noninferiority | medium_reward_delta | -16.641306 | >= -5.0 | false |
| noninferiority | medium_damage_delta | 0.253694 | <= 0.1 | false |
| noninferiority | time_pressure_resource_cost_delta | 2.790000 | <= 0.5 | false |
| noninferiority | heterogeneity_damage_delta | 0.107942 | <= 0.1 | false |
| calibration | max_absolute_stochastic_engagement_gap | 0.403730 | <= 0.05 | false |
| efficiency | decision_latency_increase_vs_task10 | 0.692264 | <= 0.25 | false |
| external | external_medium_reward | -42.449988 | >= -5.0 | false |
| external | external_time_pressure_reward | -45.899512 | >= -5.0 | false |
| external | external_medium_damage | 1.064551 | <= 0.1 | false |
| external | external_heterogeneity_damage | 0.542728 | <= 0.1 | false |
| external | external_time_pressure_cost | -9.318333 | <= 0.5 | true |
| external | time_pressure_cost_below_discrete_136 | -7.350000 | < 0 | true |

| 3 | false | -81.02 | 0.941 | 0.000 | 0.787 | -0.663 | 1024 |
| 4 | true | -95.06 | 0.000 | 1.000 | 0.589 | 0.430 | 1024 |
| 5 | true | -91.26 | 0.000 | 1.000 | 0.450 | 1.007 | 1024 |
| 6 | false | -66.42 | 0.805 | 0.000 | 0.722 | -0.287 | 4096 |
| 7 | true | -86.67 | 0.000 | 1.000 | 0.385 | 1.310 | 7168 |

## 当前结论

普通 categorical 的总交战概率被多个目标分摊，deterministic argmax 会优先选择单个 no-op。训练随后进一步分叉：成功种子的 no-op margin 转负，失败种子转正并形成 all-noop。因子化候选必须使用先判定交战、再选择目标的分层确定性规则。
